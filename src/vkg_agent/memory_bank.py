"""
Iteration Memory Bank - Records execution history for iterative refinement.

Stores success/failure patterns across iteration rounds to help LLM:
- Learn from purpose-driven exploration (ASK/CONSTRUCT)
- Follow decision insights
- Avoid repeating failed strategies
"""

from typing import Any, Dict, List, Optional
import json
from src.config.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "IterationMemoryBank",
]


class IterationMemoryBank:
    """
    Records iteration history for a single question.
    
    Lifecycle: Per-sample (initialized in decide_learn_iteratively).
    """
    
    def __init__(self, question: str, prefix_map: Optional[Dict[str, str]] = None):
        """Initialize memory bank for a question."""
        self.question = question
        self.rounds: List[Dict[str, Any]] = []
        self.prefix_map = prefix_map or {}
        
    def add_round(
        self,
        round_num: int,
        candidates: List[Dict[str, Any]],
        exec_results: List[Dict[str, Any]],
        decision: str,
        insights: str = "",
        selected_index: int = -1,
    ) -> None:
        """
        Record one iteration round.
        
        Args:
            round_num: Round number (1-based)
            candidates: List of candidate queries with exec_result_preview
            exec_results: Full execution results
            decision: YES/REFINE/FAIL
            insights: Decision insights
            selected_index: Index of selected candidate (for YES)
        """
        self.rounds.append({
            "round": round_num,
            "decision": decision,
            "insights": insights,
            "selected_index": selected_index,
            "candidates": candidates,
            "exec_results": exec_results,
        })
        
    def extract_purpose_queries(self, max_queries: int = 3) -> List[str]:
        """
        从候选查询的 purpose 中提取检索查询（优先级最高）
        
        策略：
        - 优先收集失败轮次的探索性查询（ASK/CONSTRUCT/DESCRIBE）的 purpose
        - 其次收集包含探索关键词的 SELECT 查询的 purpose
        - 最新的 round 优先
        
        Returns:
            最多 max_queries 个 purpose 文本
        """
        if not self.rounds:
            return []
        
        purpose_queries = []
        seen_purposes = set()
        
        # 逆序遍历（最新的 round 优先）
        for round_info in reversed(self.rounds):
            candidates = round_info.get("candidates", [])
            decision = round_info.get("decision", "")
            
            # 如果该轮失败或 REFINE，重点关注候选的 purpose
            if decision in ["FAIL", "REFINE"]:
                for cand in candidates:
                    purpose = cand.get("purpose", "").strip()
                    query_type = cand.get("query_type", "SELECT")
                    
                    if not purpose or len(purpose) < 10:
                        continue
                    
                    # 去重
                    purpose_lower = purpose.lower()
                    if purpose_lower in seen_purposes:
                        continue
                    
                    # 优先级 1: 探索性查询（ASK/CONSTRUCT/DESCRIBE）
                    if query_type in ["ASK", "CONSTRUCT", "DESCRIBE"]:
                        purpose_queries.append(purpose)
                        seen_purposes.add(purpose_lower)
                        if len(purpose_queries) >= max_queries:
                            return purpose_queries
                    
                    # 优先级 2: SELECT 查询（包含探索性关键词）
                    elif query_type == "SELECT":
                        exploration_keywords = [
                            "discover", "explore", "find", "identify", 
                            "check", "verify", "list", "retrieve", "get"
                        ]
                        if any(kw in purpose_lower for kw in exploration_keywords):
                            purpose_queries.append(purpose)
                            seen_purposes.add(purpose_lower)
                            if len(purpose_queries) >= max_queries:
                                return purpose_queries
        
        return purpose_queries
    
    def extract_insights_queries(self, max_queries: int = 2) -> List[str]:
        """
        从历史 insights 中提取检索查询
        
        策略：直接使用 insights 文本（不做正则匹配）
        
        Returns:
            最多 max_queries 个 insights 文本（截断到 100 字符）
        """
        if not self.rounds:
            return []
        
        insights_queries = []
        seen_insights = set()
        
        # 逆序遍历（最新的 round 优先）
        for round_info in reversed(self.rounds):
            insights_text = round_info.get("insights", "").strip()
            
            if not insights_text or len(insights_text) < 10:
                continue
            
            # 截断到 100 字符（保留完整语义）
            truncated = insights_text[:100].strip()
            truncated_lower = truncated.lower()
            
            # 去重
            if truncated_lower not in seen_insights:
                insights_queries.append(truncated)
                seen_insights.add(truncated_lower)
                
                if len(insights_queries) >= max_queries:
                    break
        
        return insights_queries
    
    def export_formatted_text(self) -> str:
        """
        Export memory as formatted text for LLM prompt.
        
        Format:
        - Round-by-round execution history (query body + status + purpose)
        - Suggested exploration directions (from purpose analysis)
        """
        if not self.rounds:
            return "(No previous attempts)"
        
        from src.tools.uri_shortener import shorten_text
        
        lines = ["## Previous Iteration History", ""]
        
        # Collect patterns for summary
        failed_queries = []
        empty_queries = []
        successful_queries = []
        
        for round_info in self.rounds:
            round_num = round_info["round"]
            decision = round_info["decision"]
            insights = round_info.get("insights", "")
            candidates = round_info.get("candidates", [])
            
            lines.append(f"### Round {round_num}: {decision}")
            if insights:
                lines.append(f"**Decision Insight**: {insights[:300]}...")
            lines.append("")
            
            # Show each candidate with execution result
            for idx, cand in enumerate(candidates):
                body = cand.get("body", "")
                query_type = cand.get("query_type", "SELECT")
                purpose = cand.get("purpose", "")
                preview = cand.get("exec_result_preview", {})
                status = preview.get("status", "UNKNOWN")
                
                # 缩短 SPARQL 查询体中的 URI（仅用于展示，不修改原始查询）
                body_display = shorten_text(body, self.prefix_map)
                
                # Determine execution outcome
                if status == "ERROR_OR_INVALID":
                    outcome = "FAILED (syntax/execution error)"
                    failed_queries.append({"type": query_type, "purpose": purpose})
                elif status == "NO_ROWS":
                    outcome = "Returned 0 rows (empty result)"
                    empty_queries.append({"type": query_type, "purpose": purpose})
                elif status == "OK":
                    row_count = preview.get("row_count", 0)
                    if query_type == "ASK":
                        ask_result = preview.get("ask_result", False)
                        outcome = f"ASK returned: {ask_result}"
                    elif query_type in ["CONSTRUCT", "DESCRIBE"]:
                        triple_count = preview.get("estimated_triple_count", 0)
                        outcome = f"Returned {triple_count} triples"
                    else:
                        outcome = f"Returned {row_count} rows"
                    
                    # Check if selected
                    if decision == "YES" and idx == round_info.get("selected_index", -1):
                        outcome += " - SELECTED AS ANSWER"
                        successful_queries.append({"type": query_type, "purpose": purpose})
                else:
                    outcome = f"Status: {status}"
                
                lines.append(f"- Candidate {idx+1} ({query_type}): {outcome}")
                if purpose:
                    lines.append(f"  Purpose: {purpose}")
                # 使用缩短后的 URI 展示
                lines.append(f"  ```sparql\n  {body_display.strip()[:200]}{'...' if len(body_display) > 200 else ''}\n  ```")
                lines.append("")
        
        # Summary section
        lines.append("### Key Learnings")
        lines.append("")
        
        if successful_queries:
            lines.append("**Successful Strategies**:")
            for i, q in enumerate(successful_queries, 1):
                lines.append(f"  {i}. {q['type']} - {q['purpose']}")
            lines.append("")
        
        if empty_queries:
            lines.append("**Empty Result Queries** (valid syntax but no data):")
            for i, q in enumerate(empty_queries[:3], 1):
                lines.append(f"  {i}. {q['type']} - {q['purpose']}")
            lines.append("")
        
        if failed_queries:
            lines.append("**Failed Queries** (syntax/execution errors):")
            for i, q in enumerate(failed_queries[:3], 1):
                lines.append(f"  {i}. {q['type']} - {q['purpose']}")
            lines.append("")
        
        if not successful_queries:
            lines.append("**No successful queries yet** - consider:")
            lines.append("  - Exploratory queries (ASK, CONSTRUCT) to discover schema")
            lines.append("  - Simpler patterns with fewer joins")
            lines.append("  - Alternative predicates from retrieved context")
            lines.append("")
        
        # Add suggested exploration directions
        purpose_suggestions = self.extract_purpose_queries(max_queries=3)
        if purpose_suggestions:
            lines.append("**Suggested Exploration Directions** (from failed attempts):")
            for i, suggestion in enumerate(purpose_suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_exec_status(self, exec_results: List[Dict[str, Any]], idx: int) -> str:
        """Get execution status for a candidate."""
        if idx >= len(exec_results):
            return "UNKNOWN"
        
        result = exec_results[idx]
        status = result.get("status", "UNKNOWN")
        
        # Normalize status
        if status in ["ERROR_OR_INVALID", "ERROR"]:
            return "ERROR"
        elif status == "NO_ROWS":
            return "EMPTY"
        elif status == "OK":
            return "SUCCESS"
        
        return "UNKNOWN"
    
    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a snapshot of all rounds (for compatibility)."""
        return self.rounds.copy()

