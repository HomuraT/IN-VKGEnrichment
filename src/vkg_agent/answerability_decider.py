"""
Answerability Decider with Iterative Refinement.

Core flow:
1. Round 1: Generate candidates → Execute → Decide (YES/REFINE/FAIL)
   - If REFINE: Execute refined query → Decide again
2. Round 2-N: If FAIL, generate new candidates with Memory Bank → Execute → Decide
3. Final: If still FAIL after max_rounds, return best_guess

Supports iteration with experience learning from failed attempts.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.config.api_and_models import get_api_configuration
from src.config.logging_config import get_logger
from src.tools.json_parser_with_fallback import invoke_with_json_fallback
from src.vkg_agent.vkg_agent import VKGAgent
from src.vkg_agent.memory_bank import IterationMemoryBank
from src.vkg_agent import prompts as _prompts
from src.tools.sparql_utils import (
    sanitize_body_remove_prefix,
    build_prefix_map_for_vkg,
    assemble_full_sparql,
)
from src.tools.json_utils import exec_results_to_json_safes
from langchain_core.callbacks import get_usage_metadata_callback
from src.tools.llm_usage_accumulator import record_call

__all__ = [
    "AnswerabilityDecider",
]

logger = get_logger(__name__)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class RefinedQuery(BaseModel):
    """Refined SPARQL query."""
    prefixes: List[str] = Field(default_factory=list)
    body: str = Field(default="")


class DecisionResult(BaseModel):
    """Result of decision analysis."""
    decision: str = Field(default="FAIL", description="YES, REFINE, or FAIL")
    selected_index: int = Field(default=1, description="Index of selected candidate (1-based: Candidate 1 → index 1)")
    refined_query: Optional[RefinedQuery] = Field(default=None, description="Improved query (for REFINE)")
    insights: str = Field(default="", description="Explanation of decision")


class BestGuessResult(BaseModel):
    """Best guess when all else fails."""
    prefixes: List[str] = Field(default_factory=list)
    body: str = Field(default="")
    rationale: str = Field(default="")


# =============================================================================
# Lightweight Experience Bank (for compatibility)
# =============================================================================

class _EmptyExperienceBank:
    """Empty experience bank for backward compatibility."""
    
    def snapshot(self) -> List[Dict[str, Any]]:
        return []
    
    def export_injection_text(self, **kwargs) -> str:
        return ""


# =============================================================================
# AnswerabilityDecider
# =============================================================================

class AnswerabilityDecider:
    """
    Simplified decider with minimal LLM calls.
    
    Key changes from original:
    - Removed 3-step experience learning (summary → advantage → optimizer)
    - Single decision call with structured output
    - Optional refinement if decision is REFINE
    - Best guess synthesis when all fails
    """

    def __init__(self, agent: VKGAgent, llm_model_key: Optional[str] = None) -> None:
        """Initialize decider."""
        self.agent = agent
        if llm_model_key:
            llm_cfg = get_api_configuration(llm_model_key)
            self.llm: ChatOpenAI = ChatOpenAI(**llm_cfg)
        else:
            self.llm = agent.llm
        
        # Empty experience bank for compatibility
        self._exp_bank = _EmptyExperienceBank()
        
        # 继承 agent 的 prefix_map
        self.prefix_map = getattr(agent, 'prefix_map', {})

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_prefixes_text(self) -> str:
        """Build VKG prefixes text for prompts."""
        prefix_map = build_prefix_map_for_vkg(getattr(self.agent, "vkg_name", None))
        if not prefix_map:
            return "(No VKG prefixes configured)"
        
        lines = []
        for alias, iri in prefix_map.items():
            if alias == ":":
                lines.append(f"PREFIX : <{iri}>")
            else:
                lines.append(f"PREFIX {alias}: <{iri}>")
        return "\n".join(lines)

    def _format_exec_results(self, exec_results: List[Dict[str, Any]], candidates: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Format execution results for prompt.
        
        If candidates with exec_result_preview are provided, use those.
        Otherwise fall back to exec_results DataFrame format.
        """
        if not exec_results:
            return "(No execution results)"
        
        from src.tools.uri_shortener import shorten_text
        
        lines = []
        for idx, item in enumerate(exec_results):
            query = str(item.get("query", ""))
            body_only = sanitize_body_remove_prefix(query)
            
            # Try to get preview from candidates first
            preview = None
            if candidates and idx < len(candidates):
                preview = candidates[idx].get("exec_result_preview")
            
            if preview:
                # Use preview format (more concise)
                status = preview.get("status", "UNKNOWN")
                query_type = preview.get("query_type", "SELECT")
                
                lines.append(f"\n--- Candidate {idx + 1} (Status: {status}) ---")
                lines.append(f"Body:\n```sparql\n{body_only}\n```")
                
                # Handle different query types
                if query_type == "ASK":
                    # ASK query result
                    ask_result = preview.get("ask_result", False)
                    explanation = preview.get("explanation", "")
                    lines.append(f"Result: {explanation}")
                elif query_type in ["CONSTRUCT", "DESCRIBE"]:
                    # CONSTRUCT/DESCRIBE query result
                    if "raw_result_preview" in preview:
                        raw_preview = preview.get("raw_result_preview", "")
                        triple_count = preview.get("estimated_triple_count", 0)
                        lines.append(f"Result: {triple_count} triples returned (text format)")
                        lines.append(f"Preview:\n{raw_preview}")
                    elif "sample_rows" in preview:
                        row_count = preview.get("row_count", 0)
                        cols = preview.get("columns", [])
                        sample_rows = preview.get("sample_rows", [])
                        lines.append(f"Result: {row_count} rows, columns: {cols}")
                        for r_idx, row_dict in enumerate(sample_rows):
                            lines.append(f"  Row {r_idx + 1}: {row_dict}")
                else:
                    # SELECT query result (default)
                    row_count = preview.get("row_count", 0)
                    cols = preview.get("columns", [])
                    sample_rows = preview.get("sample_rows", [])
                    # 缩短列名中的 URI
                    cols_short = [shorten_text(str(c), self.prefix_map) for c in cols]
                    lines.append(f"Result: {row_count} rows, columns: {cols_short}")
                    if sample_rows:
                        for r_idx, row_dict in enumerate(sample_rows):
                            # 缩短行数据中的 URI
                            row_short = {shorten_text(str(k), self.prefix_map): shorten_text(str(v), self.prefix_map) 
                                        for k, v in row_dict.items()}
                            lines.append(f"  Row {r_idx + 1}: {row_short}")
                
                if preview.get("error_msg"):
                    lines.append(f"Error: {preview['error_msg']}")
            else:
                # Fall back to original DataFrame format
                status = str(item.get("status", "UNKNOWN"))
                df = item.get("dataframe") if item.get("dataframe") is not None else item.get("result")
                
                lines.append(f"\n--- Candidate {idx + 1} (Status: {status}) ---")
                lines.append(f"Body:\n```sparql\n{body_only}\n```")
                
                if isinstance(df, pd.DataFrame):
                    if not df.empty:
                        cols = list(df.columns)
                        # 缩短列名中的 URI
                        cols_short = [shorten_text(str(c), self.prefix_map) for c in cols]
                        lines.append(f"Result: {len(df)} rows, columns: {cols_short}")
                        # Show first 3 rows
                        for r_idx, (_, row) in enumerate(df.head(3).iterrows()):
                            row_dict = {shorten_text(str(c), self.prefix_map): shorten_text(str(row[c])[:50], self.prefix_map) 
                                       for c in cols}
                            lines.append(f"  Row {r_idx + 1}: {row_dict}")
                    else:
                        cols = list(df.columns) if hasattr(df, 'columns') else []
                        cols_short = [shorten_text(str(c), self.prefix_map) for c in cols]
                        lines.append(f"Result: 0 rows (empty), columns: {cols_short}")
                else:
                    lines.append("Result: Error or non-tabular response")
        
        return "\n".join(lines)

    def _format_triples(self, triples_pack: Optional[Dict[str, Any]]) -> str:
        """Format triples for prompt."""
        if not triples_pack:
            return "(No triples retrieved)"
        
        merged = triples_pack.get("merged", [])
        if not merged:
            return "(No merged triples)"
        
        from src.tools.uri_shortener import shorten_uri, shorten_text
        
        lines = []
        for idx, md in enumerate(merged[:5]):  # Limit to 5
            if not isinstance(md, dict):
                continue
            
            subject = md.get("subject_uri", "")
            # 缩短 subject URI
            subject_short = shorten_uri(subject, self.prefix_map)
            brief = ""
            
            # Parse brief from descriptions_json
            if "descriptions_json" in md:
                desc_raw = md["descriptions_json"]
                if isinstance(desc_raw, str) and desc_raw.strip().startswith("{"):
                    desc_dict = json.loads(desc_raw)
                    brief = desc_dict.get("brief", "")
            if not brief:
                brief = md.get("brief_description", "") or md.get("brief", "")
            
            # Parse po pairs
            po_pairs = []
            if "original_data_json" in md:
                orig_raw = md["original_data_json"]
                if isinstance(orig_raw, str) and orig_raw.strip().startswith("{"):
                    orig_dict = json.loads(orig_raw)
                    # original_data_json 直接包含 {"po": [...], "triples_count": ...} 结构
                    po_data = orig_dict.get("po", [])
                    for po in po_data[:3]:
                        if isinstance(po, (list, tuple)) and len(po) >= 2:
                            po_pairs.append((str(po[0]), str(po[1])))
            
            lines.append(f"\n- Subject: {subject_short}")
            if brief:
                lines.append(f"  Brief: {brief}")
            if po_pairs:
                lines.append("  Properties:")
                for p, o in po_pairs:
                    # 缩短 predicate 和 object 中的 URI
                    p_short = shorten_uri(p, self.prefix_map)
                    o_short = shorten_text(o, self.prefix_map)
                    lines.append(f"    • {p_short} → {o_short}")
        
        return "\n".join(lines) if lines else "(No triples to display)"

    def _summarize_exec_result(self, exec_item: Dict[str, Any]) -> str:
        """
        Summarize execution result for best_guess prompt.
        
        Returns concise summary like:
        - "0 rows (NO_ROWS)"
        - "5 rows, 3 columns"
        - "ASK: TRUE"
        - "28 triples returned"
        - "ERROR: syntax error"
        """
        status = exec_item.get("status", "UNKNOWN")
        query_type = exec_item.get("query_type", "SELECT")
        
        if status != "OK":
            error_msg = exec_item.get("error_msg", "Unknown error")
            return f"ERROR: {error_msg[:100]}"
        
        # Check for preview first
        if "exec_result_preview" in exec_item:
            preview = exec_item["exec_result_preview"]
            
            if query_type == "ASK":
                ask_result = preview.get("ask_result", False)
                return f"ASK: {ask_result}"
            elif query_type in ["CONSTRUCT", "DESCRIBE"]:
                triple_count = preview.get("estimated_triple_count", 0)
                return f"{triple_count} triples returned"
            else:  # SELECT
                row_count = preview.get("row_count", 0)
                col_count = len(preview.get("columns", []))
                if row_count == 0:
                    return "0 rows (NO_ROWS)"
                return f"{row_count} rows, {col_count} columns"
        
        # Fall back to DataFrame
        df = exec_item.get("dataframe") if exec_item.get("dataframe") is not None else exec_item.get("result")
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return "0 rows (NO_ROWS)"
            return f"{len(df)} rows, {len(df.columns)} columns"
        
        return "Non-tabular result"
    
    def _extract_query_patterns(self, query_body: str) -> str:
        """
        Extract key patterns from query body for failure analysis.
        
        Returns string like:
        "URIs: obo:Z4, obo:UBERON_123; Labels: 'tonsil'@en; Filters: REGEX"
        """
        patterns = []
        
        # Extract hardcoded URIs (like obo:Z4, uberon:UBERON_123)
        import re
        uri_matches = re.findall(r'\b([a-z]+:[A-Z_][A-Za-z0-9_]+)\b', query_body)
        if uri_matches:
            unique_uris = list(set(uri_matches))[:5]  # Limit to 5
            patterns.append(f"Hardcoded URIs: {', '.join(unique_uris)}")
        
        # Extract label patterns (like "tonsil"@en, "Z4")
        label_matches = re.findall(r'rdfs:label\s+["\']([^"\']+)["\'](?:@\w+)?', query_body, re.IGNORECASE)
        if label_matches:
            unique_labels = list(set(label_matches))[:5]
            patterns.append(f"Label search: {', '.join(unique_labels)}")
        
        # Check for REGEX
        if 'REGEX' in query_body.upper():
            patterns.append("Uses REGEX")
        
        # Check for aggregations
        agg_funcs = re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', query_body, re.IGNORECASE)
        if agg_funcs:
            patterns.append(f"Aggregation: {', '.join(set(agg_funcs))}")
        
        # Check for property paths (rdfs:subClassOf*, part_of+)
        prop_paths = re.findall(r'(\w+:\w+)[\*\+]', query_body)
        if prop_paths:
            patterns.append(f"Property paths: {', '.join(set(prop_paths))}")
        
        return " | ".join(patterns) if patterns else "No special patterns"

    # =========================================================================
    # Core Decision Methods
    # =========================================================================

    def _decide_once(
        self,
        question: str,
        exec_results: List[Dict[str, Any]],
        triples_pack: Optional[Dict[str, Any]],
        candidates: Optional[List[Dict[str, Any]]] = None,
        require_select: bool = False,
        context: Optional[Dict[str, Any]] = None,
        memory_text: str = "",
    ) -> DecisionResult:
        """
        Single decision call with structured output.
        
        Args:
            question: User question
            exec_results: Execution results
            triples_pack: Triples data
            candidates: Candidate queries with previews
            require_select: If True, only consider SELECT queries for YES decision
            context: Retrieved context (ontology/mappings/triples)
        """
        prefixes_text = self._build_prefixes_text()
        exec_results_text = self._format_exec_results(exec_results, candidates=candidates)
        triples_text = self._format_triples(triples_pack)
        
        # Format context using agent's formatter
        ontology_text = "(No ontology elements retrieved)"
        mappings_text = "(No mappings retrieved)"
        triples_context_text = "(No sample triples retrieved)"
        
        if context and hasattr(self.agent, '_format_context_for_prompt'):
            ctx = context.get("context", {})
            if ctx:
                formatted_dict = self.agent._format_context_for_prompt(ctx)
                # Extract text from dict keys
                ontology_text = formatted_dict.get("ontology_text", ontology_text)
                mappings_text = formatted_dict.get("mappings_text", mappings_text)
                triples_context_text = formatted_dict.get("triples_text", triples_context_text)
        
        prompt_template = PromptTemplate.from_template(
            _prompts.decide_and_refine_json,
            template_format="jinja2"
        )
        prompt_text = prompt_template.format(
            question=question,
            prefixes_text=prefixes_text,
            ontology_text=ontology_text,
            mappings_text=mappings_text,
            triples_text=triples_context_text,
            exec_results_text=exec_results_text,
            memory_text=memory_text,
            require_select_note="**IMPORTANT**: Only SELECT queries can be chosen as final answer." if require_select else "",
        )
        
        # Use JSON parser with structured output fallback
        result, usage_meta, output_text = invoke_with_json_fallback(
            self.llm,
            prompt_text,
            DecisionResult,
            operation_name="decider.decide_once"
        )
        
        record_call(
            "decider.decide_once",
            getattr(self.llm, "model", None),
            usage_meta,
            input_text=prompt_text,
            output_text=output_text
        )
        
        # Filter out non-SELECT if require_select is True
        if require_select and result.decision.upper() == "YES" and candidates:
            selected_idx = result.selected_index - 1  # Convert 1-based to 0-based for array access
            if 0 <= selected_idx < len(candidates):
                selected_cand = candidates[selected_idx]
                query_type = selected_cand.get("query_type", "SELECT")
                if query_type != "SELECT":
                    logger.debug(f"[Decider] Rejected {query_type} query at Candidate {result.selected_index}, require_select=True")
                    result.decision = "FAIL"
                    result.insights += f" (Rejected {query_type} query, final answer must be SELECT)"
        
        return result

    def _best_guess(
        self,
        question: str,
        exec_results: List[Dict[str, Any]],
        triples_pack: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> BestGuessResult:
        """Generate best guess when decision fails."""
        prefixes_text = self._build_prefixes_text()
        
        # Build detailed context using agent's formatter
        ctx = context.get("context", {})
        if ctx and hasattr(self.agent, '_format_context_for_prompt'):
            # Use agent's formatter to get detailed ontology/mappings/triples
            formatted = self.agent._format_context_for_prompt(ctx)
            ontology_text = formatted.get('ontology_text', '(No ontology elements retrieved)')
            mappings_text = formatted.get('mappings_text', '(No VKG mappings retrieved)')
            triples_text = formatted.get('triples_text', '(No triples retrieved)')
        else:
            # Fallback: minimal context (should not happen in normal flow)
            ontology_text = "(No ontology elements retrieved)"
            mappings_text = "(No VKG mappings retrieved)"
            triples_text = "(No triples retrieved)"
        
        # Build detailed history from exec results
        history_lines = []
        for idx, item in enumerate(exec_results or []):
            query = str(item.get("query", ""))
            status = str(item.get("status", ""))
            query_type = item.get("query_type", "UNKNOWN")
            body = sanitize_body_remove_prefix(query)
            
            # Extract execution result details
            result_summary = self._summarize_exec_result(item)
            
            # Extract key patterns (predicates, URIs, filters)
            patterns = self._extract_query_patterns(body)
            
            # Show full query body for code-level debug (no prefix, just body)
            history_lines.append(
                f"Candidate {idx + 1} ({query_type}, {status}):\n"
                f"  Result: {result_summary}\n"
                f"  Key Patterns: {patterns}\n"
                f"  Query Body:\n"
                f"  ```sparql\n  {body.strip()}\n  ```"
            )
        
        history_text = "\n\n".join(history_lines) if history_lines else "(No execution history)"
        
        prompt_template = PromptTemplate.from_template(
            _prompts.best_guess_json,
            template_format="jinja2"
        )
        prompt_text = prompt_template.format(
            question=question,
            prefixes_text=prefixes_text,
            ontology_text=ontology_text,
            mappings_text=mappings_text,
            triples_text=triples_text,
            history_text=history_text,
        )
        
        # Use JSON parser with structured output fallback
        result, usage_meta, output_text = invoke_with_json_fallback(
            self.llm,
            prompt_text,
            BestGuessResult,
            operation_name="decider.best_guess"
        )
        
        record_call(
            "decider.best_guess",
            getattr(self.llm, "model", None),
            usage_meta,
            input_text=prompt_text,
            output_text=output_text
        )
        
        return result

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def decide_from_agent_result(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic decision without iteration.
        
        Returns dict with:
        - decision: YES/NO
        - sparql: Selected or empty string
        - candidates: Original candidates
        """
        question = str(agent_result.get("question", ""))
        exec_results = agent_result.get("sparql_exec_results", [])
        triples_pack = agent_result.get("triples_retrieval")
        queries = agent_result.get("queries", {})
        candidate_sparqls = queries.get("sparql_queries", [])
        candidates = agent_result.get("candidates_with_preview", [])
        context = agent_result.get("metadata", {})
        
        # Make decision
        decision_result = self._decide_once(question, exec_results, triples_pack, candidates=candidates, context=context, memory_text="")
        
        # Process result
        decision = decision_result.decision.upper()
        selected_sparql = ""
        
        if decision == "YES":
            selected_idx = decision_result.selected_index - 1  # Convert 1-based to 0-based
            if 0 <= selected_idx < len(candidate_sparqls):
                selected_sparql = candidate_sparqls[selected_idx]
        elif decision == "REFINE" and decision_result.refined_query:
            body = decision_result.refined_query.body
            if body.strip():
                prefix_map = build_prefix_map_for_vkg(self.agent.vkg_name)
                selected_sparql = assemble_full_sparql(prefix_map, sanitize_body_remove_prefix(body))
                decision = "YES"  # Treat refined as YES with new query
        
        effective_decision = "YES" if selected_sparql else "NO"
        
        return {
            "decision": effective_decision,
            "sparql": selected_sparql,
            "candidates": candidate_sparqls,
            "hypothetical_text_queries": queries.get("hypothetical_text_queries", []),
            "triples": triples_pack,
            "key_insights": [decision_result.insights] if decision_result.insights else [],
            "next_queries": [],
        }

    def decide_and_execute_from_agent_result(
        self,
        agent_result: Dict[str, Any],
        method: str = "post",
        accept_format: str = "application/json",
    ) -> Dict[str, Any]:
        """Decision with optional execution."""
        result = self.decide_from_agent_result(agent_result)
        
        if result.get("decision") == "YES" and result.get("sparql"):
            # Try to find existing DataFrame
            exec_results = agent_result.get("sparql_exec_results", [])
            selected_sparql = result["sparql"]
            
            for item in exec_results:
                if str(item.get("query", "")).strip() == selected_sparql.strip():
                    df = item.get("dataframe") if item.get("dataframe") is not None else item.get("result")
                    if isinstance(df, pd.DataFrame):
                        result["dataframe"] = df
                        break
            
            # Execute if no DataFrame found
            if "dataframe" not in result:
                df = self.agent.execute_sparql_df(selected_sparql, method=method, accept_format=accept_format)
                result["dataframe"] = df
        
        return result

    def _generate_reasoning_explanation(
        self,
        question: str,
        final_decision: str,
        final_sparql: str,
        memory_bank: IterationMemoryBank,
        context: Dict[str, Any],
        experience_trace: List[Dict],
    ) -> str:
        """
        使用 1 次 LLM 调用生成推理解释（Markdown 格式）
        
        Args:
            question: 用户问题
            final_decision: 最终决策（YES/GUESS）
            final_sparql: 最终 SPARQL 查询
            memory_bank: 记忆库
            context: 检索到的上下文
            experience_trace: 经验追踪
        
        Returns:
            str: Markdown 格式的推理解释文本
        """
        # Format inputs
        memory_summary = memory_bank.export_formatted_text()
        
        ctx = context.get("context", {})
        if ctx and hasattr(self.agent, '_format_context_for_prompt'):
            formatted = self.agent._format_context_for_prompt(ctx)
            ontology_text = formatted.get('ontology_text', '(No ontology)')
            mappings_text = formatted.get('mappings_text', '(No mappings)')
            triples_text = formatted.get('triples_text', '(No triples)')
        else:
            ontology_text = "(No ontology)"
            mappings_text = "(No mappings)"
            triples_text = "(No triples)"
        
        import json
        experience_trace_text = json.dumps(experience_trace, ensure_ascii=False, indent=2)
        
        # Build prompt
        prompt_template = PromptTemplate.from_template(
            _prompts.generate_reasoning_explanation_markdown,
            template_format="jinja2"
        )
        prompt_text = prompt_template.format(
            question=question,
            final_decision=final_decision,
            final_sparql=final_sparql,
            memory_summary=memory_summary,
            ontology_text=ontology_text,
            mappings_text=mappings_text,
            triples_text=triples_text,
            experience_trace=experience_trace_text,
        )
        
        # Direct LLM call without JSON parsing - get raw Markdown text
        with get_usage_metadata_callback() as cb:
            response = self.llm.invoke(prompt_text)
        
        # Extract text content
        if hasattr(response, 'content'):
            output_text = response.content
        else:
            output_text = str(response)
        
        # Record the call with proper usage metadata
        usage_meta = cb.usage_metadata if cb and hasattr(cb, 'usage_metadata') else {
            "input_tokens": 0, 
            "output_tokens": 0, 
            "total_tokens": 0
        }
        record_call(
            "decider.generate_reasoning_explanation",
            getattr(self.llm, "model", None),
            usage_meta,
            input_text=prompt_text,
            output_text=output_text
        )
        
        return output_text

    def decide_learn_iteratively(
        self,
        agent_result: Dict[str, Any],
        max_rounds: int = 3,
        exec_sparql: bool = True,
        k: int = 3,
    ) -> Dict[str, Any]:
        """
        Iterative decision with Memory Bank learning.
        
        Flow:
        1. Round 1: Decide on initial candidates
           - If YES: return
           - If REFINE: execute + decide again
             - If YES: return
             - If FAIL: continue to Round 2 (skip best_guess)
        2. Round 2 to max_rounds: Generate new candidates with memory
           - Update Memory Bank with previous round
           - Generate k new candidates using memory
           - Decide
           - If YES: return
        3. Final: If all rounds fail, return best_guess
        
        Args:
            agent_result: Result from VKGAgent.run()
            max_rounds: Maximum iteration rounds (from ITER_ROUNDS env var)
            exec_sparql: Whether to execute SPARQL
            k: Number of candidates per round
        """
        logger.debug(f"[Decider] Start decide_learn_iteratively (max_rounds={max_rounds})")
        question = str(agent_result.get("question", ""))
        exec_results = agent_result.get("sparql_exec_results", []) or []
        triples_pack = agent_result.get("triples_retrieval")
        queries = agent_result.get("queries", {})
        candidate_sparqls = queries.get("sparql_queries", [])
        context = agent_result.get("metadata", {})
        
        # Initialize Memory Bank
        memory_bank = IterationMemoryBank(question, prefix_map=self.prefix_map)
        
        # Get candidates with exec_result_preview
        candidates = agent_result.get("candidates_with_preview", [])
        
        trace: List[Dict[str, Any]] = []
        selected_sparql = ""
        current_round = 1
        
        # ===========================================================================
        # Round 1: Initial Decision
        # ===========================================================================
        logger.debug("[Decider] Round 1: Initial decision")
        decision_result = self._decide_once(question, exec_results, triples_pack, candidates=candidates, require_select=True, context=context, memory_text="")
        decision = decision_result.decision.upper()
        
        # Record to memory bank
        memory_bank.add_round(
            round_num=1,
            candidates=candidates,
            exec_results=exec_results,
            decision=decision,
            insights=decision_result.insights,
            selected_index=decision_result.selected_index if decision == "YES" else -1,
        )
        
        trace.append({
            "round": 1,
            "decision": decision,
            "insights": decision_result.insights,
            "num_candidates": len(candidates),
        })
        
        # Handle YES in Round 1
        if decision == "YES":
            selected_idx = decision_result.selected_index - 1  # Convert 1-based to 0-based
            if 0 <= selected_idx < len(candidate_sparqls):
                selected_sparql = candidate_sparqls[selected_idx]
                logger.debug(f"[Decider] Round 1: YES, selected Candidate {decision_result.selected_index}")
            current_round += 1  # Mark as completed
        
        # Handle REFINE in Round 1
        elif decision == "REFINE" and decision_result.refined_query and decision_result.refined_query.body.strip():
            body = decision_result.refined_query.body
            prefix_map = build_prefix_map_for_vkg(self.agent.vkg_name)
            refined_sparql = assemble_full_sparql(prefix_map, sanitize_body_remove_prefix(body))
            
            logger.debug("[Decider] Round 1: REFINE - executing refined query")
            
            if exec_sparql and self.agent.ontop_client:
                # Execute refined query
                new_exec_results = self.agent.execute_sparql_batch([refined_sparql], method="post")
                
                # Construct refined candidate object (with purpose from insights)
                refined_candidate = {
                    "text": "Refined query from decision analysis",
                    "sparql": refined_sparql,
                    "purpose": decision_result.insights[:100] if decision_result.insights else "Query refinement",
                    "query_type": "SELECT",  # REFINE usually produces SELECT
                    "body": body,
                }
                
                # Attach execution preview
                if new_exec_results:
                    query_type = refined_candidate.get("query_type", "SELECT")
                    preview = self.agent._make_exec_preview(new_exec_results[0], query_type)
                    refined_candidate["exec_result_preview"] = preview
                    refined_candidate["exec_result"] = new_exec_results[0]
                
                # Update Memory Bank: append refined candidate to Round 1
                if memory_bank.rounds and len(memory_bank.rounds) > 0:
                    round_1_record = memory_bank.rounds[-1]  # Latest record (should be Round 1)
                    round_1_record["candidates"].append(refined_candidate)
                    round_1_record["exec_results"].extend(new_exec_results)
                
                # Update global lists
                exec_results = exec_results + new_exec_results
                candidate_sparqls.append(refined_sparql)
                candidates.append(refined_candidate)  # Add to candidates list for trace
                
                # Record refined attempt
                trace.append({
                    "round": "1-refine",
                    "refined_query": refined_sparql[:100] + "...",
                    "status": new_exec_results[0].get("status") if new_exec_results else "ERROR",
                })
                
                # Check if refined succeeded (non-empty result)
                if new_exec_results and new_exec_results[0].get("status") == "OK":
                    df = new_exec_results[0].get("dataframe")
                    if df is not None and not (hasattr(df, "empty") and df.empty):
                        selected_sparql = refined_sparql
                        decision = "YES"
                        logger.debug("[Decider] Round 1: REFINE succeeded")
                        current_round += 1
                        
            # If REFINE failed, continue to Round 2 with updated Memory Bank
        
        # ===========================================================================
        # Round 2 to max_rounds: Iterative Generation with Memory
        # ===========================================================================
        while not selected_sparql and current_round < max_rounds:
            current_round += 1
            logger.debug(f"[Decider] Round {current_round}: Generating new candidates with memory")
            
            # Export memory as formatted text
            memory_text = memory_bank.export_formatted_text()
            
            # Generate retrieval queries from TWO sources
            # Priority 0: Purpose from exploratory queries (most specific intent)
            purpose_queries = memory_bank.extract_purpose_queries(max_queries=3)
            # Priority 1: Insights from decision analysis (decision feedback)
            insights_queries = memory_bank.extract_insights_queries(max_queries=2)
            
            all_retrieval_queries = purpose_queries + insights_queries
            logger.debug(f"[Decider] Round {current_round}: Purpose queries: {purpose_queries}")
            logger.debug(f"[Decider] Round {current_round}: Insights queries: {insights_queries}")
            
            # Perform additional retrieval if queries suggested
            enriched_context = context.get("context", {})
            if all_retrieval_queries and hasattr(self.agent, '_retrieve_context'):
                logger.debug(f"[Decider] Round {current_round}: Retrieving additional context")
                try:
                    # Get initial retrieval config from metadata
                    retrieval_config = context.get("retrieval_config", {})
                    initial_ont_k = retrieval_config.get("ontology_k", 10)
                    initial_map_k = retrieval_config.get("mappings_k", 10)
                    initial_tri_k = retrieval_config.get("triples_k", 5)
                    
                    # Use 30-50% of initial k for additional retrieval
                    max_new_ont = max(3, int(initial_ont_k * 0.3))
                    max_new_map = max(3, int(initial_map_k * 0.3))
                    max_new_tri = max(2, int(initial_tri_k * 0.3))
                    
                    # Build two compound queries
                    purpose_compound = " | ".join(purpose_queries) if purpose_queries else ""
                    insights_compound = " | ".join(insights_queries) if insights_queries else ""
                    
                    # Helper function: retrieve with scores
                    def retrieve_with_scores(retriever, query: str, k: int) -> List[Tuple[Any, float]]:
                        """使用 similarity_search_with_score 检索"""
                        if not retriever or not query.strip():
                            return []
                        try:
                            return retriever.similarity_search_with_score(query, k=k) or []
                        except Exception as e:
                            logger.warning(f"Retrieval with scores failed: {e}")
                            return []
                    
                    # Perform dual-source retrieval
                    import json
                    
                    # Ontology oversample factor (for whitelist filtering)
                    ont_oversample_factor = 5 if (hasattr(self.agent, 'concept_whitelist') and self.agent.concept_whitelist) else 1
                    
                    # Dual-source retrieval for ontology
                    purpose_ont_results = retrieve_with_scores(
                        self.agent.ontology_retriever, 
                        purpose_compound, 
                        max_new_ont * ont_oversample_factor
                    ) if purpose_compound else []
                    
                    insights_ont_results = retrieve_with_scores(
                        self.agent.ontology_retriever, 
                        insights_compound, 
                        max_new_ont * ont_oversample_factor
                    ) if insights_compound else []
                    
                    # Dual-source retrieval for mappings
                    purpose_map_results = retrieve_with_scores(
                        self.agent.mappings_retriever, 
                        purpose_compound, 
                        max_new_map
                    ) if purpose_compound else []
                    
                    insights_map_results = retrieve_with_scores(
                        self.agent.mappings_retriever, 
                        insights_compound, 
                        max_new_map
                    ) if insights_compound else []
                    
                    # Dual-source retrieval for triples
                    purpose_tri_results = retrieve_with_scores(
                        self.agent.triples_retriever, 
                        purpose_compound, 
                        max_new_tri
                    ) if purpose_compound else []
                    
                    insights_tri_results = retrieve_with_scores(
                        self.agent.triples_retriever, 
                        insights_compound, 
                        max_new_tri
                    ) if insights_compound else []
                    
                    # Merge by score
                    def merge_by_score(
                        purpose_results: List[Tuple[Any, float]],
                        insights_results: List[Tuple[Any, float]],
                        existing_items: List[Dict],
                        max_new: int,
                        item_type: str
                    ) -> List[Dict]:
                        """合并两个检索源，按得分排序后去重"""
                        all_results = purpose_results + insights_results
                        all_results.sort(key=lambda x: x[1], reverse=True)
                        
                        # Build dedup key based on type
                        if item_type == "ontology":
                            dedup_key = "uri"
                        elif item_type == "mappings":
                            dedup_key = "mapping_id"
                        else:  # triples
                            dedup_key = "subject_uri"
                        
                        existing_keys = {item.get(dedup_key) for item in existing_items}
                        new_items = []
                        
                        for doc, score in all_results:
                            md = getattr(doc, "metadata", {}) or {}
                            
                            if item_type == "ontology":
                                item = {
                                    "uri": md.get("uri", ""),
                                    "element_type": md.get("element_type", ""),
                                    "_retrieval_score": score,
                                }
                                labels = md.get("labels", [])
                                if labels:
                                    item["label"] = labels[0] if isinstance(labels, list) else str(labels)
                                
                                # 白名单过滤
                                if hasattr(self.agent, 'concept_whitelist') and self.agent.concept_whitelist:
                                    if not self.agent.concept_whitelist.is_allowed(item["uri"]):
                                        continue  # 跳过不在白名单中的概念
                                
                            elif item_type == "mappings":
                                item = {
                                    "mapping_id": md.get("mapping_id", ""),
                                    "target_pattern": md.get("target_pattern", ""),
                                    "source_query": md.get("source_query", ""),
                                    "_retrieval_score": score,
                                }
                                
                            else:  # triples
                                po_pairs = []
                                if "original_data_json" in md:
                                    orig_raw = md["original_data_json"]
                                    if isinstance(orig_raw, str) and orig_raw.strip().startswith("{"):
                                        try:
                                            orig_dict = json.loads(orig_raw)
                                            po_data = orig_dict.get("po", [])
                                            for po in po_data[:5]:
                                                if isinstance(po, list) and len(po) == 2:
                                                    po_pairs.append([str(po[0]), str(po[1])])
                                        except json.JSONDecodeError:
                                            pass
                                
                                item = {
                                    "subject_uri": md.get("subject_uri", ""),
                                    "subject_label": md.get("subject_label", ""),
                                    "po_pairs": po_pairs,
                                    "_retrieval_score": score,
                                }
                            
                            key_value = item.get(dedup_key)
                            if key_value and key_value not in existing_keys:
                                new_items.append(item)
                                existing_keys.add(key_value)
                                
                                if len(new_items) >= max_new:
                                    break
                        
                        return new_items
                    
                    # Merge ontology
                    added_ont = merge_by_score(
                        purpose_ont_results,
                        insights_ont_results,
                        enriched_context.get("ontology_items", []),
                        max_new_ont,
                        "ontology"
                    )
                    enriched_context.setdefault("ontology_items", []).extend(added_ont)
                    
                    # Merge mappings
                    added_map = merge_by_score(
                        purpose_map_results,
                        insights_map_results,
                        enriched_context.get("mapping_items", []),
                        max_new_map,
                        "mappings"
                    )
                    enriched_context.setdefault("mapping_items", []).extend(added_map)
                    
                    # Merge triples
                    added_tri = merge_by_score(
                        purpose_tri_results,
                        insights_tri_results,
                        enriched_context.get("triples_items", []),
                        max_new_tri,
                        "triples"
                    )
                    enriched_context.setdefault("triples_items", []).extend(added_tri)
                    
                    logger.debug(f"[Decider] Round {current_round}: Added {len(added_ont)}/{max_new_ont} ontology, "
                               f"{len(added_map)}/{max_new_map} mappings, {len(added_tri)}/{max_new_tri} triples")
                    
                    trace.append({
                        "round": f"{current_round}-retrieval",
                        "purpose_queries": purpose_queries,
                        "insights_queries": insights_queries,
                        "new_items": {
                            "ontology": len(added_ont),
                            "mappings": len(added_map),
                            "triples": len(added_tri),
                        },
                        "limits": {
                            "ontology": max_new_ont,
                            "mappings": max_new_map,
                            "triples": max_new_tri,
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"[Decider] Round {current_round}: Additional retrieval failed: {e}")
            
            # Generate new candidates using enriched context and memory
            try:
                new_candidates = self.agent._generate_candidates(
                    question=question,
                    context=enriched_context,
                    k=k,
                    memory_text=memory_text,
                )
                
                # Execute new candidates
                if exec_sparql and self.agent.ontop_client and new_candidates:
                    new_sparqls = [c["sparql"] for c in new_candidates]
                    new_exec_results = self.agent.execute_sparql_batch(new_sparqls, method="post")
                    exec_results.extend(new_exec_results)
                    candidate_sparqls.extend(new_sparqls)
                    
                    # Attach previews
                    for idx, cand in enumerate(new_candidates):
                        if idx < len(new_exec_results):
                            query_type = cand.get("query_type", "SELECT")
                            preview = self.agent._make_exec_preview(new_exec_results[idx], query_type)
                            cand["exec_result_preview"] = preview
                            cand["exec_result"] = new_exec_results[idx]
                    
                    # Make decision on new candidates
                    # In the last round, require SELECT queries only
                    is_last_round = (current_round >= max_rounds - 1)
                    decision_result = self._decide_once(
                        question, 
                        new_exec_results, 
                        triples_pack, 
                        candidates=new_candidates,
                        require_select=is_last_round,
                        context=context,
                        memory_text=memory_text
                    )
                    decision = decision_result.decision.upper()
                    
                    # Record to memory
                    memory_bank.add_round(
                        round_num=current_round,
                        candidates=new_candidates,
                        exec_results=new_exec_results,
                        decision=decision,
                        insights=decision_result.insights,
                        selected_index=decision_result.selected_index if decision == "YES" else -1,
                    )
                    
                    trace.append({
                        "round": current_round,
                        "decision": decision,
                        "insights": decision_result.insights[:200] if decision_result.insights else "",
                        "num_candidates": len(new_candidates),
                    })
                    
                    # Check if found answer
                    if decision == "YES":
                        selected_idx = decision_result.selected_index - 1  # Convert 1-based to 0-based
                        if 0 <= selected_idx < len(new_sparqls):
                            selected_sparql = new_sparqls[selected_idx]
                            logger.debug(f"[Decider] Round {current_round}: YES, selected Candidate {decision_result.selected_index}")
                        break
                    
                    # Handle REFINE in iteration rounds
                    elif decision == "REFINE" and decision_result.refined_query and decision_result.refined_query.body.strip():
                        body = decision_result.refined_query.body
                        prefix_map = build_prefix_map_for_vkg(self.agent.vkg_name)
                        refined_sparql = assemble_full_sparql(prefix_map, sanitize_body_remove_prefix(body))
                        
                        refine_exec = self.agent.execute_sparql_batch([refined_sparql], method="post")
                        if refine_exec and refine_exec[0].get("status") == "OK":
                            df = refine_exec[0].get("dataframe")
                            if df is not None and not (hasattr(df, "empty") and df.empty):
                                selected_sparql = refined_sparql
                                candidate_sparqls.append(refined_sparql)
                                exec_results.extend(refine_exec)
                                logger.debug(f"[Decider] Round {current_round}: REFINE succeeded")
                                break
                
            except Exception as e:
                logger.error(f"[Decider] Round {current_round}: Error generating candidates: {e}")
                trace.append({
                    "round": current_round,
                    "error": str(e)[:200],
                })
                break
        
        # ===========================================================================
        # Final Output
        # ===========================================================================
        if selected_sparql:
            # Found a good query
            out = {
                "decision": "YES",
                "sparql": selected_sparql,
                "candidates": candidate_sparqls,
                "hypothetical_text_queries": queries.get("hypothetical_text_queries", []),
                "triples": triples_pack,
                "key_insights": [],
                "next_queries": [],
                "experiences_snapshot": memory_bank.snapshot(),
                "experience_trace": trace,
            }
            
            # Try to attach DataFrame
            for item in exec_results:
                if str(item.get("query", "")).strip() == selected_sparql.strip():
                    df = item.get("dataframe") if item.get("dataframe") is not None else item.get("result")
                    if isinstance(df, pd.DataFrame):
                        out["dataframe"] = df
                        break
        else:
            # All rounds failed - generate best guess
            logger.debug(f"[Decider] All {max_rounds} rounds failed, generating best guess")
            best_guess_result = self._best_guess(question, exec_results, triples_pack, context)
            
            best_sparql = ""
            if best_guess_result.body.strip():
                prefix_map = build_prefix_map_for_vkg(self.agent.vkg_name)
                best_sparql = assemble_full_sparql(prefix_map, sanitize_body_remove_prefix(best_guess_result.body))
            
            trace.append({
                "final_synthesis": {
                    "body": best_guess_result.body[:100] + "..." if len(best_guess_result.body) > 100 else best_guess_result.body,
                    "rationale": best_guess_result.rationale,
                }
            })
            
            out = {
                "decision": "GUESS",
                "sparql": "",
                "candidates": candidate_sparqls,
                "hypothetical_text_queries": queries.get("hypothetical_text_queries", []),
                "triples": triples_pack,
                "key_insights": [memory_bank.export_formatted_text()[:500]],  # Include memory summary
                "next_queries": [],
                "best_guess": {
                    "sparql": best_sparql,
                    "body": best_guess_result.body,
                    "rationale": best_guess_result.rationale,
                },
                "experiences_snapshot": memory_bank.snapshot(),
                "experience_trace": trace,
            }
        
        # ===========================================================================
        # Generate Reasoning Explanation (1 LLM call)
        # ===========================================================================
        import os
        enable_explanation = os.getenv("ENABLE_REASONING_EXPLANATION", "true").lower() == "true"
        
        if enable_explanation:
            try:
                # Extract final SPARQL for explanation
                final_sparql_for_explanation = out.get("sparql", "")
                if not final_sparql_for_explanation and out.get("decision") == "GUESS":
                    final_sparql_for_explanation = out.get("best_guess", {}).get("sparql", "")
                
                reasoning_explanation = self._generate_reasoning_explanation(
                    question=question,
                    final_decision=out.get("decision"),
                    final_sparql=final_sparql_for_explanation,
                    memory_bank=memory_bank,
                    context=context,
                    experience_trace=trace,
                )
                out["reasoning_explanation"] = reasoning_explanation
                logger.debug(f"[Decider] Generated reasoning explanation ({len(reasoning_explanation)} chars)")
            except Exception as e:
                logger.error(f"[Decider] Failed to generate reasoning explanation: {e}")
                out["reasoning_explanation"] = f"(Failed to generate explanation: {str(e)[:100]})"
        
        logger.debug(f"[Decider] Finished after {current_round} rounds with decision: {out.get('decision')}")
        return out


# =============================================================================
# Main (for testing)
# =============================================================================

def main() -> None:
    """Demo entry point."""
    print("AnswerabilityDecider - Use with VKGAgent")


if __name__ == "__main__":
    main()
