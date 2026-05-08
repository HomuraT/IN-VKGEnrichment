"""
Simplified VKGAgent - Minimal LLM calls with structured output.

Core flow:
1. _retrieve_context(): Single-round retrieval from ontology/mappings/triples (no LLM)
2. _generate_candidates(): Generate k candidate SPARQL queries (1 LLM call)
3. run(): Combine retrieval + generation, return compatible output format
"""

from typing import Any, Dict, Optional, List, Set, Tuple
import pandas as pd
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.config.api_and_models import get_api_configuration
from src.config.logging_config import get_logger

logger = get_logger(__name__)
from src.vkg_agent.retrievers import create_retriever_by_db_name
from src.tools.json_parser_with_fallback import invoke_with_json_fallback
from src.tools.ontop_client import OntopClient
from src.tools.vector_db_paths import build_vector_db_paths
from src.config.prefix_for_vkgs import prefix_for_vkgs as _pfv
from src.tools.sparql_utils import (
    build_prefix_map_for_vkg,
    assemble_full_sparql,
)
from src.vkg_agent import prompts as _prompts
from src.tools.ranking import rrf_merge_from_lists
from langchain_core.callbacks import get_usage_metadata_callback
from src.tools.llm_usage_accumulator import record_call

__all__ = [
    "VKGAgent",
]


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class CandidateQuery(BaseModel):
    """Single candidate query."""
    text: str = Field(default="", description="Natural language description")
    purpose: str = Field(default="", description="What this query tries to achieve")
    query_type: str = Field(default="SELECT", description="SPARQL query type: SELECT, ASK, CONSTRUCT, DESCRIBE")
    prefixes: List[str] = Field(default_factory=list, description="Prefix aliases to use")
    body: str = Field(default="", description="SPARQL body without PREFIX declarations")


class CandidatesOutput(BaseModel):
    """Output of candidate generation."""
    candidates: List[CandidateQuery] = Field(default_factory=list)


# =============================================================================
# Lightweight MPR wrapper for backward compatibility
# =============================================================================

class _LightweightMPR:
    """Minimal MPR-like wrapper that only supports skip_blocks."""
    
    def __init__(self):
        self.skip_blocks: Set[str] = set()
        self.config = None  # Compatibility placeholder


# =============================================================================
# VKGAgent
# =============================================================================

class VKGAgent:
    """
    Simplified VKGAgent with minimal LLM calls.
    
    Key changes from original:
    - Single-round retrieval instead of two-round + rewrite
    - One LLM call to generate k candidates (structured output)
    - Removed MPR orchestration complexity
    - Full po pairs in context for better SPARQL generation
    """

    def __init__(
        self,
        ontology_db_name: str,
        mappings_db_name: str,
        triples_db_name: Optional[str] = None,
        queries_db_name: Optional[str] = None,
        llm_model_key: str = "mmm_beta_gpt_4o_mini",
        ontop_client: Optional[OntopClient] = None,
        vkg_name: Optional[str] = None,
        ontology_file_path: Optional[str] = None,
        obda_file_path: Optional[str] = None,
        enable_concept_filtering: bool = True,
    ) -> None:
        """
        Initialize agent with HTTP-based vector retrievers.
        
        Args:
            ontology_db_name: 本体向量库名称（不含.chroma后缀）
            mappings_db_name: VKG映射向量库名称
            triples_db_name: 聚合三元组向量库名称（可选）
            queries_db_name: Text-to-SPARQL查询向量库名称（可选）
            llm_model_key: LLM模型配置键名
            ontop_client: Ontop SPARQL客户端（可选）
            vkg_name: VKG名称（可选）
            ontology_file_path: 本体文件路径（可选，支持 .ttl/.owl/.rdf/.xml）
            obda_file_path: OBDA 映射文件路径（可选，用于构建本体概念白名单）
            enable_concept_filtering: 是否启用本体概念白名单过滤（默认 True）
        """
        llm_cfg = get_api_configuration(llm_model_key)
        self.llm: ChatOpenAI = ChatOpenAI(**llm_cfg)
        self.ontop_client: Optional[OntopClient] = ontop_client
        self.vkg_name: Optional[str] = vkg_name

        # Build HTTP-based retrievers
        self.memory: Optional[Any] = None
        
        # 根据向量库名称创建 HTTP 检索器
        self.ontology_retriever = create_retriever_by_db_name(ontology_db_name)
        self.mappings_retriever = create_retriever_by_db_name(mappings_db_name)
        
        self.triples_retriever = (
            create_retriever_by_db_name(triples_db_name) if triples_db_name else None
        )
        
        self.queries_retriever = (
            create_retriever_by_db_name(queries_db_name) if queries_db_name else None
        )

        # Lightweight MPR wrapper for backward compatibility
        self._mpr = _LightweightMPR()
        
        # 加载本体文件到内存（缓存）
        self._ontology_graph = None
        from pathlib import Path
        
        # 确定本体文件路径：优先使用显式传入的路径，否则基于 vkg_name 自动构建
        ontology_file: Optional[Path] = None
        if ontology_file_path:
            ontology_file = Path(ontology_file_path)
        elif vkg_name:
            ontology_file = Path(f"resources/vkg_ontologies/{vkg_name}.ttl")
        
        if ontology_file and ontology_file.exists():
            from rdflib import Graph
            self._ontology_graph = Graph()
            # 根据扩展名自动检测格式
            suffix = ontology_file.suffix.lower()
            format_map = {
                ".ttl": "turtle",
                ".owl": "xml",
                ".rdf": "xml",
                ".xml": "xml",
                ".n3": "n3",
                ".nt": "nt",
                ".jsonld": "json-ld",
            }
            rdf_format = format_map.get(suffix, "turtle")
            self._ontology_graph.parse(str(ontology_file), format=rdf_format)
            logger.info(f"Loaded ontology graph: {len(self._ontology_graph)} triples (format={rdf_format})")
        elif ontology_file:
            logger.warning(f"Ontology file not found: {ontology_file}")
        
        # 初始化本体概念白名单（基于 OBDA 映射）
        self.concept_whitelist = None
        if enable_concept_filtering and obda_file_path:
            try:
                from src.tools.ontology_concept_extractor import OntologyConceptWhitelist
                
                # 尝试从 JSON 缓存加载
                obda_path = Path(obda_file_path)
                json_cache = Path("resources/ontology_concept_whitelists") / f"{obda_path.stem}_whitelist.json"
                
                if json_cache.exists():
                    logger.info(f"Loading concept whitelist from cache: {json_cache}")
                    self.concept_whitelist = OntologyConceptWhitelist(
                        json_cache_file=str(json_cache)
                    )
                else:
                    logger.info(f"Building concept whitelist from OBDA: {obda_file_path}")
                    self.concept_whitelist = OntologyConceptWhitelist(
                        obda_file=obda_file_path,
                        ontology_file=ontology_file_path,
                        auto_convert=True,
                        include_builtin=True
                    )
                    # 保存到缓存
                    self.concept_whitelist.save_to_json(str(json_cache))
                
                stats = self.concept_whitelist.get_statistics()
                logger.info(f"Concept whitelist initialized: {stats['total_concepts']} concepts "
                          f"({stats['classes']} classes, {stats['properties']} properties)")
            except Exception as e:
                logger.warning(f"Failed to initialize concept whitelist: {e}")
                self.concept_whitelist = None
        elif enable_concept_filtering and not obda_file_path:
            logger.debug("Concept filtering enabled but no OBDA file provided, skipping whitelist")
        
        # 初始化 URI 缩短器
        from src.tools.uri_shortener import get_prefix_map_for_vkg
        self.prefix_map = get_prefix_map_for_vkg(vkg_name) if vkg_name else {}
        if self.prefix_map:
            logger.info(f"Loaded {len(self.prefix_map)} URI prefixes for VKG: {vkg_name}")

    # =========================================================================
    # Backward Compatibility Methods
    # =========================================================================

    def set_memory(self, memory: Any) -> None:
        """Set memory module (deprecated, kept for compatibility)."""
        self.memory = memory

    def set_ontop_client(self, client: OntopClient) -> None:
        """Set Ontop SPARQL client."""
        self.ontop_client = client

    # =========================================================================
    # Core Methods: Retrieval
    # =========================================================================

    def _query_ontology_properties(self, uri: str) -> Dict[str, List[str]]:
        """
        从本地本体文件查询 URI 的所有属性
        
        Args:
            uri: 本体概念的完整 URI
            
        Returns:
            Dict[str, List[str]]: {predicate_uri: [object_values]}
        """
        if not self._ontology_graph:
            return {}
        
        from rdflib import URIRef
        
        properties = {}
        uri_ref = URIRef(uri)
        
        # 查询该 URI 作为主语的所有三元组
        for pred, obj in self._ontology_graph.predicate_objects(uri_ref):
            pred_str = str(pred)
            obj_str = str(obj)
            
            if pred_str not in properties:
                properties[pred_str] = []
            properties[pred_str].append(obj_str)
        
        return properties

    def _categorize_property(self, pred_uri: str) -> tuple:
        """
        通用属性分类（基于谓词名称模式）
        
        Returns:
            (priority, display_name): priority 越小越优先
        """
        pred_lower = pred_uri.lower()
        short_name = pred_uri.split("#")[-1].split("/")[-1]
        
        if "label" in pred_lower or "name" in pred_lower:
            return (1, "Label")
        elif any(kw in pred_lower for kw in ["comment", "definition", "description"]):
            return (2, "Definition")
        else:
            return (3, short_name)

    def _retrieve_and_filter_ontology(
        self, 
        query: str, 
        k: int,
        oversample_factor: int = 5
    ) -> List[Dict[str, Any]]:
        """
        检索本体概念并应用白名单过滤
        
        Args:
            query: 检索查询
            k: 目标返回数量
            oversample_factor: 过采样倍数（默认5倍）
        
        Returns:
            过滤后的前k个本体概念
        """
        if not self.ontology_retriever:
            return []
        
        # 如果启用白名单，检索 k*oversample_factor 个候选
        retrieval_k = k * oversample_factor if self.concept_whitelist else k
        
        docs = self.ontology_retriever.similarity_search(query, k=retrieval_k) or []
        
        ontology_items = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            item = {
                "uri": md.get("uri", ""),
                "element_type": md.get("element_type", ""),
            }
            # 提取 label
            labels = md.get("labels", [])
            if labels:
                item["label"] = labels[0] if isinstance(labels, list) else str(labels)
            
            # 提取 comment
            if "original_data_json" in md:
                orig_raw = md["original_data_json"]
                if isinstance(orig_raw, str) and orig_raw.strip().startswith("{"):
                    try:
                        orig_dict = json.loads(orig_raw)
                        props = orig_dict.get("properties", {})
                        comment_values = props.get("http://www.w3.org/2000/01/rdf-schema#comment", [])
                        if comment_values:
                            item["comment"] = comment_values[0]
                    except json.JSONDecodeError:
                        pass
            
            # 查询本地本体属性
            if self._ontology_graph:
                props = self._query_ontology_properties(item["uri"])
                if props:
                    item["properties"] = props
            
            ontology_items.append(item)
        
        # 应用白名单过滤
        if self.concept_whitelist:
            original_count = len(ontology_items)
            ontology_items = self.concept_whitelist.filter_items(ontology_items, uri_key='uri')
            filtered_count = original_count - len(ontology_items)
            if filtered_count > 0:
                logger.debug(f"Filtered out {filtered_count}/{original_count} ontology items")
        
        # 取前k个
        return ontology_items[:k]

    def _retrieve_context(
        self, 
        question: str, 
        ontology_k: int = 10,
        mappings_k: int = 10,
        triples_k: int = 10
    ) -> Dict[str, Any]:
        """
        Single-round retrieval from all sources.
        
        Args:
            question: Natural language question
            ontology_k: Number of ontology elements to retrieve
            mappings_k: Number of VKG mappings to retrieve
            triples_k: Number of aggregated triples to retrieve
        
        Returns context dict with:
        - prefixes_text: VKG prefix definitions
        - ontology_items: List of ontology elements
        - mapping_items: List of VKG mappings
        - triples_items: List of aggregated triples with po pairs
        """
        skip_blocks = getattr(self._mpr, "skip_blocks", set()) or set()
        
        # Load VKG prefixes
        prefixes_text = ""
        if self.vkg_name:
            vkg_entry = _pfv.get(self.vkg_name)
            if isinstance(vkg_entry, dict):
                prefix_lines = []
                for alias, iri in vkg_entry.items():
                    alias_str = ":" if alias in (":", "") else alias.rstrip(":")
                    if alias_str and iri:
                        prefix_lines.append(f"PREFIX {alias_str}: <{iri}>")
                prefixes_text = "\n".join(prefix_lines)
        
        # Retrieve ontology elements
        ontology_items: List[Dict[str, Any]] = []
        if self.ontology_retriever and "vkg.ontology" not in skip_blocks:
            ontology_items = self._retrieve_and_filter_ontology(question, ontology_k)
        
        # Retrieve VKG mappings
        mapping_items: List[Dict[str, Any]] = []
        if self.mappings_retriever and "vkg.mappings" not in skip_blocks:
            docs = self.mappings_retriever.similarity_search(question, k=mappings_k) or []
            for d in docs:
                md = getattr(d, "metadata", {}) or {}
                mapping_items.append({
                    "mapping_id": md.get("mapping_id", ""),
                    "target_pattern": md.get("target_pattern", ""),
                    "source_query": md.get("source_query", ""),
                })
        
        # Retrieve aggregated triples (with po pairs)
        triples_items: List[Dict[str, Any]] = []
        if self.triples_retriever and "vkg.aggregated_triples" not in skip_blocks:
            docs = self.triples_retriever.similarity_search(question, k=triples_k) or []
            for d in docs:
                md = getattr(d, "metadata", {}) or {}
                
                # Parse original_data_json for po pairs
                po_pairs: List[List[str]] = []
                if "original_data_json" in md:
                    orig_raw = md["original_data_json"]
                    if isinstance(orig_raw, str) and orig_raw.strip().startswith("{"):
                        orig_dict = json.loads(orig_raw)
                        po_data = orig_dict.get("po", [])
                        # Take first 5 po pairs
                        for po in po_data[:5]:
                            if isinstance(po, (list, tuple)) and len(po) >= 2:
                                po_pairs.append([str(po[0]), str(po[1])])
                
                triples_items.append({
                    "subject_uri": md.get("subject_uri", ""),
                    "po_pairs": po_pairs,
                })
        
        return {
            "prefixes_text": prefixes_text,
            "ontology_items": ontology_items,
            "mapping_items": mapping_items,
            "triples_items": triples_items,
        }

    def _format_context_for_prompt(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Format retrieved context into prompt-ready strings."""
        from src.tools.uri_shortener import shorten_uri, shorten_text
        
        # Format ontology (使用本体文件查询结果)
        ontology_lines = []
        for item in context.get("ontology_items", [])[:10]:
            # 缩短 URI
            uri = shorten_uri(item.get('uri', ''), self.prefix_map)
            parts = [f"- {uri} ({item.get('element_type', '')})"]
            
            properties = item.get("properties", {})
            
            if properties:
                # 按优先级排序
                sorted_props = sorted(
                    properties.items(),
                    key=lambda x: self._categorize_property(x[0])
                )
                
                # 最多显示 5 个属性
                for pred, values in sorted_props[:5]:
                    _, display_name = self._categorize_property(pred)
                    value = values[0] if values else ""
                    
                    # 截断过长文本（适度截断：120 字符）
                    if len(value) > 120:
                        value = value[:120] + "..."
                    
                    # 缩短属性名和值中的 URI
                    pred_short = shorten_uri(pred, self.prefix_map)
                    value_short = shorten_text(value, self.prefix_map)
                    parts.append(f"  {display_name}: {value_short}")
            else:
                # 降级：使用向量库的 label/comment
                if item.get('label'):
                    parts.append(f"  Label: {item['label']}")
                if item.get('comment'):
                    comment = item['comment']
                    # 适度截断：120 字符
                    if len(comment) > 120:
                        comment = comment[:120] + "..."
                    parts.append(f"  Comment: {comment}")
            
            ontology_lines.append("\n".join(parts))
        ontology_text = "\n".join(ontology_lines) if ontology_lines else "(No ontology elements retrieved)"
        
        # Format mappings (展示 Mapping ID + 完整 Target Pattern + 完整 SQL)
        mapping_lines = []
        for item in context.get("mapping_items", [])[:10]:
            # Target Pattern 保持完整 URI，不缩短
            lines = [
                f"- Mapping ID: {item.get('mapping_id', '')}",
                f"  Target: {item.get('target_pattern', '')}",
                f"  Source SQL: {item.get('source_query', '')}"
            ]
            mapping_lines.append("\n".join(lines))
        mappings_text = "\n".join(mapping_lines) if mapping_lines else "(No VKG mappings retrieved)"
        
        # Format triples (移除 brief，只保留事实数据)
        triples_lines = []
        for item in context.get("triples_items", [])[:8]:
            # 缩短 subject URI
            subject = shorten_uri(item.get('subject_uri', ''), self.prefix_map)
            lines = [f"- Subject: {subject}"]
            po_pairs = item.get("po_pairs", [])
            if po_pairs:
                lines.append("  Properties:")
                # 适度截断：4 个 po_pairs，每个 Object 截断到 40 字符
                for p, o in po_pairs[:4]:
                    o_str = str(o)
                    if len(o_str) > 40:
                        o_str = o_str[:40] + "..."
                    # 缩短 predicate 和 object 中的 URI
                    p_short = shorten_uri(p, self.prefix_map)
                    o_short = shorten_text(o_str, self.prefix_map)
                    lines.append(f"    {p_short} → {o_short}")
            triples_lines.append("\n".join(lines))
        triples_text = "\n".join(triples_lines) if triples_lines else "(No triples retrieved)"
        
        return {
            "prefixes_text": context.get("prefixes_text", ""),
            "ontology_text": ontology_text,
            "mappings_text": mappings_text,
            "triples_text": triples_text,
        }

    # =========================================================================
    # Core Methods: Generation
    # =========================================================================

    def _generate_candidates(
        self,
        question: str,
        context: Dict[str, Any],
        k: int = 3,
        memory_text: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Generate k candidate SPARQL queries with Fast & Slow hybrid strategy.
        
        - Round 1 (no memory): FAST PATH with structured output (no analysis text)
        - Round 2+ (with memory): SLOW PATH with JSON fallback (deep analysis)
        
        Args:
            question: User question
            context: Retrieved context (ontology/mappings/triples)
            k: Number of candidates to generate
            memory_text: Formatted iteration memory (for subsequent rounds)
        
        Returns list of dicts with:
        - text: Natural language description
        - sparql: Full SPARQL query
        - purpose: What this query tries to achieve
        - query_type: SPARQL query type (SELECT/ASK/CONSTRUCT/DESCRIBE)
        - exec_result_preview: Preview of execution results (if ontop_client available)
        """
        formatted = self._format_context_for_prompt(context)
        
        # === Fast & Slow Branch Logic ===
        if not memory_text:
            # === Branch 1: Round 1 - FAST PATH (System 1: Intuition) ===
            logger.info("Round 1: Fast Path (Structured Output)")
            
            # Use minimal prompt (no analysis required)
            prompt_template = PromptTemplate.from_template(
                _prompts.generate_candidates_simple_json,
                template_format="jinja2"
            )
            prompt_text = prompt_template.format(
                k=k,
                prefixes_text=formatted["prefixes_text"],
                ontology_text=formatted["ontology_text"],
                mappings_text=formatted["mappings_text"],
                triples_text=formatted["triples_text"],
                question=question,
            )
            
            # Use structured output (Function Calling) for speed
            with get_usage_metadata_callback() as cb:
                structured_llm = self.llm.with_structured_output(CandidatesOutput)
                result = structured_llm.invoke(prompt_text)
                usage_meta = cb.usage_metadata or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            # Record usage (serialize structured output to JSON string)
            output_text = result.model_dump_json(indent=2)
            record_call(
                "vkg_agent.generate_candidates",
                getattr(self.llm, "model", None),
                usage_meta,
                input_text=prompt_text,
                output_text=output_text
            )
        else:
            # === Branch 2: Round 2+ - SLOW PATH (System 2: Deep Thinking) ===
            logger.info("Round >1: Slow Path (CoT Analysis)")
            
            # Use deep prompt with memory and analysis requirement
            prompt_template = PromptTemplate.from_template(
                _prompts.generate_candidates_with_memory_json,
                template_format="jinja2"
            )
            prompt_text = prompt_template.format(
                k=k,
                prefixes_text=formatted["prefixes_text"],
                ontology_text=formatted["ontology_text"],
                mappings_text=formatted["mappings_text"],
                triples_text=formatted["triples_text"],
                question=question,
                memory_text=memory_text,
            )
            
            # Use JSON parser with fallback (allows thinking process)
            result, usage_meta, output_text = invoke_with_json_fallback(
                self.llm,
                prompt_text,
                CandidatesOutput,
                operation_name="vkg_agent.generate_candidates"
            )
            
            record_call(
                "vkg_agent.generate_candidates",
                getattr(self.llm, "model", None),
                usage_meta,
                input_text=prompt_text,
                output_text=output_text
            )
        
        # Convert to output format
        prefix_map = build_prefix_map_for_vkg(self.vkg_name)
        candidates: List[Dict[str, Any]] = []
        
        for cand in result.candidates:
            body = cand.body.strip()
            if not body:
                continue
            # Assemble full SPARQL with prefixes
            sparql_full = assemble_full_sparql(prefix_map, body)
            candidates.append({
                "text": cand.text,
                "sparql": sparql_full,
                "purpose": cand.purpose,
                "query_type": cand.query_type,  # Store query type
                "body": body,  # Store body for memory bank
            })
        
        # Execute candidates immediately if ontop_client is available
        if self.ontop_client and candidates:
            queries_to_exec = [c["sparql"] for c in candidates]
            exec_results = self.execute_sparql_batch(queries_to_exec, method="post")
            
            # Attach both full result and preview to each candidate
            for idx, cand in enumerate(candidates):
                if idx < len(exec_results):
                    exec_res = exec_results[idx]
                    query_type = cand.get("query_type", "SELECT")
                    preview = self._make_exec_preview(exec_res, query_type)
                    cand["exec_result_preview"] = preview
                    cand["exec_result"] = exec_res  # Keep full result for experiment script
        
        return candidates[:k]

    def _make_exec_preview(self, exec_result: Dict[str, Any], query_type: str = "SELECT") -> Dict[str, Any]:
        """
        Create a preview of execution result for prompt context.
        
        Handles different query types:
        - SELECT: First 10 rows, each cell truncated to 50 chars
        - ASK: Boolean result + explanation
        - CONSTRUCT: First 5 triples, URIs truncated to 60 chars
        - DESCRIBE: First 3 subjects with type info, max 200 tokens
        
        Args:
            exec_result: Execution result dict
            query_type: SPARQL query type (SELECT/ASK/CONSTRUCT/DESCRIBE)
        
        Returns dict with:
        - query_type: Query type
        - status: OK/NO_ROWS/ERROR_OR_INVALID
        - row_count: Number of rows
        - columns: List of column names
        - sample_rows: Sample data (format varies by query type)
        """
        import math
        
        status = exec_result.get("status", "UNKNOWN")
        df = exec_result.get("dataframe") if exec_result.get("dataframe") is not None else exec_result.get("result")
        
        preview = {"query_type": query_type, "status": status}
        
        # Handle ASK queries
        if query_type == "ASK":
            if isinstance(df, pd.DataFrame) and not df.empty:
                # ASK results have a 'boolean' column
                if 'boolean' in df.columns:
                    result_val = df.iloc[0]['boolean']
                    preview["ask_result"] = bool(result_val)
                    preview["explanation"] = f"ASK query returned: {result_val}"
                    preview["row_count"] = 1
                    preview["columns"] = ['boolean']
                    preview["sample_rows"] = [{"boolean": bool(result_val)}]
                else:
                    # Try first column as fallback
                    result_val = df.iloc[0, 0]
                    preview["ask_result"] = bool(result_val)
                    preview["explanation"] = f"ASK query returned: {result_val}"
                    preview["row_count"] = 1
                    preview["columns"] = list(df.columns)
                    preview["sample_rows"] = [df.iloc[0].to_dict()]
            else:
                preview["status"] = "ERROR_OR_INVALID"
                preview["error_msg"] = "ASK query returned unexpected format"
            return preview
        
        # Handle CONSTRUCT/DESCRIBE queries
        if query_type in ["CONSTRUCT", "DESCRIBE"]:
            if isinstance(df, pd.DataFrame):
                preview["row_count"] = len(df)
                preview["columns"] = list(df.columns) if hasattr(df, 'columns') else []
                
                # Check if this is a raw text result (from execute_sparql_mixed)
                if not df.empty and 'raw_result' in df.columns:
                    raw_text = str(df.iloc[0]['raw_result'])
                    # Truncate to ~200 tokens (800 chars)
                    truncated_text = raw_text[:800]
                    if len(raw_text) > 800:
                        truncated_text += "\n... (truncated)"
                    
                    # Count triples (rough estimate from text)
                    triple_count = raw_text.count('\n') if '\n' in raw_text else 1
                    
                    preview["raw_result_preview"] = truncated_text
                    preview["estimated_triple_count"] = triple_count
                    preview["format"] = "text"
                    preview["explanation"] = f"{query_type} query returned {triple_count} triples (text format)"
                elif not df.empty:
                    # Standard DataFrame with columns (e.g., subject, predicate, object)
                    limit = 5 if query_type == "CONSTRUCT" else 3
                    sample_rows = []
                    token_estimate = 0
                    
                    for _, row in df.head(limit).iterrows():
                        if token_estimate > 180:  # Reserve 20 tokens for structure
                            break
                        
                        row_dict = {}
                        for col in preview["columns"]:
                            val = row[col]
                            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                                row_dict[col] = None
                            elif pd.isna(val):
                                row_dict[col] = None
                            else:
                                # Truncate URIs to 60 chars
                                val_str = str(val)[:60]
                                row_dict[col] = val_str
                                token_estimate += len(val_str) // 4  # Rough token estimate
                        
                        sample_rows.append(row_dict)
                    
                    preview["sample_rows"] = sample_rows
                    preview["truncated"] = len(df) > len(sample_rows)
                else:
                    preview["sample_rows"] = []
            else:
                preview["row_count"] = 0
                preview["columns"] = []
                preview["sample_rows"] = []
            return preview
        
        # Handle SELECT queries (default)
        if isinstance(df, pd.DataFrame):
            preview["row_count"] = len(df)
            preview["columns"] = list(df.columns) if hasattr(df, 'columns') else []
            
            # Sample first 10 rows
            if not df.empty:
                sample_rows = []
                for _, row in df.head(10).iterrows():
                    row_dict = {}
                    for col in preview["columns"]:
                        val = row[col]
                        # Handle NaN/Infinity - convert to None for valid JSON
                        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                            row_dict[col] = None
                        else:
                            row_dict[col] = str(val)[:50]
                    sample_rows.append(row_dict)
                preview["sample_rows"] = sample_rows
            else:
                preview["sample_rows"] = []
        else:
            preview["row_count"] = 0
            preview["columns"] = []
            preview["sample_rows"] = []
            if "text" in exec_result:
                preview["error_msg"] = str(exec_result["text"])[:200]
        
        return preview

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run(
        self,
        question: str,
        config: Optional[Any] = None,
        num_candidates: int = 3,
        ontology_k: int = 10,
        mappings_k: int = 10,
        triples_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Main entry point - retrieval + generation.
        
        Compatible with run_experiment.py interface.
        
        Args:
            question: Natural language question
            config: Optional configuration (deprecated, for compatibility)
            num_candidates: Number of candidate queries to generate
            ontology_k: Number of ontology elements to retrieve
            mappings_k: Number of VKG mappings to retrieve
            triples_k: Number of aggregated triples to retrieve
        
        Returns:
            Dict with keys:
            - question: str
            - paragraphs: List[str] (for compatibility)
            - metadata: Dict (context info + retrieval_config)
            - queries: {"hypothetical_text_queries": List[str], "sparql_queries": List[str]}
        """
        # Use the provided retrieval k values directly
        actual_ontology_k = ontology_k
        actual_mappings_k = mappings_k
        actual_triples_k = triples_k
        
        logger.debug("[VKGAgent.run] Starting | question={}", question[:50])
        
        # 1. Retrieve context (no LLM)
        logger.debug("[VKGAgent.run] Step 1: Retrieving context...")
        context = self._retrieve_context(
            question, 
            ontology_k=actual_ontology_k,
            mappings_k=actual_mappings_k,
            triples_k=actual_triples_k
        )
        logger.debug("[VKGAgent.run] Step 1 done: Retrieved {} ontology, {} mappings, {} triples",
                    len(context.get("ontology_items", [])),
                    len(context.get("mapping_items", [])),
                    len(context.get("triples_items", [])))
        
        # 2. Generate candidates (1 LLM call + execution if ontop_client available)
        logger.debug("[VKGAgent.run] Step 2: Generating candidates...")
        candidates = self._generate_candidates(question, context, k=num_candidates)
        logger.debug("[VKGAgent.run] Step 2 done: Generated {} candidates", len(candidates))
        
        # 3. Build compatible output
        hypothetical_text_queries = [c["text"] for c in candidates]
        sparql_queries = [c["sparql"] for c in candidates]
        
        # Build metadata for compatibility
        formatted = self._format_context_for_prompt(context)
        per_source = {
            "vkg.ontology": {"items_json": json.dumps(context.get("ontology_items", []), ensure_ascii=False)},
            "vkg.mappings": {"items_json": json.dumps(context.get("mapping_items", []), ensure_ascii=False)},
            "vkg.aggregated_triples": {"items_json": json.dumps(context.get("triples_items", []), ensure_ascii=False)},
        }
        
        # Generate simple paragraph summary for compatibility
        paragraphs = []
        if context.get("ontology_items"):
            paragraphs.append(f"Ontology: Found {len(context['ontology_items'])} relevant elements.")
        if context.get("mapping_items"):
            paragraphs.append(f"Mappings: Found {len(context['mapping_items'])} relevant VKG mappings.")
        if context.get("triples_items"):
            paragraphs.append(f"Triples: Found {len(context['triples_items'])} relevant subjects with data samples.")
        
        return {
            "question": question,
            "paragraphs": paragraphs,
            "metadata": {
                "per_source": per_source,
                "context": context,
                "retrieval_config": {
                    "ontology_k": actual_ontology_k,
                    "mappings_k": actual_mappings_k,
                    "triples_k": actual_triples_k,
                    "num_candidates": num_candidates,
                },
            },
            "stats": {"num_candidates": num_candidates},
            "queries": {
                "hypothetical_text_queries": hypothetical_text_queries,
                "sparql_queries": sparql_queries,
            },
            "candidates_with_preview": candidates,  # Include full candidate info with exec previews
        }

    # =========================================================================
    # SPARQL Execution Methods (unchanged)
    # =========================================================================

    def execute_sparql(
        self,
        query: str,
        method: str = "post",
        accept_format: str = "application/json"
    ) -> pd.DataFrame:
        """Execute single SPARQL query, return DataFrame."""
        if self.ontop_client is None:
            raise ValueError("OntopClient not set. Call set_ontop_client() first.")
        if method.lower() == "get":
            return self.ontop_client.execute_sparql_with_get_to_dataframe(query, accept_format=accept_format)
        return self.ontop_client.execute_sparql_to_dataframe(query, accept_format=accept_format)

    def execute_sparql_batch(
        self,
        queries: List[str],
        method: str = "post",
        accept_format: str = "application/json"
    ) -> List[Dict[str, Any]]:
        """
        Batch execute SPARQL queries.
        
        Supports all query types: SELECT, ASK, CONSTRUCT, DESCRIBE.
        Uses execute_sparql_mixed for better handling of different result formats.
        """
        results: List[Dict[str, Any]] = []
        for q in queries:
            try:
                # Use execute_sparql_mixed to handle all query types
                if hasattr(self.ontop_client, 'execute_sparql_mixed'):
                    mixed_result = self.ontop_client.execute_sparql_mixed(q)
                else:
                    # Fallback to old method
                    mixed_result = self.execute_sparql(q, method=method, accept_format=accept_format)
            except Exception as e:
                results.append({
                    "query": q,
                    "result": None,
                    "dataframe": pd.DataFrame(),
                    "text": f"ERROR: {e.__class__.__name__}: {str(e)}",
                    "status": "ERROR_OR_INVALID",
                })
                continue
            
            # Handle mixed result (can be DataFrame or str)
            if isinstance(mixed_result, pd.DataFrame):
                df = mixed_result
                if not df.empty:
                    status = "OK"
                elif len(list(df.columns)) > 0:
                    status = "NO_ROWS"
                else:
                    status = "ERROR_OR_INVALID"
                results.append({
                    "query": q,
                    "result": df,
                    "dataframe": df,
                    "status": status,
                })
            elif isinstance(mixed_result, str):
                # CONSTRUCT/DESCRIBE queries return text (Turtle/JSON-LD)
                # Parse as DataFrame with special columns
                df = pd.DataFrame({'raw_result': [mixed_result]})
                status = "OK" if mixed_result.strip() else "NO_ROWS"
                results.append({
                    "query": q,
                    "result": df,
                    "dataframe": df,
                    "status": status,
                    "text": mixed_result,
                })
            else:
                results.append({
                    "query": q,
                    "result": pd.DataFrame(),
                    "dataframe": pd.DataFrame(),
                    "status": "ERROR_OR_INVALID",
                })
        return results

    def execute_sparql_df(
        self,
        query: str,
        method: str = "post",
        accept_format: str = "application/json"
    ) -> pd.DataFrame:
        """Execute single SPARQL query, return DataFrame."""
        return self.execute_sparql(query, method=method, accept_format=accept_format)

    def execute_sparql_batch_df(
        self,
        queries: List[str],
        method: str = "post",
        accept_format: str = "application/json"
    ) -> List[Dict[str, Any]]:
        """Batch execute SPARQL queries, return DataFrames."""
        results: List[Dict[str, Any]] = []
        for q in queries:
            df = self.execute_sparql_df(q, method=method, accept_format=accept_format)
            if isinstance(df, pd.DataFrame) and not df.empty:
                status = "OK"
            elif isinstance(df, pd.DataFrame) and df.empty and len(list(df.columns)) > 0:
                status = "NO_ROWS"
            else:
                status = "ERROR_OR_INVALID"
            results.append({
                "query": q,
                "dataframe": df,
                "status": status,
            })
        return results

    # =========================================================================
    # Triple Retrieval (for AnswerabilityDecider compatibility)
    # =========================================================================

    @staticmethod
    def _doc_id_for_triple(doc: Any) -> str:
        """Compute stable ID for triple document."""
        md = getattr(doc, "metadata", {}) or {}
        for key in ("subject_uri", "uri", "id"):
            val = md.get(key)
            if val:
                return str(val)
        content = getattr(doc, "page_content", "") or ""
        return content[:64]

    def retrieve_and_merge_triples_for_text_queries(
        self,
        text_queries: List[str],
        per_query_k: int = 5,
        merged_topn: Optional[int] = 20,
    ) -> Dict[str, Any]:
        """Retrieve and merge triples for text queries using RRF."""
        results_per_query: List[Dict[str, Any]] = []
        if not text_queries or self.triples_retriever is None:
            return {"per_query": results_per_query, "merged": []}

        per_list_best: List[Dict[str, Tuple[int, Any]]] = []
        for q in text_queries:
            hits_meta: List[Dict[str, Any]] = []
            best_map: Dict[str, Tuple[int, Any]] = {}
            docs = self.triples_retriever.similarity_search(q, k=int(per_query_k)) or []
            for idx, d in enumerate(docs):
                md = getattr(d, "metadata", {}) or {}
                hits_meta.append(dict(md))
                doc_id = self._doc_id_for_triple(d)
                rank_1based = idx + 1
                prev = best_map.get(doc_id)
                if prev is None or rank_1based < prev[0]:
                    best_map[doc_id] = (rank_1based, d)
            results_per_query.append({"text_query": q, "hits": hits_meta})
            per_list_best.append(best_map)

        # RRF merge
        all_lists = []
        for best_map in per_list_best:
            lst = sorted(best_map.values(), key=lambda t: t[0])
            all_lists.append([d for _, d in lst])
        scored = rrf_merge_from_lists(all_lists, id_getter=lambda d: self._doc_id_for_triple(d))
        
        merged: List[Dict[str, Any]] = []
        for score, best_rank, d in scored:
            md = getattr(d, "metadata", {}) or {}
            merged.append(dict(md))
        if isinstance(merged_topn, int) and merged_topn > 0:
            merged = merged[:merged_topn]

        return {"per_query": results_per_query, "merged": merged}


# =============================================================================
# Main (for testing)
# =============================================================================

def main() -> None:
    """Demo entry point."""
    import os
    import argparse

    parser = argparse.ArgumentParser(description="VKGAgent demo")
    parser.add_argument("--embedding-model", "-e", default="local_qwen_3_8b_embedding")
    parser.add_argument("--ontology-name", "-on", default="bgee_v14_genex")
    parser.add_argument("--vkg-name", "-vn", default="bgee_v14_genex")
    parser.add_argument("--vector-db-dir", "-o", default="resources/vector_databases")
    parser.add_argument("--llm-model-key", "-L", default="mmm_beta_gpt_4o")
    parser.add_argument("--question", "-q", default="List gene expression data")
    args = parser.parse_args()

    resolved_paths = build_vector_db_paths(
        base_directory=args.vector_db_dir,
        embedding_model_key=args.embedding_model,
        ontology_name=args.ontology_name,
        vkg_name=args.vkg_name,
        llm_model_key=args.llm_model_key,
    )

    agent = VKGAgent(
        sparql_vector_db_path=None,
        ontology_vector_db_path=resolved_paths["ontology"],
        vkg_mappings_vector_db_path=resolved_paths["vkg"],
        aggregated_triples_vector_db_path=None,
        embedding_model_key=args.embedding_model,
        llm_model_key=args.llm_model_key,
        vkg_name=args.vkg_name,
    )

    result = agent.run(args.question, k=3)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
