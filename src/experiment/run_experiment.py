import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple
import sys
import time as _time

from src.experiment.load_dataset import iter_jsonl
from src.vkg_agent.vkg_agent import VKGAgent
from src.vkg_agent.answerability_decider import AnswerabilityDecider
from src.tools.vector_db_paths import build_vector_db_names
from src.config.logging_config import setup_logging, get_logger
from src.config.vkg_endpoints import get_vkg_sparql_endpoint_url
from langchain_core.callbacks import get_usage_metadata_callback
from src.tools import llm_usage_accumulator as _usage


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize data structure for JSON serialization.
    
    Converts NaN/Infinity to None, handles nested dicts/lists/tuples.
    This runs BEFORE json.dumps() to ensure no invalid JSON values slip through.
    """
    import math
    import pandas as _pd
    
    # Handle None first
    if obj is None:
        return None
    
    # Handle floats - convert NaN/Infinity to None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Handle pandas DataFrame
    if _pd is not None and isinstance(obj, _pd.DataFrame):
        df_clean = obj.where(_pd.notna(obj), None)
        return {
            "_type": "pandas.DataFrame",
            "columns": list(df_clean.columns),
            "n_rows": int(len(df_clean)),
            "n_cols": int(df_clean.shape[1]),
            "rows": _sanitize_for_json(df_clean.to_dict(orient="records")),
        }
    
    # Handle pandas Series
    if _pd is not None and isinstance(obj, _pd.Series):
        series_clean = obj.where(_pd.notna(obj), None)
        return {"_type": "pandas.Series", "values": _sanitize_for_json(series_clean.to_dict())}
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    
    # Handle sets - convert to list
    if isinstance(obj, set):
        return [_sanitize_for_json(item) for item in obj]
    
    # Return as-is for other types (json.dumps will handle or use default)
    return obj


def _json_default(obj: Any):
    """Best-effort JSON serializer for experiment outputs.

    - pandas.DataFrame → structured dict with columns/shape/rows
    - pandas.Series → dict
    - set/tuple → list
    - objects with model_dump()/dict() → use those
    - bytes → utf-8 (replace errors)
    - Exception → structured repr
    - NaN/Infinity → null
    - fallback → str(obj)
    """
    import pandas as _pd  # type: ignore
    import math

    if _pd is not None and isinstance(obj, _pd.DataFrame):
        # Replace NaN with None before converting to dict
        df_clean = obj.where(_pd.notna(obj), None)
        return {
            "_type": "pandas.DataFrame",
            "columns": list(df_clean.columns),
            "n_rows": int(len(df_clean)),
            "n_cols": int(df_clean.shape[1]),
            "rows": df_clean.to_dict(orient="records"),
        }
    if _pd is not None and isinstance(obj, _pd.Series):
        series_clean = obj.where(_pd.notna(obj), None)
        return {"_type": "pandas.Series", "values": series_clean.to_dict()}

    if isinstance(obj, (set, tuple)):
        return list(obj)
    
    # Handle NaN and Infinity from numpy/pandas
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return obj.model_dump()  # pydantic v2
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return obj.dict()

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, Exception):
        return {"_type": "Exception", "repr": repr(obj), "str": str(obj)}

    return str(obj)


def _safe_key(s: str) -> str:
    """Make model/VKG keys safe for file names (align with vector DB naming).

    Replace '/' and spaces with '_'.
    """
    return str(s).replace("/", "_").replace(" ", "_")


def _prefix_filename(
    path: Optional[str], 
    llm_model_key: str, 
    vkg_name: str, 
    ontology_k: int,
    mappings_k: int,
    triples_k: int,
    num_candidates: int,
    iter_rounds: Optional[int] = None,
) -> Optional[str]:
    """If path is provided, prefix its base name with '<llm>.<vkg>.ont<o>.map<m>.tri<t>.cand<c>[.iter<r>].' and return.

    - Preserve original directory and extension.
    - If base name is empty, return path unchanged.
    - If path is None, return None.
    """
    if not path:
        return path
    base_dir, base_name = os.path.split(path)
    if not base_name:
        return path
    prefix = f"{_safe_key(llm_model_key)}.{_safe_key(vkg_name)}.ont{int(ontology_k)}.map{int(mappings_k)}.tri{int(triples_k)}.cand{int(num_candidates)}"
    
    if iter_rounds is not None:
        prefix = f"{prefix}.iter{int(iter_rounds)}"
    new_name = f"{prefix}.{base_name}"
    return os.path.join(base_dir, new_name)

def _init_agent(
    embedding_model_key: str,
    llm_model_key: str,
    textualize_llm_model_key: Optional[str],
    ontology_name: str,
    vkg_name: str,
    use_sparql_db: bool,
    triples_k: int = 0,
    skip_blocks: Optional[List[str]] = None,
    ontology_file_path: Optional[str] = None,
    obda_file_path: Optional[str] = None,
    enable_concept_filtering: bool = True,
) -> Optional[VKGAgent]:
    log = get_logger("run_experiment.init")
    log.debug(
        "Building vector DB names | embedding_model={}, ontology_name={}, vkg_name={}, use_sparql_db={}, ontology_file={}",
        embedding_model_key, ontology_name, vkg_name, use_sparql_db, ontology_file_path,
    )
    
    # 生成向量库名称（不含路径，不含.chroma后缀）
    # 用于路径命名的 LLM 键采用"文本化模型键"，与推理模型解耦
    db_names = build_vector_db_names(
        embedding_model_key=embedding_model_key,
        ontology_name=ontology_name,
        vkg_name=vkg_name,
        llm_model_key=(textualize_llm_model_key or llm_model_key),
    )
    log.debug("Generated DB names: {}", db_names)
    
    # 去掉 .chroma 后缀
    ontology_db_name = db_names["ontology"].replace(".chroma", "")
    mappings_db_name = db_names["vkg"].replace(".chroma", "")
    queries_db_name = db_names["t2s"].replace(".chroma", "") if use_sparql_db else None
    
    # aggregated_triples 的名称构建
    # 根据 triples_k 参数自动决定：triples_k > 0 时启用
    triples_db_name = None
    if triples_k > 0:
        llm_safe = textualize_llm_model_key or llm_model_key
        if llm_safe:
            llm_safe = llm_safe.replace("/", "_").replace(" ", "_")
        embed_safe = embedding_model_key.replace("/", "_").replace(" ", "_")
        if llm_safe:
            triples_db_name = f"{vkg_name}.{llm_safe}.{embed_safe}.textualized_aggregated_triples"
        else:
            triples_db_name = f"{vkg_name}.{embed_safe}.textualized_aggregated_triples"

    # 规范化 skip 集
    skip_set: Set[str] = set([str(x).strip() for x in (skip_blocks or []) if str(x).strip()])

    # 应用 skip：被跳过的 block 设置为 None
    if "queries" in skip_set:
        queries_db_name = None
    if "vkg.ontology" in skip_set:
        raise ValueError("Cannot skip vkg.ontology - it is required")
    if "vkg.mappings" in skip_set:
        raise ValueError("Cannot skip vkg.mappings - it is required")
    if "vkg.aggregated_triples" in skip_set:
        triples_db_name = None

    log.debug(
        "Constructing VKGAgent | ontology={}, mappings={}, triples={}, queries={}, llm_model={}",
        ontology_db_name, mappings_db_name, triples_db_name, queries_db_name, llm_model_key,
    )
    
    agent = VKGAgent(
        ontology_db_name=ontology_db_name,
        mappings_db_name=mappings_db_name,
        triples_db_name=triples_db_name,
        queries_db_name=queries_db_name,
        llm_model_key=llm_model_key,
        vkg_name=vkg_name,
        ontology_file_path=ontology_file_path,
        obda_file_path=obda_file_path,
        enable_concept_filtering=enable_concept_filtering,
    )
    log.debug("VKGAgent constructed")

    # 可选注入 OntopClient（仅当执行 SPARQL 时才需要）
    return agent


def _load_processed_ids(out_path: Optional[str]) -> Set[str]:
    """Load processed record ids from an existing JSONL output file.

    If file doesn't exist or out_path is None, returns an empty set.
    Robust to lines that lack 'id'.
    """
    processed: Set[str] = set()
    if not out_path or not os.path.exists(out_path):
        return processed
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = str((rec or {}).get("id") or "").strip()
            if rid:
                processed.add(rid)
    return processed


def _ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _format_hms(seconds: float) -> str:
    seconds_int = int(max(0, seconds))
    h = seconds_int // 3600
    m = (seconds_int % 3600) // 60
    s = seconds_int % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _progress_line(done: int, total: int, start_ts: float, width: int = 28) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    frac = done / total
    filled = int(frac * width)
    bar = "=" * filled + ">" + "-" * max(0, width - filled - 1)
    elapsed = max(0.0, _time.time() - start_ts)
    rate = (done / elapsed) if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    pct = int(frac * 100)
    return (
        f"[{bar}] {pct:3d}% {done}/{total} | elapsed {_format_hms(elapsed)} | eta {_format_hms(remaining)}"
    )


def run_experiment(
    dataset_path: str,
    vkg_name: str,
    ontology_name: Optional[str],
    embedding_model_key: str,
    llm_model_key: str,
    textualize_llm_model_key: Optional[str],
    start: int,
    limit: Optional[int],
    use_sparql_db: bool,
    exec_sparql: bool,
    out_path: Optional[str],
    skip_blocks: Optional[List[str]] = None,
    out_decision_only_path: Optional[str] = None,
    workers: int = 1,
    ontology_k: int = 10,
    mappings_k: int = 10,
    triples_k: int = 5,
    num_candidates: int = 3,
    iter_rounds: int = 3,
    ontology_file_path: Optional[str] = None,
    obda_file_path: Optional[str] = None,
    enable_concept_filtering: bool = True,
) -> None:
    log = get_logger("run_experiment")
    log.debug(
        "Experiment params | dataset={}, start={}, limit={}, exec_sparql={}, out_path={}, workers={}",
        dataset_path, start, limit, exec_sparql, out_path, workers,
    )

    # 预先验证向量库存在性（调用一次 _init_agent 即可）
    _agent_probe = _init_agent(
        embedding_model_key=embedding_model_key,
        llm_model_key=llm_model_key,
        textualize_llm_model_key=textualize_llm_model_key,
        ontology_name=(ontology_name or vkg_name),
        vkg_name=vkg_name,
        use_sparql_db=use_sparql_db,
        triples_k=triples_k,
        skip_blocks=skip_blocks,
        ontology_file_path=ontology_file_path,
        obda_file_path=obda_file_path,
        enable_concept_filtering=enable_concept_filtering,
    )
    if _agent_probe is None:
        log.debug("Agent init failed due to missing vector DBs")
        return

    # 解析并前缀输出路径（若提供）
    effective_out_path = _prefix_filename(
        out_path, 
        llm_model_key=llm_model_key, 
        vkg_name=vkg_name, 
        ontology_k=ontology_k,
        mappings_k=mappings_k,
        triples_k=triples_k,
        num_candidates=num_candidates,
        iter_rounds=iter_rounds
    )
    effective_out_decision_only_path = _prefix_filename(
        out_decision_only_path, 
        llm_model_key=llm_model_key, 
        vkg_name=vkg_name, 
        ontology_k=ontology_k,
        mappings_k=mappings_k,
        triples_k=triples_k,
        num_candidates=num_candidates,
        iter_rounds=iter_rounds
    )

    # 线程安全输出锁
    write_lock = threading.Lock()
    _ensure_parent_dir(effective_out_path)
    _ensure_parent_dir(effective_out_decision_only_path)
    log.debug(
        "Output targets resolved | out_path={} | decisions_path={}",
        effective_out_path,
        effective_out_decision_only_path,
    )

    # 断点续跑：默认启用（基于 out_path 自动判断）
    processed_ids: Set[str] = _load_processed_ids(effective_out_path)
    if processed_ids:
        log.debug("Loaded {} processed ids for resume", len(processed_ids))

    # 构建候选任务列表
    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for idx, rec in enumerate(iter_jsonl(dataset_path)):
        rid = str((rec or {}).get("id") or "").strip()
        if idx < start:
            continue
        question = str(rec.get("question") or "").strip()
        if not question:
            continue
        if rid and rid in processed_ids:
            continue
        tasks.append((idx, rec))
    if limit is not None:
        tasks = tasks[: int(limit)]
    total_tasks = len(tasks)
    log.debug("Prepared {} tasks for execution", total_tasks)

    # 工作者函数：本地化 agent/decider，避免跨线程共享
    def _process_one(task: Tuple[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        idx, rec = task
        question_local = str((rec or {}).get("question") or "").strip()
        # 样本级统计开始
        rid = str((rec or {}).get("id") or f"index-{idx}")
        _usage.start_sample(rid)
        
        worker_log = get_logger("run_experiment.worker")
        worker_log.debug("Starting task | idx={} id={}", idx, rid)
        
        # 【健康检查】每个样本开始前检查 VKG 端点可用性
        if exec_sparql:
            from src.tools.ontop_client import OntopClient as _HealthCheckClient
            env_url = os.environ.get("ONTOP_ENDPOINT_URL", os.environ.get("ONTOP_SPARQL_ENDPOINT", None))
            cfg_url = get_vkg_sparql_endpoint_url(vkg_name=vkg_name)
            endpoint_url = str(env_url or cfg_url or "http://localhost:8080/sparql")
            
            health_check_client = _HealthCheckClient(endpoint_url=endpoint_url, timeout=10)
            health_query = "SELECT * WHERE { ?sub ?pred ?obj . } LIMIT 1"
            
            worker_log.debug("VKG health check | endpoint={} idx={} id={}", endpoint_url, idx, rid)
            
            # 发送健康检查请求，失败则终止样本处理
            is_healthy = health_check_client.test_connection()
            health_check_client.close()
            
            if not is_healthy:
                worker_log.error("VKG endpoint unreachable, aborting sample | endpoint={} idx={} id={}", endpoint_url, idx, rid)
                sys.stderr.write(f"\n❌ VKG 端点不可用 (id={rid}), 实验终止\n")
                sys.stderr.flush()
                raise RuntimeError(f"VKG endpoint {endpoint_url} is unreachable for sample {rid}")
            
            worker_log.debug("VKG health check passed | idx={} id={}", idx, rid)
        
        agent_local = _init_agent(
            embedding_model_key=embedding_model_key,
            llm_model_key=llm_model_key,
            textualize_llm_model_key=textualize_llm_model_key,
            ontology_name=(ontology_name or vkg_name),
            vkg_name=vkg_name,
            use_sparql_db=use_sparql_db,
            triples_k=triples_k,
            skip_blocks=skip_blocks,
            ontology_file_path=ontology_file_path,
            obda_file_path=obda_file_path,
            enable_concept_filtering=enable_concept_filtering,
        )
        if agent_local is None:
            return None
        # 将 skip 传递给 MPR（即使路径层面已跳过，编排也不执行）
        try:
            if isinstance(skip_blocks, list):
                setattr(agent_local._mpr, "skip_blocks", set([s for s in skip_blocks if isinstance(s, str)]))
        except Exception:
            pass
        if exec_sparql:
            from src.tools.ontop_client import OntopClient as _OntopClient  # local import per thread
            # 优先环境变量，其次按 VKG 名称查找配置，最后本地默认
            env_url = os.environ.get("ONTOP_ENDPOINT_URL", os.environ.get("ONTOP_SPARQL_ENDPOINT", None))
            cfg_url = get_vkg_sparql_endpoint_url(vkg_name=vkg_name)
            endpoint_url = str(env_url or cfg_url or "http://localhost:8080/sparql")
            _client = _OntopClient(endpoint_url=endpoint_url)
            try:
                setattr(_client, "vkg_name", vkg_name)
            except Exception:
                pass
            agent_local.set_ontop_client(_client)

        decider_local: AnswerabilityDecider = AnswerabilityDecider(agent_local)

        import time as _time
        _t0 = _time.time()
        full_result = agent_local.run(
            question_local, 
            config=None, 
            num_candidates=int(num_candidates),
            ontology_k=int(ontology_k),
            mappings_k=int(mappings_k),
            triples_k=int(triples_k)
        )
        _t1 = _time.time()
        get_logger("run_experiment.worker").debug("agent.run finished index={} elapsed={:.3f}s", idx, (_t1 - _t0))

        if exec_sparql:
            # Check if candidates already have execution results (from _generate_candidates)
            candidates_with_preview = full_result.get("candidates_with_preview", [])
            has_exec_results = any(c.get("exec_result") for c in candidates_with_preview)
            
            if has_exec_results:
                # Reuse execution results from candidates (already executed in agent.run)
                get_logger("run_experiment.worker").debug("Reusing SPARQL execution results from candidates (index={})", idx)
                full_result["sparql_exec_results"] = [c.get("exec_result") for c in candidates_with_preview if c.get("exec_result")]
            else:
                # Fall back to batch execution (for backward compatibility or when ontop_client not available during generation)
                sparql_list = (full_result.get("queries") or {}).get("sparql_queries") or []
                if sparql_list:
                    _s0 = _time.time()
                    full_result["sparql_exec_results"] = agent_local.execute_sparql_batch(sparql_list, method="post")
                    _s1 = _time.time()
                    get_logger("run_experiment.worker").debug("SPARQL batch finished index={} elapsed={:.3f}s", idx, (_s1 - _s0))

        # 初始聚合三元组检索（为迭代学习提供证据）
        htq_list = (full_result.get("queries") or {}).get("hypothetical_text_queries") or []
        if htq_list and getattr(agent_local, "triples_retriever", None) is not None:
            _tq0 = _time.time()
            triples_pack = agent_local.retrieve_and_merge_triples_for_text_queries(
                htq_list,
                per_query_k=int(num_candidates),
                merged_topn=int(num_candidates),
            )
            full_result["triples_retrieval"] = triples_pack
            _tq1 = _time.time()
            get_logger("run_experiment.worker").debug("Triples retrieval finished index={} elapsed={:.3f}s", idx, (_tq1 - _tq0))

        # 始终使用迭代式决策；当 iter_rounds=0 时仅做首次判别，不进入学习循环
        decision_obj = decider_local.decide_learn_iteratively(
            full_result,
            max_rounds=int(iter_rounds),
            exec_sparql=bool(exec_sparql),
            k=int(num_candidates),
        )

        # 确保经验库轨迹与快照随结果一并写出（即使模型在首轮即 YES）
        if not isinstance(decision_obj.get("experiences_snapshot"), list):
            if hasattr(decider_local, "_exp_bank") and hasattr(decider_local._exp_bank, "snapshot"):
                decision_obj["experiences_snapshot"] = decider_local._exp_bank.snapshot()
        if not isinstance(decision_obj.get("experience_trace"), list):
            decision_obj["experience_trace"] = []

        # 样本级统计结束
        llm_usage = _usage.end_sample()

        payload = {
            "id": rec.get("id"),
            "vkg": rec.get("vkg"),
            "question": question_local,
            "result": decision_obj,
            "sample": rec,
            "llm_usage": llm_usage,
        }

        # 线程安全增量写出
        worker_log.debug("Waiting for write lock | idx={} id={}", idx, rid)
        with write_lock:
            worker_log.debug("Acquired write lock | idx={} id={}", idx, rid)
            # 1) 决策-only 优先写入，与全量写出解耦，避免被异常短路
            if effective_out_decision_only_path:
                rec_obj = dict(rec or {})
                dec_obj = (decision_obj or {})
                dec_str = str((dec_obj or {}).get("decision") or "").upper()
                # 统一构造 final_sparql：
                # - YES: 使用决策体中的 sparql
                # - GUESS: 使用 best_guess.sparql
                # - 其他：为空字符串（确保字段存在，便于评估统计）
                if dec_str == "YES":
                    final_sparql_val = str((dec_obj or {}).get("sparql") or "")
                elif dec_str == "GUESS":
                    best_guess = dec_obj.get("best_guess") or {}
                    final_sparql_val = str((best_guess or {}).get("sparql") or "")
                else:
                    final_sparql_val = ""

                out_line = {**rec_obj, "decision": dec_str, "final_sparql": final_sparql_val}
                out_line_clean = _sanitize_for_json(out_line)

                with open(effective_out_decision_only_path, "a", encoding="utf-8") as f_dec:
                    f_dec.write(json.dumps(out_line_clean, ensure_ascii=False, default=_json_default) + "\n")
                    f_dec.flush()
                get_logger("run_experiment.writer").debug(
                    "Decision-only written | id={} decision={} path={}",
                    payload.get("id"),
                    dec_str,
                    effective_out_decision_only_path,
                )

            # 2) 全量输出其次写入，不捕获异常，便于暴露问题
            if effective_out_path:
                payload_clean = _sanitize_for_json(payload)
                with open(effective_out_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(payload_clean, ensure_ascii=False, default=_json_default) + "\n")
                    f_out.flush()

            print_obj = _sanitize_for_json({"index": idx, "id": payload.get("id"), "question": question_local})
            print(json.dumps(print_obj, ensure_ascii=False, default=_json_default))
            
            worker_log.debug("Released write lock | idx={} id={}", idx, rid)

        worker_log.debug("Task completed | idx={} id={}", idx, rid)
        return payload

    # 执行（线程池，workers<=1时也复用统一逻辑）
    max_workers = max(1, int(workers or 1))
    count = 0
    # 进度条：写入 stderr，不干扰 stdout JSON 输出
    progress_start_ts = _time.time()
    if total_tasks > 0:
        sys.stderr.write("\r" + _progress_line(count, total_tasks, progress_start_ts))
        sys.stderr.flush()
    if max_workers == 1:
        for t in tasks:
            res = _process_one(t)
            if res is not None:
                count += 1
            if total_tasks > 0:
                sys.stderr.write("\r" + _progress_line(count, total_tasks, progress_start_ts))
                sys.stderr.flush()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_one, t): t for t in tasks}
            for future in as_completed(futures):
                count += 1
                task = futures[future]
                idx, rec = task
                rid = str((rec or {}).get("id") or f"index-{idx}")
                
                # 获取结果，捕获并记录异常
                result = future.result()
                if result is None:
                    log.warning("Task returned None | idx={} id={}", idx, rid)
                
                if total_tasks > 0:
                    sys.stderr.write("\r" + _progress_line(count, total_tasks, progress_start_ts))
                    sys.stderr.flush()
    log.debug("Finished processing {} tasks", count)
    if total_tasks > 0:
        # 打印最终 100% 并换行
        sys.stderr.write("\r" + _progress_line(total_tasks, total_tasks, progress_start_ts) + "\n")
        sys.stderr.flush()
    # 打印全局 LLM 用量汇总
    gt = _usage.global_totals()
    sys.stderr.write(f"[LLM usage totals] input_tokens={gt.get('input_tokens',0)} output_tokens={gt.get('output_tokens',0)} total_tokens={gt.get('total_tokens',0)}\n")
    sys.stderr.flush()
    # 统一逻辑下已在处理中增量写出；此处不再二次写文件


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment by feeding questions to VKGAgent")
    parser.add_argument("--dataset", required=True, help="JSONL 数据集路径")
    parser.add_argument("--vkg-name", default="bgee_v14_genex", help="VKG 名称（用于前缀与路径命名）")
    parser.add_argument("--ontology-name", default=None, help="本体名称（默认同 --vkg-name，用于路径命名）")
    parser.add_argument("--embedding-model", default="local_qwen_3_8b_embedding", help="嵌入模型键（用于路径命名与运行时检索）")
    parser.add_argument("--llm-model-key", default="mmm_beta_gpt_4o", help="推理用 LLM 模型键")
    parser.add_argument("--textualize-llm-model-key", default="local_qwen_2_5_7b", help="文本化用 LLM 模型键（用于路径命名），默认 local_qwen_2_5_7b")
    parser.add_argument("--start", type=int, default=0, help="从索引 start 开始读取")
    parser.add_argument("--limit", type=int, default=None, help="最多运行多少条（None 表示全部）")
    parser.add_argument("--use-sparql-db", action="store_true", help="启用 SPARQL 示例向量库检索")
    parser.add_argument("--exec-sparql", action="store_true", help="是否执行生成的 SPARQL")
    parser.add_argument("--out", default=None, help="将每条运行结果写出到 JSONL 文件")
    parser.add_argument("--out-decision-only", default=None, help="将每条仅包含决策（YES/NO）的最小结果写出到 JSONL 文件")
    parser.add_argument("--workers", type=int, default=1, help="并行工作线程数，默认 1（串行）")
    # 检索参数
    parser.add_argument("--ontology-k", type=int, default=10, help="本体检索数量，默认 10")
    parser.add_argument("--mappings-k", type=int, default=10, help="映射检索数量，默认 10")
    parser.add_argument("--triples-k", type=int, default=5, help="三元组检索数量，默认 5")
    parser.add_argument("--num-candidates", type=int, default=3, help="每轮生成的候选 SPARQL 数量，默认 3")
    # 迭代轮数（为 0 时关闭迭代，仅做首次判别）
    parser.add_argument("--iter-rounds", type=int, default=3, help="迭代轮数上限；为 0 时关闭迭代，默认 3")
    parser.add_argument("--skip-blocks", default=None, help="以逗号分隔的块名列表：queries,vkg.ontology,vkg.mappings,vkg.aggregated_triples,memory")
    parser.add_argument("--log-level", default="DEBUG", help="日志级别：DEBUG/INFO/WARNING/ERROR/CRITICAL，默认 DEBUG")
    parser.add_argument("--ontology-file", default=None, help="本体文件路径（可选，支持 .ttl/.owl/.rdf/.xml 等格式）")
    parser.add_argument("--obda-file", default=None, help="OBDA 映射文件路径（可选，用于构建本体概念白名单）")
    parser.add_argument("--enable-concept-filtering", action="store_true", default=True, help="启用本体概念白名单过滤（默认 True）")
    parser.add_argument("--disable-concept-filtering", dest="enable_concept_filtering", action="store_false", help="禁用本体概念白名单过滤")
    args = parser.parse_args()

    # 初始化日志
    setup_logging(app_name="experiment_runner", log_level=str(args.log_level).upper(), enable_file_logging=True)
    log = get_logger("run_experiment")
    log.debug("Experiment runner started with args: {}", vars(args))

    limit_val: Optional[int] = None if args.limit is None else int(args.limit)

    # 解析 skip-blocks
    skip_blocks_list: Optional[List[str]] = None
    if args.skip_blocks is not None and str(args.skip_blocks).strip():
        skip_blocks_list = [s.strip() for s in str(args.skip_blocks).split(",") if s.strip()]
    
    # 从环境变量读取 OBDA 文件（如果命令行未提供）
    obda_file = args.obda_file or os.environ.get("OBDA_FILE")

    run_experiment(
        dataset_path=args.dataset,
        vkg_name=args.vkg_name,
        ontology_name=args.ontology_name,
        embedding_model_key=args.embedding_model,
        llm_model_key=args.llm_model_key,
        textualize_llm_model_key=args.textualize_llm_model_key,
        start=int(args.start),
        limit=limit_val,
        use_sparql_db=bool(args.use_sparql_db),
        exec_sparql=bool(args.exec_sparql),
        out_path=args.out,
        skip_blocks=skip_blocks_list,
        out_decision_only_path=args.out_decision_only,
        workers=int(args.workers),
        ontology_k=int(args.ontology_k),
        mappings_k=int(args.mappings_k),
        triples_k=int(args.triples_k),
        num_candidates=int(args.num_candidates),
        iter_rounds=int(args.iter_rounds),
        ontology_file_path=args.ontology_file,
        obda_file_path=obda_file,
        enable_concept_filtering=bool(args.enable_concept_filtering),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


