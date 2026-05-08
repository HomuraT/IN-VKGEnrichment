import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import queue as _queue
from multiprocessing.pool import TimeoutError as MPTimeoutError

import pandas as pd  # type: ignore
from src.experiment.load_dataset import iter_jsonl
from src.config.vkg_endpoints import get_vkg_sparql_endpoint_url
from loguru import logger
from tqdm import tqdm  # type: ignore
import hashlib
import time


# =========================
# Decision Status Constants
# =========================

class DecisionStatus:
    """统一管理决策状态常量和判断逻辑"""
    YES = "YES"
    REFINE = "REFINE"
    FAIL = "FAIL"
    
    # 成功状态：YES 和 REFINE 都算产生了最终查询
    SUCCESS_STATES = frozenset([YES])
    
    @classmethod
    def is_successful(cls, decision: str) -> bool:
        """判断决策是否为成功状态（YES 或 REFINE）"""
        return str(decision or "").upper() in cls.SUCCESS_STATES
    
    @classmethod
    def has_final_query(cls, decision: str, final_sparql: str) -> bool:
        """判断是否有最终查询：决策成功且查询非空"""
        return cls.is_successful(decision) and bool(str(final_sparql or "").strip())
    
    @classmethod
    def has_any_query(cls, final_sparql: str) -> bool:
        """判断是否有任何查询（包括 FAIL 状态的 best_guess）"""
        return bool(str(final_sparql or "").strip())


# =========================
# Caching utilities (CSV.gz)
# =========================

def _normalize_sparql(s: str) -> str:
    return str(s or "").strip()


def _cache_key(vkg_name: str, sparql: str) -> str:
    payload = json.dumps([str(vkg_name or ""), _normalize_sparql(sparql)], ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _cache_dir_for_vkg(base_dir: str, vkg_name: str) -> str:
    d = os.path.join(base_dir, str(vkg_name or "default"))
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(base_dir: str, vkg_name: str, key: str) -> str:
    return os.path.join(_cache_dir_for_vkg(base_dir, vkg_name), f"{key}.csv.gz")


def _cache_load_df(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            return pd.read_csv(path, compression="gzip")
    except Exception:
        return None
    return None


def _cache_save_df(path: str, df: pd.DataFrame) -> None:
    try:
        # 存为 CSV.gz；DataFrame 为空也缓存，避免重复请求
        df.to_csv(path, index=False, compression="gzip")
    except Exception:
        # 写缓存失败不应影响主流程
        pass


def _safe_key(s: str) -> str:
    """Make keys safe for filenames: replace '/' and spaces with '_' (align with runner)."""
    return str(s).replace("/", "_").replace(" ", "_")


def _extract_prefix_from_decisions_path(decisions_path: str) -> Optional[str]:
    """从 decisions 文件路径中提取前缀信息。
    
    期望格式: <prefix>.<basename>
    例如: local_qwen_2_5_7b.bgee_v14_genex.ont10.map10.tri10.cand3.iter2.decisions.jsonl
    返回: local_qwen_2_5_7b.bgee_v14_genex.ont10.map10.tri10.cand3.iter2
    """
    if not decisions_path:
        return None
    
    base_name = os.path.basename(decisions_path)
    # 查找 decisions.jsonl 或 decisions 的位置
    if "decisions.jsonl" in base_name:
        prefix = base_name.split("decisions.jsonl")[0].rstrip(".")
    elif "decisions" in base_name:
        prefix = base_name.split("decisions")[0].rstrip(".")
    else:
        # 如果没有 decisions 标记，尝试提取最后一个点之前的部分
        parts = base_name.rsplit(".", 1)
        if len(parts) > 1:
            prefix = parts[0]
        else:
            return None
    
    return prefix if prefix else None


def _prefix_output_filename(output_path: Optional[str], prefix: Optional[str]) -> Optional[str]:
    """使用提取的前缀为输出文件名添加前缀。
    
    Args:
        output_path: 输出文件路径
        prefix: 从 decisions 路径提取的前缀
    
    Returns:
        添加了前缀的完整路径
    """
    if not output_path or not prefix:
        return output_path
    
    base_dir, base_name = os.path.split(output_path)
    if not base_name:
        return output_path
    
    new_name = f"{prefix}.{base_name}"
    return os.path.join(base_dir, new_name)


def _build_id_to_gold_sparql_map(dataset_path: Optional[str]) -> Dict[str, str]:
    """Build an id -> gold SPARQL mapping from a dataset JSONL file.

    If the dataset path is None or does not exist, returns an empty mapping.
    """
    id_to_sparql: Dict[str, str] = {}
    if not dataset_path:
        return id_to_sparql
    try:
        if not os.path.exists(dataset_path):
            return id_to_sparql
        for rec in iter_jsonl(dataset_path):
            rid = str((rec or {}).get("id") or "").strip()
            sparql = str((rec or {}).get("sparql") or "").strip()
            if rid and sparql:
                id_to_sparql[rid] = sparql
    except Exception:
        # 按约定不吞错：此处仅限最小范围的健壮性保护，不影响主流程
        return id_to_sparql
    return id_to_sparql


def _build_id_to_references_map(dataset_path: Optional[str]) -> Dict[str, bool]:
    """Build an id -> has_references mapping from a dataset JSONL file.

    Returns a dict mapping sample id to True if it has non-empty references, False otherwise.
    """
    id_to_has_refs: Dict[str, bool] = {}
    if not dataset_path:
        return id_to_has_refs
    try:
        if not os.path.exists(dataset_path):
            return id_to_has_refs
        for rec in iter_jsonl(dataset_path):
            rid = str((rec or {}).get("id") or "").strip()
            if not rid:
                continue
            refs = rec.get("references", []) or []
            # 简单判断：只要 references 字段存在且不为空列表
            has_refs = bool(refs and len(refs) > 0)
            id_to_has_refs[rid] = has_refs
    except Exception:
        return id_to_has_refs
    return id_to_has_refs


def _build_id_to_details_map(predictions_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Build an id -> details mapping from predictions JSONL file.
    
    Extracts input, model, llm_usage, and other metadata for each sample.
    """
    id_to_details: Dict[str, Dict[str, Any]] = {}
    if not predictions_path:
        return id_to_details
    try:
        if not os.path.exists(predictions_path):
            return id_to_details
        for rec in iter_jsonl(predictions_path):
            rid = str((rec or {}).get("id") or "").strip()
            if not rid:
                continue
            
            # 提取关键信息
            sample = rec.get("sample", {}) or {}
            llm_usage = rec.get("llm_usage", {}) or {}
            
            id_to_details[rid] = {
                "input": sample.get("question", ""),
                "model": llm_usage.get("model", ""),
                "llm_usage": llm_usage,
            }
    except Exception:
        return id_to_details
    return id_to_details


def _normalize_cell(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s.lower()


def _df_columns_to_value_counters(df) -> Dict[str, Counter]:
    if df is None:
        return {}
    col_to_counts: Dict[str, Counter] = {}
    for col in list(df.columns):
        counts: Counter = Counter()
        for v in df[col].tolist():
            nv = _normalize_cell(v)
            if nv is not None:
                counts[nv] += 1
        col_to_counts[str(col)] = counts
    return col_to_counts


def _best_match_for_gold_values(gold_counts: Counter, pred_map: Dict[str, Counter]) -> Tuple[Optional[str], int, int]:
    best_col: Optional[str] = None
    best_intersection = 0
    best_pred_total = 0
    for pcol, pcounts in pred_map.items():
        # multiset intersection size: sum of min counts per value
        inter = 0
        for val, gcnt in gold_counts.items():
            inter += min(gcnt, pcounts.get(val, 0))
        pred_total = sum(pcounts.values())
        if inter > best_intersection:
            best_intersection = inter
            best_col = pcol
            best_pred_total = pred_total
    return best_col, best_intersection, best_pred_total


def _compute_per_sample_scores(gold_df, pred_df) -> Dict[str, Any]:
    gold_map = _df_columns_to_value_counters(gold_df)
    pred_map = _df_columns_to_value_counters(pred_df)

    gold_cols = list(gold_map.keys())
    pred_cols = list(pred_map.keys())
    Cg = len(gold_cols)
    Cp = len(pred_cols)

    # Column-wise metrics (macro over gold columns)
    per_col: List[Dict[str, Any]] = []
    sum_f1 = 0.0
    total_inter = 0
    total_gold_vals = 0
    total_pred_vals_for_matched = 0

    for gcol in gold_cols:
        gcnts = gold_map[gcol]
        gold_total = sum(gcnts.values())
        best_col, inter, pred_total = _best_match_for_gold_values(gcnts, pred_map)
        prec = (inter / pred_total) if pred_total > 0 else 0.0
        rec = (inter / gold_total) if gold_total > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_col.append(
            {
                "gold_col": gcol,
                "pred_col": best_col,
                "intersection": inter,
                "gold_size": gold_total,
                "pred_size": pred_total,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
        sum_f1 += f1
        total_inter += inter
        total_gold_vals += gold_total
        total_pred_vals_for_matched += pred_total

    macro_f1 = (sum_f1 / Cg) if Cg > 0 else 0.0

    # Asymmetric column-count factor (only penalize over-prediction), alpha=1
    if Cp is None or Cp == 0:
        adj_factor = 1.0
    else:
        ratio = (Cg / Cp) if Cp > 0 else 1.0
        adj_factor = ratio if ratio < 1.0 else 1.0
    adj_macro_f1 = macro_f1 * adj_factor

    return {
        "macro_f1": macro_f1,
        "adj_factor": adj_factor,
        "adj_macro_f1": adj_macro_f1,
        "per_columns": per_col,
        "Cg": Cg,
        "Cp": Cp,
    }


_POOL: Optional[mp_pool.Pool] = None
# 默认并发（可通过 CLI 覆盖）
_POOL_PROCESSES: int = 16


def _reset_pool() -> None:
    try:
        global _POOL
        if _POOL is not None:
            _POOL.close()
            _POOL.terminate()
            _POOL.join()
            _POOL = None
    except Exception:
        pass


def _init_pool() -> mp_pool.Pool:
    global _POOL
    if _POOL is None:
        processes = int(max(1, int(_POOL_PROCESSES or 1)))
        # 复用进程池，限制任务数以定期刷新 worker 状态
        _POOL = mp.Pool(processes=processes, maxtasksperchild=100)
    return _POOL


def _worker_execute_df(url: str, query: str, http_timeout: float, vkg_name: Optional[str] = None):
    # 在 worker 进程中复用 OntopClient（模块级缓存）
    # 注意：此函数必须是顶层定义，便于 multiprocessing pickle
    from typing import Dict, Tuple
    from src.tools.ontop_client import OntopClient
    global _WORKER_CLIENTS  # type: ignore
    try:
        _WORKER_CLIENTS
    except NameError:
        _WORKER_CLIENTS = {}  # type: Dict[Tuple[str, int], OntopClient]
    key = (str(url), int(http_timeout))
    client = _WORKER_CLIENTS.get(key)
    if client is None:
        client = OntopClient(endpoint_url=url, timeout=int(http_timeout))
        # 设置 VKG 名称以启用 OntopClient 缓存目录隔离
        try:
            setattr(client, "vkg_name", vkg_name)
        except Exception:
            pass
        _WORKER_CLIENTS[key] = client
    try:
        df = client.execute_sparql_to_dataframe(query)
        return ("ok", df)
    except Exception as e:
        # 返回可序列化错误，避免 httpx 异常对象 pickling 问题
        return ("err", f"{e.__class__.__name__}: {e}")


def _hard_timeout_execute_df(endpoint_url: str, sparql_query: str, total_timeout: float, inner_timeout: float, vkg_name: Optional[str] = None) -> pd.DataFrame:
    """Execute SPARQL to DataFrame via a persistent single-process pool with a hard timeout."""
    pool = _init_pool()
    async_res = pool.apply_async(_worker_execute_df, (endpoint_url, sparql_query, inner_timeout, vkg_name))
    try:
        res = async_res.get(timeout=total_timeout)
    except MPTimeoutError:
        # 硬超时：重置进程池，避免单任务卡死拖累后续样本
        _reset_pool()
        raise
    if isinstance(res, tuple) and len(res) == 2:
        status, payload = res
        if status == "ok":
            return payload  # pandas.DataFrame
        raise RuntimeError(str(payload))
    # 兼容旧返回（直接 DataFrame）
    return res


def _worker_eval_sample(payload) -> Dict[str, Any]:
    """Worker进程：评估单个样本，返回 sample_score 字典。

    payload: (
        endpoint_url: str,
        vkg_name: str,
        http_timeout: float,
        rid: str,
        decision: str,
        final_sparql: str,
        gold_sparql: str,
    )
    """
    from src.tools.ontop_client import OntopClient
    import pandas as pd  # type: ignore

    endpoint_url, vkg_name, http_timeout, rid, decision, final_sparql, gold_sparql = payload

    client = OntopClient(endpoint_url=str(endpoint_url), timeout=int(http_timeout or 30))
    try:
        setattr(client, "vkg_name", vkg_name)
    except Exception:
        pass

    has_final = DecisionStatus.has_final_query(decision, final_sparql)
    has_any_final = DecisionStatus.has_any_query(final_sparql)

    final_exec_ok = False
    final_exec_empty = False
    gold_exec_ok = False

    final_df = None
    gold_df = None

    saw_zero_columns = False
    saw_empty = False
    pred_error = None
    pred_error_type = None
    if has_any_final:
        try:
            final_df = client.execute_sparql_to_dataframe(final_sparql)
            if isinstance(final_df, pd.DataFrame):
                num_cols = len(list(final_df.columns))
                final_exec_ok = (num_cols > 0)
                final_exec_empty = bool(final_exec_ok and getattr(final_df, "empty", False))
                if num_cols == 0:
                    saw_zero_columns = True
                if final_exec_empty:
                    saw_empty = True
            else:
                final_exec_ok = False
        except Exception as e:
            final_df = None
            final_exec_ok = False
            final_exec_empty = False
            pred_error = repr(e)
            pred_error_type = e.__class__.__name__

    has_gold = bool(str(gold_sparql or "").strip())
    gold_error = None
    gold_error_type = None
    if has_gold and final_exec_ok:
        try:
            gold_df = client.execute_sparql_to_dataframe(gold_sparql)
            if isinstance(gold_df, pd.DataFrame):
                gold_exec_ok = (len(list(gold_df.columns)) > 0)
            else:
                gold_exec_ok = False
        except Exception as e:
            gold_df = None
            gold_exec_ok = False
            gold_error = repr(e)
            gold_error_type = e.__class__.__name__

    # 仅在两侧都可执行时计算得分
    macro_f1 = None
    adj_macro_f1 = None
    adj_factor = None
    Cg = None
    Cp = None
    if has_gold and final_exec_ok and gold_exec_ok:
        scores = _compute_per_sample_scores(gold_df, final_df)
        macro_f1 = scores["macro_f1"]
        adj_macro_f1 = scores["adj_macro_f1"]
        adj_factor = scores["adj_factor"]
        Cg = scores["Cg"]
        Cp = scores["Cp"]

    sample_score: Dict[str, Any] = {
        "id": rid,
        "has_final": has_final,
        "has_any_final": has_any_final,
        "final_exec_ok": final_exec_ok,
        "final_exec_empty": final_exec_empty,
        "gold_exec_ok": gold_exec_ok,
        "macro_f1": macro_f1,
        "adj_macro_f1": adj_macro_f1,
        "adj_factor": adj_factor,
        "Cg": Cg,
        "Cp": Cp,
        # for logging in main process
        "saw_zero_columns": saw_zero_columns,
        "saw_empty": saw_empty,
        "pred_error": pred_error,
        "pred_error_type": pred_error_type,
        "gold_error": gold_error,
        "gold_error_type": gold_error_type,
        # has_references, input, model, llm_usage will be set later in main process
        "has_references": None,
        "input": "",
        "model": "",
        "llm_usage": {},
    }

    return sample_score


def evaluate(
    decisions_path: str,
    timeout: Optional[float] = None,
    save_json: Optional[str] = None,
    save_csv: Optional[str] = None,
    dataset_path: Optional[str] = None,
    use_hard_timeout: bool = True,
    hard_timeout_seconds: Optional[float] = None,
    hard_timeout_workers: Optional[int] = None,
    inner_timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    from src.tools.ontop_client import OntopClient

    # 优先环境变量，再尝试由 vkg-name 前缀推断（需从 decisions 文件内容推断或由调用者提前设置环境变量），最后默认 localhost
    env_url = os.environ.get("ONTOP_ENDPOINT_URL", os.environ.get("ONTOP_SPARQL_ENDPOINT", None))
    # 尝试从调用者提供的评估脚本 CLI 前缀参数来推断 VKG 名称（保持与 main 中参数一致）
    vkg_name = os.environ.get("EVAL_VKG_NAME", "bgee_v14_genex")
    cfg_url = get_vkg_sparql_endpoint_url(vkg_name=vkg_name)
    endpoint_url = str(env_url or cfg_url or "http://localhost:8080/sparql")
    logger.info(f"Using SPARQL endpoint: {endpoint_url}")
    logger.info(f"Loading decisions from: {decisions_path}")
    logger.info(f"Gold dataset path: {dataset_path or 'None'}")
    client = OntopClient(endpoint_url=endpoint_url, timeout=timeout)
    # 为 OntopClient 设置 VKG 名称，使其内部缓存按 VKG 隔离
    try:
        setattr(client, "vkg_name", vkg_name)
    except Exception:
        pass

    # 配置硬超时进程池并发
    global _POOL_PROCESSES
    try:
        _POOL_PROCESSES = int(hard_timeout_workers or 16)
    except Exception:
        _POOL_PROCESSES = 16
    logger.info(f"Hard-timeout pool workers: {_POOL_PROCESSES}")

    # 计算超时基线
    _total_to_base = float(hard_timeout_seconds or (timeout or 30.0))
    _inner_to_base = float(inner_timeout_seconds) if inner_timeout_seconds is not None else float(min(_total_to_base, (timeout or 30.0)))

    # 缓存目录（按 VKG）
    cache_base_dir = os.path.join("resources", "cache")
    os.makedirs(cache_base_dir, exist_ok=True)
    cache_vkg_dir = _cache_dir_for_vkg(cache_base_dir, vkg_name)

    all_records = list(iter_jsonl(decisions_path))
    # 优先从数据集按 id 匹配 gold SPARQL；若不存在则回退到决策文件中的 sparql 字段
    id_to_gold = _build_id_to_gold_sparql_map(dataset_path)
    # 构建 id -> has_references 映射：优先从决策文件本身读取（因为决策文件已包含完整样本信息）
    # 如果决策文件中没有 references，再尝试从数据集文件读取
    logger.debug(f"DEBUG: Building references map from dataset: {dataset_path}")
    id_to_has_refs_from_dataset = _build_id_to_references_map(dataset_path)
    logger.debug(f"DEBUG: Dataset refs map size: {len(id_to_has_refs_from_dataset)}, with refs: {sum(id_to_has_refs_from_dataset.values())}")
    
    # 根据数据集中的 ID 过滤 decisions 记录
    dataset_ids = set(id_to_gold.keys()) if id_to_gold else set()
    if dataset_path and dataset_ids:
        # 如果数据集存在且有 ID，只保留数据集中存在的样本
        records_before_filter = len(all_records)
        records = [rec for rec in all_records if str((rec or {}).get("id") or "").strip() in dataset_ids]
        records_after_filter = len(records)
        logger.info(f"Filtered by dataset IDs: {records_before_filter} → {records_after_filter} records (removed {records_before_filter - records_after_filter} samples not in dataset)")
    else:
        # 如果数据集不存在或为空，保留所有记录
        records = all_records
        logger.info(f"No dataset filtering applied, evaluating all {len(records)} records")
    
    id_to_has_refs: Dict[str, bool] = {}
    refs_from_decision = 0
    refs_from_dataset = 0
    for rec in records:
        rid = str((rec or {}).get("id") or "").strip()
        if not rid:
            continue
        # 先从决策记录本身提取 references：简单判断，只要字段存在且非空列表
        refs = rec.get("references", []) or []
        has_refs = bool(refs and len(refs) > 0)
        if has_refs:
            refs_from_decision += 1
        # 如果决策记录中没有，回退到数据集
        if not has_refs:
            has_refs = id_to_has_refs_from_dataset.get(rid, False)
            if has_refs:
                refs_from_dataset += 1
        id_to_has_refs[rid] = has_refs
    logger.debug(f"DEBUG: refs_from_decision={refs_from_decision}, refs_from_dataset={refs_from_dataset}")
    
    # 自动推断 predictions 文件路径：将 decisions 路径中的 "decisions" 替换为 "predictions"
    predictions_path = None
    if "decisions" in decisions_path:
        predictions_path = decisions_path.replace("decisions", "predictions")
        if os.path.exists(predictions_path):
            logger.info(f"Auto-detected predictions file: {predictions_path}")
        else:
            logger.debug(f"Predictions file not found (tried: {predictions_path}), token stats will be unavailable")
            predictions_path = None
    
    # 加载 predictions 文件以获取输入、模型、token 消耗等详细信息
    id_to_details = _build_id_to_details_map(predictions_path)
    logger.info(
        f"Loaded {len(records)} decision records; gold id->sparql entries: {len(id_to_gold)}; with_refs: {sum(id_to_has_refs.values())}, without_refs: {len(id_to_has_refs) - sum(id_to_has_refs.values())}; predictions details: {len(id_to_details)}"
    )

    n_total = len(records)
    n_has_final = 0
    n_final_exec_ok = 0
    n_final_exec_empty = 0

    per_sample_scores: List[Dict[str, Any]] = []

    # 若开启并发（workers > 1），使用并行评估；否则走原有单线程路径
    if int(_POOL_PROCESSES or 1) > 1:
        logger.info(f"Parallel evaluating with {int(_POOL_PROCESSES)} workers")
        # 选择较小的 HTTP 超时，避免长尾阻塞（与顺序模式的 inner_to 对齐）
        http_timeout = float(_inner_to_base)
        total_to = float(_total_to_base)
        payloads = []
        rid_to_has_refs = {}  # 记录 rid -> has_references 映射
        for rec in records:
            decision = str(rec.get("decision") or "").upper()
            final_sparql = str(rec.get("final_sparql") or "").strip()
            rid = str((rec or {}).get("id") or "").strip()
            gold_sparql = str(id_to_gold.get(rid) or rec.get("sparql") or "").strip()
            rid_to_has_refs[rid] = id_to_has_refs.get(rid, False)
            payloads.append((client.endpoint_url, vkg_name, http_timeout, rid, decision, final_sparql, gold_sparql))

        # 本地专用并发池，带每任务硬超时收割；超时则重建池并继续（避免队头阻塞）
        def _spawn_pool() -> mp_pool.Pool:
            return mp.Pool(processes=int(_POOL_PROCESSES), maxtasksperchild=100)

        par_pool = _spawn_pool()
        try:
            pending: List[Tuple[Tuple[Any, ...], Any, float]] = []  # (payload, AsyncResult, start_ts)
            for p in payloads:
                pending.append((p, par_pool.apply_async(_worker_eval_sample, (p,)), time.monotonic()))

            pbar = tqdm(total=len(payloads), desc="Evaluating", unit="sample")
            while pending:
                now = time.monotonic()
                harvested_any = False
                timed_out_any = False
                still_pending: List[Tuple[Tuple[Any, ...], Any, float]] = []
                for payload, async_res, start_ts in list(pending):
                    rid = payload[3]
                    # 超时优先处理
                    if now - start_ts >= total_to:
                        logger.warning(f"Parallel hard-timeout → mark failed | id={rid}")
                        per_sample_scores.append({
                            "id": rid,
                            "has_final": DecisionStatus.has_final_query(payload[4], payload[5]),
                            "has_any_final": DecisionStatus.has_any_query(payload[5]),
                            "final_exec_ok": False,
                            "final_exec_empty": False,
                            "gold_exec_ok": False,
                            "macro_f1": None,
                            "adj_macro_f1": None,
                            "adj_factor": None,
                            "Cg": None,
                            "Cp": None,
                            "pred_error_type": "HardTimeout",
                            "pred_error": None,
                            "gold_error_type": None,
                            "gold_error": None,
                            "has_references": rid_to_has_refs.get(rid, False),
                            "input": "",
                            "model": "",
                            "llm_usage": {},
                        })
                        # 附加输入输出和 token 信息
                        details = id_to_details.get(rid, {})
                        per_sample_scores[-1]["input"] = details.get("input", "")
                        per_sample_scores[-1]["model"] = details.get("model", "")
                        per_sample_scores[-1]["llm_usage"] = details.get("llm_usage", {})
                        pbar.update(1)
                        timed_out_any = True
                        harvested_any = True
                        continue
                    # 收割已完成任务
                    if async_res.ready():
                        try:
                            s = async_res.get(timeout=0.001)
                        except Exception as e:
                            # Worker 侧异常：按 predicted 执行错误处理
                            logger.warning(f"Predicted SPARQL execution error → final_exec_ok=false | id={rid} | err={e.__class__.__name__}: {repr(e)}")
                            s = {
                                "id": rid,
                                "has_final": DecisionStatus.has_final_query(payload[4], payload[5]),
                                "has_any_final": DecisionStatus.has_any_query(payload[5]),
                                "final_exec_ok": False,
                                "final_exec_empty": False,
                                "gold_exec_ok": False,
                                "macro_f1": None,
                                "adj_macro_f1": None,
                                "adj_factor": None,
                                "Cg": None,
                                "Cp": None,
                                "pred_error_type": e.__class__.__name__,
                                "pred_error": repr(e),
                                "gold_error_type": None,
                                "gold_error": None,
                                "has_references": rid_to_has_refs.get(rid, False),
                                "input": "",
                                "model": "",
                                "llm_usage": {},
                            }
                        # 附加输入输出和 token 信息
                        details = id_to_details.get(rid, {})
                        s["input"] = details.get("input", "")
                        s["model"] = details.get("model", "")
                        s["llm_usage"] = details.get("llm_usage", {})
                        if s.get("pred_error_type"):
                            logger.warning(f"Predicted SPARQL execution error → final_exec_ok=false | id={rid} | err={s.get('pred_error_type')}: {s.get('pred_error')}")
                        else:
                            if s.get("final_exec_ok") is False and s.get("saw_zero_columns"):
                                logger.debug(f"0 columns (treated as error) for predicted SPARQL | id={rid}")
                            elif s.get("final_exec_empty"):
                                logger.debug(f"Empty result for predicted SPARQL | id={rid}")
                        if s.get("gold_error_type"):
                            logger.warning(f"Gold SPARQL execution error → gold_exec_ok=false | id={rid} | err={s.get('gold_error_type')}: {s.get('gold_error')}")
                        # 确保 has_references 字段存在且不为 None
                        if s.get("has_references") is None:
                            s["has_references"] = rid_to_has_refs.get(rid, False)
                        # 附加输入输出和 token 信息
                        if not s.get("input"):
                            details = id_to_details.get(rid, {})
                            s["input"] = details.get("input", "")
                            s["model"] = details.get("model", "")
                            s["llm_usage"] = details.get("llm_usage", {})
                        per_sample_scores.append(s)
                        pbar.update(1)
                        harvested_any = True
                    else:
                        still_pending.append((payload, async_res, start_ts))

                pending = still_pending

                # 若存在超时，重建进程池并重提剩余任务
                if timed_out_any and pending:
                    try:
                        par_pool.terminate(); par_pool.join()
                    except Exception:
                        pass
                    par_pool = _spawn_pool()
                    pending = [(pl, par_pool.apply_async(_worker_eval_sample, (pl,)), time.monotonic()) for (pl, _, __) in pending]

                # 若无收割，稍作等待避免忙等
                if not harvested_any and pending:
                    time.sleep(0.05)
            pbar.close()
        finally:
            try:
                par_pool.close()
                par_pool.terminate()
                par_pool.join()
            except Exception:
                pass

    else:
        for rec in tqdm(records, desc="Evaluating", unit="sample"):
            decision = str(rec.get("decision") or "").upper()
            final_sparql = str(rec.get("final_sparql") or "").strip()
            rid = str((rec or {}).get("id") or "").strip()
            # 数据集优先，其次回退到记录内自带的 gold sparql（若有）
            gold_sparql = str(id_to_gold.get(rid) or rec.get("sparql") or "").strip()

            has_final = DecisionStatus.has_final_query(decision, final_sparql)
            has_any_final = DecisionStatus.has_any_query(final_sparql)
            if has_final:
                n_has_final += 1

            final_exec_ok = False
            final_exec_empty = False
            gold_exec_ok = False

            final_df = None
            gold_df = None

            # Execute predicted SPARQL if present (YES or GUESS)
            if has_any_final:
                try:
                    # 先查缓存
                    ck = _cache_key(vkg_name, final_sparql)
                    cp = _cache_path(cache_base_dir, vkg_name, ck)
                    cached_df = _cache_load_df(cp)
                    if cached_df is not None:
                        final_df = cached_df
                    else:
                        if use_hard_timeout:
                            total_to = float(hard_timeout_seconds or (timeout or 30.0))
                            inner_to = float(min(total_to, (timeout or 30.0), 10.0))
                            final_df = _hard_timeout_execute_df(client.endpoint_url, final_sparql, total_timeout=total_to, inner_timeout=inner_to, vkg_name=vkg_name)
                        else:
                            final_df = client.execute_sparql_to_dataframe(final_sparql)
                        if isinstance(final_df, pd.DataFrame):
                            _cache_save_df(cp, final_df)
                    if isinstance(final_df, pd.DataFrame):
                        num_cols = len(list(final_df.columns))
                        final_exec_ok = (num_cols > 0)
                        final_exec_empty = bool(final_exec_ok and getattr(final_df, "empty", False))
                        if not final_exec_ok:
                            logger.debug(f"0 columns (treated as error) for predicted SPARQL | id={rid}")
                        elif final_exec_empty:
                            logger.debug(f"Empty result for predicted SPARQL | id={rid}")
                    else:
                        final_exec_ok = False
                except MPTimeoutError:
                    logger.warning(f"Predicted SPARQL hard-timeout → final_exec_ok=false | id={rid}")
                    final_df = None
                    final_exec_ok = False
                    final_exec_empty = False
                except Exception as e:
                    # 执行报错：视为 final_exec_ok=false，不创建空 DataFrame
                    logger.warning(f"Predicted SPARQL execution error → final_exec_ok=false | id={rid} | err={e.__class__.__name__}: {repr(e)}")
                    final_df = None
                    final_exec_ok = False
                    final_exec_empty = False

            if final_exec_ok:
                n_final_exec_ok += 1
                if final_exec_empty:
                    n_final_exec_empty += 1

            # Correctness requires both gold and predicted queries
            has_gold = bool(gold_sparql)
            if has_gold and final_exec_ok:
                try:
                    # 先查缓存
                    ck_g = _cache_key(vkg_name, gold_sparql)
                    cp_g = _cache_path(cache_base_dir, vkg_name, ck_g)
                    cached_df_g = _cache_load_df(cp_g)
                    if cached_df_g is not None:
                        gold_df = cached_df_g
                    else:
                        if use_hard_timeout:
                            total_to = float(hard_timeout_seconds or (timeout or 30.0))
                            inner_to = float(min(total_to, (timeout or 30.0), 10.0))
                            gold_df = _hard_timeout_execute_df(client.endpoint_url, gold_sparql, total_timeout=total_to, inner_timeout=inner_to, vkg_name=vkg_name)
                        else:
                            gold_df = client.execute_sparql_to_dataframe(gold_sparql)
                        if isinstance(gold_df, pd.DataFrame):
                            _cache_save_df(cp_g, gold_df)
                    if isinstance(gold_df, pd.DataFrame):
                        gold_exec_ok = (len(list(gold_df.columns)) > 0)
                    else:
                        gold_exec_ok = False
                except MPTimeoutError:
                    logger.warning(f"Gold SPARQL hard-timeout → gold_exec_ok=false | id={rid}")
                    gold_df = None
                    gold_exec_ok = False
                except Exception as e:
                    logger.warning(f"Gold SPARQL execution error → gold_exec_ok=false | id={rid} | err={e.__class__.__name__}: {repr(e)}")
                    gold_df = None
                    gold_exec_ok = False

            # 附加输入输出和 token 信息
            details = id_to_details.get(rid, {})
            sample_score: Dict[str, Any] = {
                "id": rec.get("id"),
                "has_final": has_final,
                "has_any_final": has_any_final,
                "final_exec_ok": final_exec_ok,
                "final_exec_empty": final_exec_empty,
                "gold_exec_ok": gold_exec_ok,
                "macro_f1": None,
                "adj_macro_f1": None,
                "has_references": id_to_has_refs.get(rid, False),
                "input": details.get("input", ""),
                "model": details.get("model", ""),
                "llm_usage": details.get("llm_usage", {}),
            }

            if has_gold and final_exec_ok and gold_exec_ok:
                scores = _compute_per_sample_scores(gold_df, final_df)
                sample_score.update(
                    {
                        "macro_f1": scores["macro_f1"],
                        "adj_macro_f1": scores["adj_macro_f1"],
                        "adj_factor": scores["adj_factor"],
                        "Cg": scores["Cg"],
                        "Cp": scores["Cp"],
                    }
                )
            else:
                # 无法计算时不再强行写 0.0；保持为空，便于区分“未能对齐评估”的情况
                pass

            per_sample_scores.append(sample_score)

    # 使用 per-sample 结果重新汇总计数，兼容并发与顺序两种路径
    n_has_final = sum(1 for s in per_sample_scores if bool(s.get("has_final")))
    n_final_exec_ok = sum(1 for s in per_sample_scores if bool(s.get("final_exec_ok")))
    n_final_exec_empty = sum(1 for s in per_sample_scores if bool(s.get("final_exec_empty")))

    # Aggregate correctness over samples that have any final SPARQL (YES or GUESS); missing scores are treated as 0
    n_any_final_samples = sum(1 for s in per_sample_scores if bool(s.get("has_any_final")))
    if n_any_final_samples > 0:
        avg_macro_f1 = (
            sum(float(s.get("macro_f1") or 0.0) for s in per_sample_scores if bool(s.get("has_any_final"))) / n_any_final_samples
        )
        avg_adj_macro_f1 = (
            sum(float(s.get("adj_macro_f1") or 0.0) for s in per_sample_scores if bool(s.get("has_any_final"))) / n_any_final_samples
        )
    else:
        avg_macro_f1 = 0.0
        avg_adj_macro_f1 = 0.0

    # Aggregate over samples that have final SPARQL specifically (treat missing scores as 0)
    n_has_final_samples = sum(1 for s in per_sample_scores if bool(s.get("has_final")))
    if n_has_final_samples > 0:
        avg_macro_f1_has_final = (
            sum(float(s.get("macro_f1") or 0.0) for s in per_sample_scores if bool(s.get("has_final"))) / n_has_final_samples
        )
        avg_adj_macro_f1_has_final = (
            sum(float(s.get("adj_macro_f1") or 0.0) for s in per_sample_scores if bool(s.get("has_final"))) / n_has_final_samples
        )
    else:
        avg_macro_f1_has_final = 0.0
        avg_adj_macro_f1_has_final = 0.0

    # === 新增：按 references 分类统计 ===
    # DEBUG: 检查 per_sample_scores 中的 has_references 分布
    has_refs_count_in_scores = sum(1 for s in per_sample_scores if bool(s.get("has_references")))
    logger.debug(f"DEBUG: per_sample_scores has {has_refs_count_in_scores} samples with has_references=True")
    # 分组：有 references 的样本
    samples_with_refs = [s for s in per_sample_scores if bool(s.get("has_references"))]
    n_with_refs = len(samples_with_refs)
    n_with_refs_has_final = sum(1 for s in samples_with_refs if bool(s.get("has_final")))
    n_with_refs_exec_ok = sum(1 for s in samples_with_refs if bool(s.get("final_exec_ok")))
    n_with_refs_any_final = sum(1 for s in samples_with_refs if bool(s.get("has_any_final")))
    
    if n_with_refs_any_final > 0:
        avg_macro_f1_with_refs = (
            sum(float(s.get("macro_f1") or 0.0) for s in samples_with_refs if bool(s.get("has_any_final"))) / n_with_refs_any_final
        )
        avg_adj_macro_f1_with_refs = (
            sum(float(s.get("adj_macro_f1") or 0.0) for s in samples_with_refs if bool(s.get("has_any_final"))) / n_with_refs_any_final
        )
    else:
        avg_macro_f1_with_refs = 0.0
        avg_adj_macro_f1_with_refs = 0.0

    # 分组：无 references 的样本
    samples_without_refs = [s for s in per_sample_scores if not bool(s.get("has_references"))]
    n_without_refs = len(samples_without_refs)
    n_without_refs_has_final = sum(1 for s in samples_without_refs if bool(s.get("has_final")))
    n_without_refs_exec_ok = sum(1 for s in samples_without_refs if bool(s.get("final_exec_ok")))
    n_without_refs_any_final = sum(1 for s in samples_without_refs if bool(s.get("has_any_final")))
    
    if n_without_refs_any_final > 0:
        avg_macro_f1_without_refs = (
            sum(float(s.get("macro_f1") or 0.0) for s in samples_without_refs if bool(s.get("has_any_final"))) / n_without_refs_any_final
        )
        avg_adj_macro_f1_without_refs = (
            sum(float(s.get("adj_macro_f1") or 0.0) for s in samples_without_refs if bool(s.get("has_any_final"))) / n_without_refs_any_final
        )
    else:
        avg_macro_f1_without_refs = 0.0
        avg_adj_macro_f1_without_refs = 0.0

    # === 计算 token 消耗统计 ===
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    n_samples_with_usage = 0
    model_name = ""
    
    for s in per_sample_scores:
        rid = s.get("id", "")
        details = id_to_details.get(str(rid), {})
        llm_usage = details.get("llm_usage", {}) or {}
        totals = llm_usage.get("totals", {}) or {}
        
        if totals:
            total_input_tokens += int(totals.get("input_tokens", 0))
            total_output_tokens += int(totals.get("output_tokens", 0))
            total_tokens += int(totals.get("total_tokens", 0))
            n_samples_with_usage += 1
            
            # 提取模型名称（从第一个有数据的样本）
            if not model_name:
                # 尝试从 calls 中提取
                calls = llm_usage.get("calls", [])
                if calls and len(calls) > 0:
                    model_name = str(calls[0].get("model", ""))
    
    avg_input_tokens = (total_input_tokens / n_samples_with_usage) if n_samples_with_usage > 0 else 0.0
    avg_output_tokens = (total_output_tokens / n_samples_with_usage) if n_samples_with_usage > 0 else 0.0
    avg_total_tokens = (total_tokens / n_samples_with_usage) if n_samples_with_usage > 0 else 0.0

    summary = {
        "n_total": n_total,
        "n_has_final": n_has_final,
        "ratio_has_final": (n_has_final / n_total) if n_total > 0 else 0.0,
        "n_final_exec_ok": n_final_exec_ok,
        "ratio_final_exec_ok": (n_final_exec_ok / n_total) if n_total > 0 else 0.0,
        "n_final_exec_empty": n_final_exec_empty,
        "ratio_final_exec_empty": (n_final_exec_empty / n_total) if n_total > 0 else 0.0,
        "avg_macro_f1": avg_macro_f1,
        "avg_adj_macro_f1": avg_adj_macro_f1,
        "avg_macro_f1_has_final": avg_macro_f1_has_final,
        "avg_adj_macro_f1_has_final": avg_adj_macro_f1_has_final,
        # 新增：token 消耗统计
        "model": model_name,
        "token_usage": {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_total_tokens,
            "n_samples_with_usage": n_samples_with_usage,
        },
        # 新增：按 references 分类的统计
        "with_references": {
            "n_total": n_with_refs,
            "n_has_final": n_with_refs_has_final,
            "ratio_has_final": (n_with_refs_has_final / n_with_refs) if n_with_refs > 0 else 0.0,
            "n_final_exec_ok": n_with_refs_exec_ok,
            "ratio_final_exec_ok": (n_with_refs_exec_ok / n_with_refs) if n_with_refs > 0 else 0.0,
            "avg_macro_f1": avg_macro_f1_with_refs,
            "avg_adj_macro_f1": avg_adj_macro_f1_with_refs,
        },
        "without_references": {
            "n_total": n_without_refs,
            "n_has_final": n_without_refs_has_final,
            "ratio_has_final": (n_without_refs_has_final / n_without_refs) if n_without_refs > 0 else 0.0,
            "n_final_exec_ok": n_without_refs_exec_ok,
            "ratio_final_exec_ok": (n_without_refs_exec_ok / n_without_refs) if n_without_refs > 0 else 0.0,
            "avg_macro_f1": avg_macro_f1_without_refs,
            "avg_adj_macro_f1": avg_adj_macro_f1_without_refs,
        },
    }

    report = {"summary": summary, "details": per_sample_scores}
    logger.info(
        "Summary: total={n_total}, has_final={n_has_final} ({ratio_has_final:.2%}), "
        "exec_ok={n_final_exec_ok} ({ratio_final_exec_ok:.2%}), empty={n_final_exec_empty} ({ratio_final_exec_empty:.2%})".format(
            **summary
        )
    )
    logger.info(
        f"Token usage: model={summary['model']}, "
        f"total={summary['token_usage']['total_tokens']:,}, "
        f"avg={summary['token_usage']['avg_total_tokens']:.1f} "
        f"(input={summary['token_usage']['avg_input_tokens']:.1f}, output={summary['token_usage']['avg_output_tokens']:.1f})"
    )
    logger.info(
        f"With references: n={summary['with_references']['n_total']}, "
        f"has_final={summary['with_references']['n_has_final']} ({summary['with_references']['ratio_has_final']:.2%}), "
        f"exec_ok={summary['with_references']['n_final_exec_ok']} ({summary['with_references']['ratio_final_exec_ok']:.2%}), "
        f"avg_adj_macro_f1={summary['with_references']['avg_adj_macro_f1']:.4f}"
    )
    logger.info(
        f"Without references: n={summary['without_references']['n_total']}, "
        f"has_final={summary['without_references']['n_has_final']} ({summary['without_references']['ratio_has_final']:.2%}), "
        f"exec_ok={summary['without_references']['n_final_exec_ok']} ({summary['without_references']['ratio_final_exec_ok']:.2%}), "
        f"avg_adj_macro_f1={summary['without_references']['avg_adj_macro_f1']:.4f}"
    )

    if save_json:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON report to: {save_json}")

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df = pd.DataFrame(per_sample_scores)
        df.to_csv(save_csv, index=False)
        logger.info(f"Saved CSV details to: {save_csv}")

    # 评估结束：优雅关闭进程池
    try:
        global _POOL
        if _POOL is not None:
            _POOL.close()
            _POOL.terminate()
            _POOL.join()
            _POOL = None
    except Exception:
        pass

    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate decision-only outputs for SPARQL quality")
    parser.add_argument("--decisions", required=True, help="决策-only JSONL 路径")
    parser.add_argument("--timeout", type=float, default=60, help="SPARQL 请求超时（秒）")
    parser.add_argument(
        "--disable-hard-timeout",
        action="store_true",
        help="关闭子进程级硬超时（默认开启）",
        default=False,
    )
    parser.add_argument(
        "--hard-timeout-seconds",
        type=float,
        default=60,
        help="子进程级硬超时秒数（默认等于 --timeout）",
    )
    parser.add_argument(
        "--inner-timeout-seconds",
        type=float,
        default=None,
        help="子进程内 httpx 超时秒数（默认 min(total, timeout, 10)）",
    )
    parser.add_argument("--out-dir", default="./evaluations", help="默认输出目录（未指定具体文件名时生效）")
    parser.add_argument("--save-json", default=None, help="保存评估汇总与明细到 JSON 文件（默认写入 out-dir/eval_report.json）")
    parser.add_argument("--save-csv", default=None, help="保存逐样本明细到 CSV 文件（默认写入 out-dir/eval_details.csv）")
    parser.add_argument(
        "--hard-timeout-workers",
        type=int,
        default=16,
        help="硬超时子进程池并发（默认16）",
    )
    parser.add_argument(
        "--dataset",
        default="resources/datasets/easybgee_v14_2.jsonl",
        help="用于 gold SPARQL 的数据集 JSONL（按 id 匹配，默认 resources/datasets/easybgee_v14_2.jsonl）",
    )
    args = parser.parse_args(argv)

    # 从 decisions 路径提取前缀信息
    decisions_path = args.decisions
    prefix = _extract_prefix_from_decisions_path(decisions_path)
    
    if prefix:
        logger.info(f"Extracted prefix from decisions path: {prefix}")
    else:
        logger.warning(f"Could not extract prefix from decisions path: {decisions_path}")
    
    # 默认输出位置：./evaluations
    save_json = args.save_json
    save_csv = args.save_csv
    out_dir = args.out_dir
    if not save_json and out_dir:
        save_json = os.path.join(out_dir, "eval_report.json")
    if not save_csv and out_dir:
        save_csv = os.path.join(out_dir, "eval_details.csv")

    # 使用提取的前缀为输出文件命名
    save_json = _prefix_output_filename(save_json, prefix)
    save_csv = _prefix_output_filename(save_csv, prefix)

    logger.info(f"Start evaluation | decisions={decisions_path}, timeout={args.timeout}")
    if save_json:
        logger.info(f"Planned JSON output: {save_json}")
    if save_csv:
        logger.info(f"Planned CSV output: {save_csv}")

    report = evaluate(
        decisions_path=decisions_path,
        timeout=args.timeout,
        save_json=save_json,
        save_csv=save_csv,
        dataset_path=args.dataset,
        use_hard_timeout=(not args.disable_hard_timeout),
        hard_timeout_seconds=args.hard_timeout_seconds,
        hard_timeout_workers=args.hard_timeout_workers,
        inner_timeout_seconds=args.inner_timeout_seconds,
    )

    s = report["summary"]
    print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


