from typing import Any, Dict, List, Optional
import threading


class _TLSState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.sample_id: Optional[str] = None
        self.calls: List[Dict[str, Any]] = []
        self.totals: Dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self.last_result: Optional[Dict[str, Any]] = None


_tls = _TLSState()
_global_lock = threading.Lock()
_global_totals: Dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


def start_sample(sample_id: str) -> None:
    _tls.sample_id = str(sample_id)
    _tls.calls = []
    _tls.totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    _tls.last_result = None


def _normalize_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    """
    将不同来源/版本的 usage 结构规范化为扁平计数。

    支持：
    - 扁平结构：{"input_tokens": int, "output_tokens": int, "total_tokens": int}
    - 别名键：prompt_tokens/completion_tokens
    - 嵌套结构：{"model_name": {"input_tokens": ..., ...}, ...}（多模型时求和）
    """
    def extract_one(d: Dict[str, Any]) -> tuple[int, int, int]:
        ti = int(d.get("input_tokens") or d.get("prompt_tokens") or 0)
        to = int(d.get("output_tokens") or d.get("completion_tokens") or 0)
        ttv = int(d.get("total_tokens") or (ti + to))
        return ti, to, ttv

    if not isinstance(usage, dict) or not usage:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if ("input_tokens" in usage) or ("output_tokens" in usage) or ("total_tokens" in usage) or \
       ("prompt_tokens" in usage) or ("completion_tokens" in usage):
        ti, to, ttv = extract_one(usage)
        return {"input_tokens": ti, "output_tokens": to, "total_tokens": ttv}

    ti_sum = 0
    to_sum = 0
    tt_sum = 0
    has_nested = False
    for v in usage.values():
        if isinstance(v, dict):
            has_nested = True
            ti, to, ttv = extract_one(v)
            ti_sum += ti
            to_sum += to
            tt_sum += ttv
    if has_nested:
        return {"input_tokens": ti_sum, "output_tokens": to_sum, "total_tokens": tt_sum}

    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _infer_model_from_usage(usage: Dict[str, Any]) -> Optional[str]:
    """
    当调用点未提供模型名时，尝试从嵌套 usage 的第一层键推断模型名。
    """
    if not isinstance(usage, dict) or not usage:
        return None
    first_key = next(iter(usage.keys()), None)
    if isinstance(first_key, str) and isinstance(usage.get(first_key), dict):
        return first_key
    return None


def record_call(
    where: str,
    model: Optional[str],
    usage: Dict[str, int],
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
) -> None:
    norm = _normalize_usage(usage)
    it = int(norm.get("input_tokens") or 0)
    ot = int(norm.get("output_tokens") or 0)
    tt = int(norm.get("total_tokens") or (it + ot))

    _tls.totals["input_tokens"] += it
    _tls.totals["output_tokens"] += ot
    _tls.totals["total_tokens"] += tt

    model_name = str(model or _infer_model_from_usage(usage) or "")

    _tls.calls.append({
        "where": str(where or ""),
        "model": model_name,
        "input_tokens": it,
        "output_tokens": ot,
        "total_tokens": tt,
        "input_text": str(input_text) if input_text is not None else "",
        "output_text": str(output_text) if output_text is not None else "",
    })


def end_sample() -> Dict[str, Any]:
    result = {
        "sample_id": _tls.sample_id,
        "totals": dict(_tls.totals),
        "calls": list(_tls.calls),
    }
    with _global_lock:
        _global_totals["input_tokens"] += result["totals"]["input_tokens"]
        _global_totals["output_tokens"] += result["totals"]["output_tokens"]
        _global_totals["total_tokens"] += result["totals"]["total_tokens"]
    _tls.last_result = result
    _tls.sample_id = None
    _tls.calls = []
    _tls.totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return result


def global_totals() -> Dict[str, int]:
    with _global_lock:
        return dict(_global_totals)


def last_sample_result() -> Optional[Dict[str, Any]]:
    return _tls.last_result


