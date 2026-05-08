from typing import Any, Callable, Dict, List, Tuple


def rrf_merge_from_lists(
    lists: List[List[Any]],
    id_getter: Callable[[Any], str],
) -> List[Tuple[float, int, Any]]:
    """
    对多个有序列表进行 1/rank 融合排序（RRF-like，简化版）：

    - 对每个输入列表，仅保留同一文档的最佳名次（rank 从 1 起）
    - 跨列表计算分数：score(doc) = Σ 1/rank
    - 若分数相同，按最佳名次升序

    Args:
        lists (List[List[Any]]): 多个有序列表，每个元素为任意“文档”对象。
        id_getter (Callable[[Any], str]): 返回文档稳定 ID 的函数。

    Returns:
        List[Tuple[float, int, Any]]: (score, best_rank, representative_doc) 排序后的结果。
    """
    if not isinstance(lists, list) or not lists:
        return []

    # 1) 为每个列表构建 doc_id -> (best_rank, Document) 的映射
    per_list_best: List[Dict[str, Tuple[int, Any]]] = []
    for lst in lists:
        best_map: Dict[str, Tuple[int, Any]] = {}
        for idx, d in enumerate(lst or []):
            doc_id = id_getter(d)
            rank_1based = idx + 1
            prev = best_map.get(doc_id)
            if prev is None or rank_1based < prev[0]:
                best_map[doc_id] = (rank_1based, d)
        per_list_best.append(best_map)

    # 2) 融合计分：score = Σ 1/rank
    all_ids = set()
    for m in per_list_best:
        all_ids.update(m.keys())

    scored: List[Tuple[float, int, Any]] = []
    for doc_id in all_ids:
        total = 0.0
        best_rank = None
        rep_doc: Any = None
        for m in per_list_best:
            info = m.get(doc_id)
            if info is not None:
                r, d = info
                total += 1.0 / float(r)
                if best_rank is None or r < best_rank:
                    best_rank = r
                    rep_doc = d
        scored.append((total, best_rank or 10**9, rep_doc))

    scored.sort(key=lambda x: (-(x[0]), x[1]))
    return scored


__all__ = [
    "rrf_merge_from_lists",
]


