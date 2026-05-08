from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore

__all__ = [
    "dataframe_to_preview_json",
    "exec_results_to_json_safes",
]


def dataframe_to_preview_json(df: Optional[pd.DataFrame], max_rows: int = 3) -> Optional[Dict[str, Any]]:
    """
    Convert a pandas DataFrame into a compact JSON-safe preview:
    { columns: [...], n_rows: int, n_cols: int, rows_preview: [ {col: str} ] }
    Returns None if df is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        return None
    columns: List[str] = list(df.columns)
    n_cols = len(columns)
    n_rows = 0 if df.empty else len(df)
    rows_preview: List[Dict[str, Any]] = []
    if not df.empty and max_rows > 0:
        head_df = df.head(max_rows)
        for _, row in head_df.iterrows():
            row_dict: Dict[str, Any] = {c: str(row[c]) for c in columns}
            rows_preview.append(row_dict)
    return {
        "columns": columns,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "rows_preview": rows_preview,
    }


def _extract_df_from_exec_item(item: Dict[str, Any]) -> Optional[pd.DataFrame]:
    if not isinstance(item, dict):
        return None
    val = item.get("result")
    if isinstance(val, pd.DataFrame):
        return val
    val2 = item.get("dataframe")
    if isinstance(val2, pd.DataFrame):
        return val2
    return None


def exec_results_to_json_safes(exec_results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Build a JSON-serializable list from exec_results, converting DataFrame fields to previews.
    Each item keeps: query, status/reason if present, and dataframe_preview when available.
    """
    json_items: List[Dict[str, Any]] = []
    if not exec_results:
        return json_items
    for it in exec_results:
        if not isinstance(it, dict):
            continue
        q = str((it or {}).get("query") or "")
        status = str((it or {}).get("status") or "")
        reason = str((it or {}).get("reason") or "")
        df = _extract_df_from_exec_item(it or {})
        df_prev = dataframe_to_preview_json(df, max_rows=3)
        json_items.append({
            "query": q,
            "status": status,
            "reason": reason,
            "dataframe_preview": df_prev,
        })
    return json_items


