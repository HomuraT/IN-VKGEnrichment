import os
import json
import hashlib
from typing import Optional

import pandas as pd


def _normalize_sparql(s: str) -> str:
    return str(s or "").strip()


def cache_key(vkg_name: Optional[str], sparql: str) -> str:
    payload = json.dumps([str(vkg_name or "default"), _normalize_sparql(sparql)], ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def cache_dir_for_vkg(base_dir: str, vkg_name: Optional[str]) -> str:
    d = os.path.join(base_dir, str(vkg_name or "default"))
    os.makedirs(d, exist_ok=True)
    return d


def cache_path(base_dir: str, vkg_name: Optional[str], key: str) -> str:
    return os.path.join(cache_dir_for_vkg(base_dir, vkg_name), f"{key}.csv.gz")


def load_df(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            return pd.read_csv(path, compression="gzip")
    except Exception:
        return None
    return None


def save_df(path: str, df: pd.DataFrame) -> None:
    try:
        df.to_csv(path, index=False, compression="gzip")
    except Exception:
        pass


