import os
import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

# 路径设定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATASETS_DIR = os.path.join(BASE_DIR, "resources", "datasets")
UI_DIR = os.path.join(os.path.dirname(__file__), "static")
SETTINGS_PATH = os.path.join(DATASETS_DIR, ".ui_settings.json")

os.makedirs(DATASETS_DIR, exist_ok=True)

# 依赖 OntopClient 执行 SPARQL
from src.tools.ontop_client import OntopClient


def _read_settings() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_PATH):
        return {"endpoint_url": "http://localhost:8080/sparql"}
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        # 仅保留端点设置；忽略旧文件中的 annotator 字段
        return {"endpoint_url": data.get("endpoint_url") or "http://localhost:8080/sparql"}


def _write_settings(settings: Dict[str, Any]) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({"endpoint_url": settings.get("endpoint_url")}, f, ensure_ascii=False, indent=2)


def _list_jsonl_files() -> List[str]:
    return [
        name for name in sorted(os.listdir(DATASETS_DIR))
        if name.lower().endswith(".jsonl") and os.path.isfile(os.path.join(DATASETS_DIR, name))
    ]


def _jsonl_path(filename: str) -> str:
    if not filename.lower().endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="filename must end with .jsonl")
    path = os.path.join(DATASETS_DIR, filename)
    if not os.path.abspath(path).startswith(os.path.abspath(DATASETS_DIR)):
        raise HTTPException(status_code=400, detail="invalid filename")
    return path


def _read_jsonl_all(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise HTTPException(status_code=400, detail="invalid jsonl line: not an object")
            records.append(obj)
    return records


def _write_jsonl_all(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _file_version(path: str) -> str:
    """基于文件内容计算版本哈希（md5）。文件不存在时返回固定串。"""
    if not os.path.exists(path):
        return "missing:0"
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _now_iso() -> str:
    # 使用中国标准时间（Asia/Shanghai，UTC+08:00），精确到秒，包含 +08:00 偏移
    return datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0).isoformat()


def _normalize_annotators(sample: Dict[str, Any], prev_annotators: Optional[List[str]] = None) -> List[str]:
    """返回规范化后的 annotators 历史列表。
    - 去除空白并忽略空项；
    - 若当前 annotator 与最后一位不同则追加；
    - prev_annotators 可来自旧样本（用于更新时保留历史）。
    """
    annot = str(sample.get("annotator") or "").strip()
    if not annot:
        # 上层已做非空校验，这里保持一致性
        raise HTTPException(status_code=400, detail="annotator cannot be empty (stored on client cookie)")
    hist: List[str] = []
    if isinstance(prev_annotators, list):
        for a in prev_annotators:
            sa = str(a).strip()
            if sa:
                hist.append(sa)
    if not hist or hist[-1] != annot:
        hist.append(annot)
    return hist


def _validate_sample(sample: Dict[str, Any]) -> None:
    # 检查标注者来自前端（本地 cookie），样本中必须存在且非空
    annot = str(sample.get("annotator") or "").strip()
    if not annot:
        raise HTTPException(status_code=400, detail="annotator cannot be empty (stored on client cookie)")
    # 最小字段检查
    required = ["id", "vkg", "question", "sample_type", "sparql"]
    for k in required:
        if k not in sample:
            raise HTTPException(status_code=400, detail=f"missing field: {k}")


app = FastAPI(title="Dataset UI API")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(UI_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/api/files")
def list_files() -> Dict[str, Any]:
    files = _list_jsonl_files()
    meta = []
    for name in files:
        p = _jsonl_path(name)
        meta.append({"name": name, "version": _file_version(p)})
    return {"files": files, "meta": meta}


@app.post("/api/files")
def create_file(filename: str = Query(..., description="new jsonl filename")) -> Dict[str, Any]:
    path = _jsonl_path(filename)
    if os.path.exists(path):
        raise HTTPException(status_code=400, detail="file already exists")
    _write_jsonl_all(path, [])
    return {"ok": True}


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return _read_settings()


@app.post("/api/settings")
def set_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = str(payload.get("endpoint_url") or "").strip()
    if not endpoint:
        raise HTTPException(status_code=400, detail="endpoint_url cannot be empty")
    settings = {"endpoint_url": endpoint}
    _write_settings(settings)
    return settings


@app.get("/api/samples")
def get_sample(filename: str, index: int) -> Dict[str, Any]:
    path = _jsonl_path(filename)
    data = _read_jsonl_all(path)
    if index < 0 or index >= len(data):
        raise HTTPException(status_code=404, detail="index out of range")
    return {"index": index, "total": len(data), "sample": data[index], "version": _file_version(path)}


@app.get("/api/samples/by-id")
def get_sample_by_id(filename: str, sample_id: str) -> Dict[str, Any]:
    path = _jsonl_path(filename)
    data = _read_jsonl_all(path)
    for i, obj in enumerate(data):
        if str(obj.get("id")) == str(sample_id):
            return {"index": i, "total": len(data), "sample": obj}
    raise HTTPException(status_code=404, detail="id not found")


@app.post("/api/samples")
def create_sample(filename: str, sample: Dict[str, Any], expected_version: Optional[str] = None) -> Dict[str, Any]:
    path = _jsonl_path(filename)
    _validate_sample(sample)
    current_ver = _file_version(path)
    if expected_version is not None and expected_version != current_ver:
        raise HTTPException(status_code=409, detail="version conflict: file has changed, reload first")
    data = _read_jsonl_all(path)
    # 补齐创建/更新时间
    created_at = sample.get("created_at") or _now_iso()
    updated_at = _now_iso()
    sample["created_at"] = created_at
    sample["updated_at"] = updated_at
    # 维护标注者历史
    sample["annotators"] = _normalize_annotators(sample, prev_annotators=None)
    data.append(sample)
    _write_jsonl_all(path, data)
    return {"ok": True, "index": len(data) - 1, "total": len(data), "version": _file_version(path)}


@app.put("/api/samples")
def update_sample(filename: str, index: int, sample: Dict[str, Any], expected_version: Optional[str] = None) -> Dict[str, Any]:
    path = _jsonl_path(filename)
    _validate_sample(sample)
    current_ver = _file_version(path)
    if expected_version is not None and expected_version != current_ver:
        raise HTTPException(status_code=409, detail="version conflict: file has changed, reload first")
    data = _read_jsonl_all(path)
    if index < 0 or index >= len(data):
        raise HTTPException(status_code=404, detail="index out of range")
    # 保留原创建时间，刷新更新时间
    old_created = data[index].get("created_at")
    sample["created_at"] = old_created or sample.get("created_at") or _now_iso()
    sample["updated_at"] = _now_iso()
    # 维护标注者历史（基于旧样本追加）
    prev_hist = data[index].get("annotators") if isinstance(data[index].get("annotators"), list) else []
    sample["annotators"] = _normalize_annotators(sample, prev_annotators=prev_hist)
    data[index] = sample
    _write_jsonl_all(path, data)
    return {"ok": True, "version": _file_version(path)}


@app.delete("/api/samples")
def delete_sample(filename: str, index: int, expected_version: Optional[str] = None) -> Dict[str, Any]:
    path = _jsonl_path(filename)
    current_ver = _file_version(path)
    if expected_version is not None and expected_version != current_ver:
        raise HTTPException(status_code=409, detail="version conflict: file has changed, reload first")
    data = _read_jsonl_all(path)
    if index < 0 or index >= len(data):
        raise HTTPException(status_code=404, detail="index out of range")
    data.pop(index)
    _write_jsonl_all(path, data)
    return {"ok": True, "total": len(data), "version": _file_version(path)}


@app.get("/api/version")
def get_file_version(filename: str) -> Dict[str, Any]:
    path = _jsonl_path(filename)
    return {"version": _file_version(path)}


@app.post("/api/run-sparql")
def run_sparql(payload: Dict[str, Any]) -> JSONResponse:
    query = payload.get("sparql")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400, detail="sparql is required")
    # 允许前端通过 payload 覆盖端点；未提供时回退到服务器配置
    override_endpoint = str(payload.get("endpoint_url") or "").strip()
    if override_endpoint:
        endpoint = override_endpoint
    else:
        settings = _read_settings()
        endpoint = settings.get("endpoint_url") or "http://localhost:8080/sparql"
    client = OntopClient(endpoint_url=endpoint)
    result = client.execute_sparql_mixed(query)
    if hasattr(result, "to_dict"):
        # pandas DataFrame
        df_dict = {
            "columns": list(result.columns),
            "n_rows": int(result.shape[0]),
            "n_cols": int(result.shape[1]),
            "rows_preview": result.head(20).to_dict(orient="records"),
        }
        return JSONResponse(content={"type": "table", "data": df_dict})
    return JSONResponse(content={"type": "text", "data": str(result)})


# 便捷启动：uvicorn src.dataset.ui.server:app --reload --port 8000

