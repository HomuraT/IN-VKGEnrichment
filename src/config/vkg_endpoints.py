"""VKG endpoints configuration.

提供各 VKG 的基础端点，并辅助构造 SPARQL 端点 URL。

优先级（调用方可自行再叠加环境变量覆盖）：
1) 本模块内置映射
2) 调用方显式传入 default_base_url（若提供）

约定：
- 若传入/配置的是基础地址（如 http://host:port/ ），则自动追加 '/sparql' 构造 SPARQL 端点。
- 若已包含 '/sparql'，则保持不变。
"""

from typing import Optional


# 基础端点（非 /sparql 路径），结尾保留斜杠便于拼接
VKG_BASE_ENDPOINTS = {
    # bgee_v14_genex 的基础端点
    # "bgee_v14_genex": "http://47.117.72.43:8080/",
    "bgee_v14_genex": "http://10.201.113.164:8014/",
    "npd": "http://10.201.113.164:8012/",
    "uobm": "http://127.0.0.1:8000/",
}


def _ensure_sparql_path(url: str) -> str:
    s = str(url or "").strip()
    if not s:
        return s
    # 已是 /sparql 端点
    if s.rstrip("/").endswith("/sparql"):
        return s
    # 作为基础端点，统一追加 /sparql
    if not s.endswith("/"):
        s = s + "/"
    return s + "sparql"


def get_vkg_sparql_endpoint_url(vkg_name: str, default_base_url: Optional[str] = None) -> Optional[str]:
    """根据 VKG 名称返回 SPARQL 端点 URL。

    - 优先查找内置映射；没有则使用 default_base_url（若提供）。
    - 若都没有，返回 None。
    """
    base = VKG_BASE_ENDPOINTS.get(str(vkg_name).strip())
    if not base:
        base = default_base_url
    if not base:
        return None
    return _ensure_sparql_path(base)


