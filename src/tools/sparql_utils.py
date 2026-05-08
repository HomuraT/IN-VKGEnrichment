from typing import Dict, List

from src.config.prefix_for_vkgs import prefix_for_vkgs as _pfv


def normalize_alias(alias_raw: str) -> str:
    """
    规范化前缀别名：空字符串或":" 正常化为":"；否则去掉尾部冒号。
    """
    alias = str(alias_raw)
    return ":" if alias in (":", "") else alias.rstrip(":")


def strip_prefix_lines(sparql_text: str) -> str:
    """
    从可能包含 PREFIX 段的 SPARQL 文本中移除所有以 'PREFIX ' 开头的行，返回查询体。
    """
    if not isinstance(sparql_text, str) or not sparql_text.strip():
        return ""
    lines = [ln for ln in sparql_text.splitlines() if ln.strip()]
    non_prefix_lines: List[str] = []
    for ln in lines:
        if ln.strip().upper().startswith("PREFIX "):
            continue
        non_prefix_lines.append(ln)
    return "\n".join(non_prefix_lines).strip()


def sanitize_body_remove_prefix(body: str) -> str:
    """确保 body 内不含任何 PREFIX 行。"""
    return strip_prefix_lines(body)


def build_prefix_map_for_vkg(vkg_name: str | None) -> Dict[str, str]:
    """
    基于 vkg_name 从配置中构建 {alias_norm -> IRI} 映射；无则返回空。
    """
    if not vkg_name:
        return {}
    vkg_entry = _pfv.get(vkg_name)
    if not isinstance(vkg_entry, dict):
        return {}
    prefix_map: Dict[str, str] = {}
    for k, v in vkg_entry.items():
        alias_norm = normalize_alias(str(k))
        prefix_map[alias_norm] = str(v)
    return prefix_map


def prefix_map_to_section(prefix_map: Dict[str, str]) -> str:
    """将 {alias -> IRI} 转换为多行 PREFIX 段。"""
    lines: List[str] = []
    for alias_norm, iri in prefix_map.items():
        if not iri:
            continue
        if alias_norm == ":":
            lines.append(f"PREFIX : <{iri}>")
        else:
            lines.append(f"PREFIX {alias_norm}: <{iri}>")
    return "\n".join(lines)


def assemble_full_sparql(prefix_map: Dict[str, str], body: str) -> str:
    """
    根据前缀映射与查询体，装配完整 SPARQL 字符串（PREFIX 段 + 空行 + body）。
    若 body 为空，返回空字符串。
    """
    body_val = (body or "").strip()
    if not body_val:
        return ""
    prefix_section = prefix_map_to_section(prefix_map)
    return (prefix_section + "\n\n" + body_val).strip()


def existing_pairs_to_llm_body(existing: List[dict]) -> List[dict]:
    """
    将已有 {text, body|sparql} 对转换为 {text, body}，并移除 body 中的 PREFIX。
    """
    result: List[dict] = []
    for e in (existing or []):
        if not isinstance(e, dict):
            continue
        text_e = str(e.get("text") or "")
        body_e_raw = str(e.get("body") or e.get("sparql") or "")
        body_e = strip_prefix_lines(body_e_raw)
        result.append({"text": text_e, "body": body_e})
    return result


