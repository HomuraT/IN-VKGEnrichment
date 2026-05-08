from __future__ import annotations

import os
from typing import Dict, Optional


def _sanitize_embedding_key(embedding_model_key: str) -> str:
    """
    将嵌入模型键名做简单安全化：替换 '/' 与 空格 为 '_'
    与 script/01_build_knowldege_base.sh 的 DB_SUFFIX_SAFE 逻辑保持一致。
    """
    safe = embedding_model_key.replace("/", "_")
    safe = safe.replace(" ", "_")
    return safe


def derive_vkg_name_from_obda_path(obda_file_path: str) -> str:
    """
    根据 OBDA 文件路径推导 VKG 名称（去除 '.obda' 扩展名）。
    """
    base = os.path.basename(obda_file_path)
    if base.endswith(".obda"):
        return base[:-5]
    return os.path.splitext(base)[0]


def derive_ontology_name_from_ttl_path(ttl_file_path: str) -> str:
    """
    根据 TTL 文件路径推导本体名称（去除 '.ttl' 扩展名）。
    若调用方已有明确名称，建议直接传入名称而非路径。
    """
    base = os.path.basename(ttl_file_path)
    if base.endswith(".ttl"):
        return base[:-4]
    return os.path.splitext(base)[0]


def _sanitize_llm_key(llm_model_key: Optional[str]) -> Optional[str]:
    if llm_model_key is None:
        return None
    safe = llm_model_key.replace("/", "_")
    safe = safe.replace(" ", "_")
    return safe


def build_vector_db_names(
    *,
    embedding_model_key: str,
    ontology_name: str,
    vkg_name: str,
    llm_model_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    生成三个向量库的目录名（不含父目录）：
    - ontology: "{ONTOLOGY_NAME}.{LLM_SAFE}.{EMBED_SAFE}.textualized_ontology_elements.chroma"
    - vkg:      "{VKG_NAME}.{LLM_SAFE}.{EMBED_SAFE}.textualized_vkg_mappings.chroma"
    - t2s:      "{EMBED_SAFE}.text_to_sparql_vector_db.chroma"  # 不涉及文本化
    """
    embed_safe = _sanitize_embedding_key(embedding_model_key)
    llm_safe = _sanitize_llm_key(llm_model_key)

    def _with_llm(prefix: str, tail: str) -> str:
        # 若未提供 LLM 键，则保持兼容旧命名（仅包含嵌入模型）
        if llm_safe:
            return f"{prefix}.{llm_safe}.{embed_safe}.{tail}"
        return f"{prefix}.{embed_safe}.{tail}"

    return {
        "ontology": _with_llm(ontology_name, "textualized_ontology_elements.chroma"),
        "vkg": _with_llm(vkg_name, "textualized_vkg_mappings.chroma"),
        "t2s": f"{embed_safe}.text_to_sparql_vector_db.chroma",
    }


def build_vector_db_paths(
    base_directory: str,
    *,
    embedding_model_key: str,
    ontology_name: str,
    vkg_name: str,
    llm_model_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    生成三个向量库的完整路径（含父目录）。
    """
    names = build_vector_db_names(
        embedding_model_key=embedding_model_key,
        ontology_name=ontology_name,
        vkg_name=vkg_name,
        llm_model_key=llm_model_key,
    )
    return {k: os.path.join(base_directory, v) for k, v in names.items()}


def resolve_paths_from_env(
    *,
    fallback_embedding_model_key: str = "mmm_beta_text_embedding_3_small",
    fallback_llm_model_key: str = "mmm_beta_gpt_4o_mini",
    fallback_ontology_name: str = "bgee_v14_genex",
    fallback_vkg_name: str = "bgee_v14_genex",
    fallback_base_directory: str = "resources/vector_databases",
) -> Dict[str, str]:
    """
    从环境变量解析并构建三个向量库完整路径。
    支持的环境变量：
    - EMBEDDING_MODEL: 嵌入模型键名
    - TEXTUALIZE_LLM_MODEL 或 LLM_MODEL_KEY: 文本化时使用的 LLM 键名
    - ONTOLOGY_NAME: 本体名称；若未提供且存在 ONTOLOGY_TTL，则从路径推导
    - VKG_NAME: VKG 名称；若未提供且存在 OBDA_FILE，则从路径推导
    - VECTOR_DB_DIR: 向量库父目录
    - ONTOLOGY_TTL: 可选，本体 TTL 文件路径（用于自动推导 ONTOLOGY_NAME）
    - OBDA_FILE: 可选，OBDA 文件路径（用于自动推导 VKG_NAME）
    """
    embedding = os.environ.get("EMBEDDING_MODEL", fallback_embedding_model_key)
    llm_model = os.environ.get("TEXTUALIZE_LLM_MODEL") or os.environ.get("LLM_MODEL_KEY") or fallback_llm_model_key
    base_dir = os.environ.get("VECTOR_DB_DIR", fallback_base_directory)

    ontology_name = os.environ.get("ONTOLOGY_NAME")
    if not ontology_name:
        ttl_path = os.environ.get("ONTOLOGY_TTL")
        if ttl_path:
            ontology_name = derive_ontology_name_from_ttl_path(ttl_path)
        else:
            ontology_name = fallback_ontology_name

    vkg_name = os.environ.get("VKG_NAME")
    if not vkg_name:
        obda_path = os.environ.get("OBDA_FILE")
        if obda_path:
            vkg_name = derive_vkg_name_from_obda_path(obda_path)
        else:
            vkg_name = fallback_vkg_name

    return build_vector_db_paths(
        base_directory=base_dir,
        embedding_model_key=embedding,
        ontology_name=ontology_name,
        vkg_name=vkg_name,
        llm_model_key=llm_model,
    )


