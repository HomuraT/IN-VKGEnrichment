"""
向量数据库服务配置

定义向量库名称到服务URL的映射关系。
每个向量库对应一个独立的服务实例。
"""

import os
from typing import Dict

# 默认服务地址映射
# 格式: {向量库名称: 服务URL}
VECTOR_DB_SERVICES: Dict[str, str] = {
    # Ontology 本体元素向量库
    "bgee_v14_genex.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_ontology_elements": 
        "http://localhost:8001",
    
    # VKG Mappings 映射向量库
    "bgee_v14_genex.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_vkg_mappings": 
        "http://localhost:8002",
    
    # Aggregated Triples 聚合三元组向量库
    "bgee_v14_genex.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_aggregated_triples": 
        "http://localhost:8003",
    
    # Text-to-SPARQL 查询向量库
    "local_qwen_3_8b_embedding.text_to_sparql_vector_db": 
        "http://localhost:8004",
    "npd.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_ontology_elements":
        "http://localhost:8005",
    "npd.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_vkg_mappings":
        "http://localhost:8101",
    "npd.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_aggregated_triples":
        "http://localhost:8007",
}


def get_service_url(db_name: str) -> str:
    """
    根据向量库名称获取服务URL。
    
    Args:
        db_name: 向量库名称（不含.chroma后缀）
    
    Returns:
        服务URL
    
    Raises:
        ValueError: 如果找不到对应的服务配置
    """
    # 支持从环境变量覆盖
    env_key = f"VECTOR_DB_SERVICE_{db_name.upper().replace('.', '_').replace('-', '_')}"
    url = os.environ.get(env_key)
    
    if url:
        return url
    
    url = VECTOR_DB_SERVICES.get(db_name)
    if not url:
        raise ValueError(
            f"No service URL found for vector database: {db_name}\n"
            f"Available databases: {list(VECTOR_DB_SERVICES.keys())}\n"
            f"You can also set environment variable: {env_key}"
        )
    
    return url


def list_available_databases() -> Dict[str, str]:
    """
    列出所有可用的向量库配置。
    
    Returns:
        {向量库名称: 服务URL} 字典
    """
    return VECTOR_DB_SERVICES.copy()

