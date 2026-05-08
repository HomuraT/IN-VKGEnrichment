"""
向量数据库 HTTP 客户端检索器

纯 HTTP 客户端实现，通过调用向量数据库服务进行检索。
所有本地向量库代码已移除，由服务端处理。
"""

from typing import List, Tuple
import httpx
from langchain_core.documents import Document

from src.config.logging_config import get_logger
from src.config.vector_db_services import get_service_url

logger = get_logger(__name__)


class VectorRetriever:
    """
    向量数据库 HTTP 客户端检索器
    
    通过 HTTP API 调用向量数据库服务进行检索。
    """
    
    def __init__(self, service_url: str, timeout: float = 30.0):
        """
        初始化 HTTP 客户端检索器
        
        Args:
            service_url: 向量数据库服务URL（例如：http://localhost:8001）
            timeout: 请求超时时间（秒）
        """
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=self.timeout)
        
        logger.debug(f"VectorRetriever initialized | service_url={self.service_url}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量，默认为5，如果为0则不查询直接返回空列表
            
        Returns:
            List[Document]: 检索到的相似文档列表
        """
        # 兼容 k=0 的情况，直接返回空列表
        if k <= 0:
            logger.debug(f"k={k} <= 0, skipping search and returning empty list")
            return []
        
        try:
            response = self.client.post(
                f"{self.service_url}/search",
                json={
                    "query": query,
                    "k": k,
                    "with_score": False,
                }
            )
            response.raise_for_status()
            
            data = response.json()
            documents = []
            
            for doc_dict in data.get("documents", []):
                doc = Document(
                    page_content=doc_dict.get("page_content", ""),
                    metadata=doc_dict.get("metadata", {}),
                )
                documents.append(doc)
            
            return documents
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during search: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Search failed: HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error during search: {e}")
            raise RuntimeError(f"Search failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        带相似度分数的检索
        
        Args:
            query: 查询文本  
            k: 返回的文档数量，默认为5，如果为0则不查询直接返回空列表
            
        Returns:
            List[Tuple[Document, float]]: (Document, score) 元组列表
        """
        # 兼容 k=0 的情况，直接返回空列表
        if k <= 0:
            logger.debug(f"k={k} <= 0, skipping search and returning empty list")
            return []
        
        try:
            response = self.client.post(
                f"{self.service_url}/search",
                json={
                    "query": query,
                    "k": k,
                    "with_score": True,
                }
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for doc_dict in data.get("documents", []):
                doc = Document(
                    page_content=doc_dict.get("page_content", ""),
                    metadata=doc_dict.get("metadata", {}),
                )
                score = doc_dict.get("score", 0.0)
                if score is None:
                    score = 0.0
                results.append((doc, float(score)))
            
            return results
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during search: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Search failed: HTTP {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error during search: {e}")
            raise RuntimeError(f"Search failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            bool: 服务是否正常
        """
        try:
            response = self.client.get(f"{self.service_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """关闭 HTTP 客户端"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_service_url_by_db_name(db_name: str) -> str:
    """
    根据向量库名称获取服务URL
    
    Args:
        db_name: 向量库名称（不含.chroma后缀）
    
    Returns:
        服务URL
    
    Raises:
        ValueError: 如果找不到对应的服务配置
    """
    return get_service_url(db_name)


def create_retriever_by_db_name(db_name: str, timeout: float = 30.0) -> VectorRetriever:
    """
    根据向量库名称创建检索器
    
    Args:
        db_name: 向量库名称（不含.chroma后缀）
        timeout: 请求超时时间（秒）
    
    Returns:
        VectorRetriever 实例
    """
    service_url = get_service_url_by_db_name(db_name)
    return VectorRetriever(service_url=service_url, timeout=timeout)


if __name__ == "__main__":
    """
    简单测试向量数据库 HTTP 检索功能
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="向量数据库 HTTP 检索测试")
    parser.add_argument("--service-url", "-u", required=True, help="服务URL（例如：http://localhost:8001）")
    parser.add_argument("--query", "-q", default="测试查询", help="测试查询")
    parser.add_argument("--k", "-k", type=int, default=3, help="返回结果数量")
    
    args = parser.parse_args()
    
    # 初始化检索器
    logger.info("初始化 HTTP 检索器...")
    retriever = VectorRetriever(args.service_url)
    
    # 健康检查
    if retriever.health_check():
        logger.success(f"✅ 服务健康检查通过: {args.service_url}")
    else:
        logger.error(f"❌ 服务不可用: {args.service_url}")
        exit(1)
    
    # 测试查询
    logger.info(f"\n测试查询: {args.query}")
    
    # 相似度检索
    logger.info("\n=== 相似度检索结果 ===")
    results = retriever.similarity_search(args.query, k=args.k)
    
    for i, doc in enumerate(results, 1):
        logger.info(f"\n结果 {i}:")
        logger.info(f"内容: {doc.page_content[:200]}...")
        logger.info(f"元数据: {doc.metadata}")
    
    # 带分数的检索
    logger.info("\n=== 带相似度分数的检索结果 ===")
    results_with_scores = retriever.similarity_search_with_score(args.query, k=args.k)
    
    for i, (doc, score) in enumerate(results_with_scores, 1):
        logger.info(f"\n结果 {i} (相似度分数: {score:.4f}):")
        logger.info(f"内容: {doc.page_content[:200]}...")
        logger.info(f"元数据: {doc.metadata}")
    
    retriever.close()
