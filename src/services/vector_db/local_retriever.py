"""
服务端本地向量库检索器

复制自原 src/vkg_agent/retrievers.py，仅供服务端使用。
服务端负责加载和管理向量库，客户端通过 HTTP 调用。
"""

from typing import List, Tuple, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config.api_and_models import apis, api_model_configs
from src.config.logging_config import get_logger
import threading
import os

logger = get_logger(__name__)


# 进程级缓存：避免每个样本重复加载嵌入器与 Chroma 向量库
_EMBEDDINGS_CACHE: Dict[str, OpenAIEmbeddings] = {}
_CHROMA_CACHE: Dict[Tuple[str, str, str], Chroma] = {}
_CACHE_LOCK = threading.Lock()

# 可选：FAISS 内存索引缓存（按库/集合/嵌入器区分）
_FAISS_CACHE: Dict[Tuple[str, str, str], FAISS] = {}
_FAISS_LOCK = threading.Lock()

# 环境变量控制是否启用 FAISS 内存检索（默认关闭）
_USE_FAISS: bool = (str(os.environ.get("VKG_USE_FAISS_IN_MEMORY", "0")).strip() == "1")
_FAISS_GET_BATCH: int = int(os.environ.get("VKG_FAISS_GET_BATCH", "5000") or 5000)


class LocalVectorRetriever:
    """
    本地向量数据库检索器（仅供服务端使用）
    
    提供简单的向量相似度检索功能
    """
    
    def __init__(self, vector_db_path: str, collection_name: str = "default", embedding_model_key: str = "mmm_beta_text_embedding_3_small"):
        """
        初始化检索器
        
        Args:
            vector_db_path (str): 向量数据库路径
            collection_name (str): 集合名称，默认为"default"
            embedding_model_key (str): 嵌入模型配置键名，默认为"mmm_beta_text_embedding_3_small"
        """
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.embedding_model_key = embedding_model_key
        
        logger.info(f"Initializing LocalVectorRetriever | path={vector_db_path}, collection={collection_name}, embedding={embedding_model_key}")
        
        # 初始化嵌入模型（进程级缓存）
        api_config = api_model_configs.get(embedding_model_key, {})
        api_url_and_key = apis[api_config['api_name']]
        emb_key = embedding_model_key
        embeddings = _EMBEDDINGS_CACHE.get(emb_key)
        if embeddings is None:
            with _CACHE_LOCK:
                embeddings = _EMBEDDINGS_CACHE.get(emb_key)
                if embeddings is None:
                    logger.info(f"Creating embedding model: {embedding_model_key}")
                    embeddings = OpenAIEmbeddings(
                        model=api_config.get("model"),
                        openai_api_base=api_url_and_key.get("base_url"),
                        openai_api_key=api_url_and_key.get("api_key")
                    )
                    _EMBEDDINGS_CACHE[emb_key] = embeddings
        self.embeddings = embeddings

        # 加载向量数据库（进程级缓存）
        chroma_key = (vector_db_path, collection_name, embedding_model_key)
        vectorstore = _CHROMA_CACHE.get(chroma_key)
        if vectorstore is None:
            with _CACHE_LOCK:
                vectorstore = _CHROMA_CACHE.get(chroma_key)
                if vectorstore is None:
                    logger.info(f"Loading Chroma vector store from: {vector_db_path}")
                    vectorstore = Chroma(
                        persist_directory=vector_db_path,
                        embedding_function=self.embeddings,
                        collection_name=collection_name
                    )
                    _CHROMA_CACHE[chroma_key] = vectorstore
                    logger.success(f"Chroma vector store loaded successfully")
        self.vectorstore = vectorstore
        self._faiss_key: Tuple[str, str, str] = (self.vector_db_path, self.collection_name, self.embedding_model_key)
        self._faiss: Any = None
        if _USE_FAISS:
            # 延迟构建，首次查询时再触发；也可在此预构建
            pass
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度检索
        
        Args:
            query (str): 查询文本
            k (int): 返回的文档数量，默认为5
            
        Returns:
            List[Document]: 检索到的相似文档列表
        """
        f = self._get_or_build_faiss()
        if f is not None and hasattr(f, "similarity_search"):
            try:
                return f.similarity_search(query, k=k)
            except Exception:
                pass
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        带相似度分数的检索
        
        Args:
            query (str): 查询文本  
            k (int): 返回的文档数量，默认为5
            
        Returns:
            List[tuple]: (Document, score) 元组列表，score越小表示越相似
        """
        f = self._get_or_build_faiss()
        if f is not None:
            # 兼容不同方法名
            if hasattr(f, "similarity_search_with_score"):
                try:
                    return f.similarity_search_with_score(query, k=k)
                except Exception:
                    pass
            if hasattr(f, "similarity_search_with_relevance_scores"):
                try:
                    pairs = f.similarity_search_with_relevance_scores(query, k=k)
                    # 统一返回 (Document, score)
                    return [(doc, score) for (doc, score) in pairs]
                except Exception:
                    pass
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def get_info(self) -> Dict[str, Any]:
        """
        获取向量库信息
        
        Returns:
            包含向量库基本信息的字典
        """
        info = {
            "vector_db_path": self.vector_db_path,
            "collection_name": self.collection_name,
            "embedding_model_key": self.embedding_model_key,
        }
        
        # 尝试获取文档数量
        try:
            collection = getattr(self.vectorstore, "_collection", None)
            if collection:
                count = collection.count()
                info["document_count"] = int(count)
        except Exception as e:
            info["document_count"] = "unknown"
            info["count_error"] = str(e)
        
        return info

    # 内部：按需构建或返回 FAISS 索引
    def _get_or_build_faiss(self) -> Any:
        if not _USE_FAISS:
            return None
        if self._faiss is not None:
            return self._faiss
        # 命中缓存
        f_cached = _FAISS_CACHE.get(self._faiss_key)
        if f_cached is not None:
            self._faiss = f_cached
            return self._faiss
        # 双检加锁构建
        with _FAISS_LOCK:
            f_cached = _FAISS_CACHE.get(self._faiss_key)
            if f_cached is not None:
                self._faiss = f_cached
                return self._faiss
            built = self._build_faiss_from_chroma()
            if built is not None:
                _FAISS_CACHE[self._faiss_key] = built
                self._faiss = built
        return self._faiss

    def _build_faiss_from_chroma(self) -> Any:
        try:
            logger.info("Building FAISS index from Chroma...")
            collection = getattr(self.vectorstore, "_collection", None)
            if collection is None:
                return None
            total = 0
            try:
                total = int(collection.count())
            except Exception:
                total = 0
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            embeddings: List[List[float]] = []
            # 分批拉取，避免一次性占用过多内存
            offset = 0
            bs = max(1, int(_FAISS_GET_BATCH))
            while True:
                res: Dict[str, Any] = collection.get(
                    ids=None,
                    include=["documents", "metadatas", "embeddings"],
                    limit=bs,
                    offset=offset,
                )
                docs_batch = list(res.get("documents") or [])
                metas_batch = list(res.get("metadatas") or [])
                embs_batch = list(res.get("embeddings") or [])
                if not docs_batch:
                    break
                texts.extend([str(t or "") for t in docs_batch])
                metadatas.extend([dict(m or {}) for m in metas_batch])
                # 有的 Chroma 实例可能未持久 embeddings
                if embs_batch and len(embs_batch) == len(docs_batch):
                    embeddings.extend([[float(x) for x in v] for v in embs_batch])
                offset += len(docs_batch)
                if total and offset >= total:
                    break
            if not texts:
                return None
            # 若缺 embeddings，则现场计算一次
            if len(embeddings) != len(texts):
                logger.info(f"Computing embeddings for {len(texts)} documents...")
                embeddings = self.embeddings.embed_documents(texts)
            # 优先用 from_embeddings（保留 metadata）
            pairs = list(zip(texts, embeddings))
            faiss_index = FAISS.from_embeddings(text_embeddings=pairs, embedding=self.embeddings, metadatas=metadatas)
            logger.success(f"FAISS index built with {len(texts)} documents")
            return faiss_index
        except Exception as e:
            logger.warning(f"Failed to build FAISS index: {e}")
            return None

