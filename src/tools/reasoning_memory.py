import uuid
from typing import List, Dict, Any, Optional, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config.api_and_models import apis, api_model_configs
from src.config.logging_config import get_logger


logger = get_logger(__name__)


class ReasoningMemory:
    """
    仅内存的推理内存模块（不落地）。

    - 支持添加记忆（文本+向量）
    - 支持相似度检索
    - 使用 LangChain 的 OpenAIEmbeddings + FAISS（内存向量库）
    - 无本地持久化；推荐每个样本单独实例化以避免并发冲突（无锁）
    """

    def __init__(self, embedding_model_key: str = "mmm_beta_text_embedding_3_small") -> None:
        """
        初始化内存向量存储。

        Args:
            embedding_model_key (str): 嵌入模型配置键名，默认 "mmm_beta_text_embedding_3_small"。

        Returns:
            None
        """
        api_config = api_model_configs.get(embedding_model_key, {})
        api_url_and_key = apis[api_config['api_name']]

        self._embeddings_model: OpenAIEmbeddings = OpenAIEmbeddings(
            model=api_config.get("model"),
            openai_api_base=api_url_and_key.get("base_url"),
            openai_api_key=api_url_and_key.get("api_key"),
        )

        # 存储内容
        self._texts: Dict[str, str] = {}
        self._metadatas: Dict[str, Dict[str, Any]] = {}
        self._vectorstore: Optional[FAISS] = None

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None, memory_id: Optional[str] = None) -> str:
        """
        添加一条记忆（文本+向量）。

        Args:
            text (str): 记忆的原始文本内容。
            metadata (Optional[Dict[str, Any]]): 关联的元数据，可为空。
            memory_id (Optional[str]): 可选的记忆 ID；若不提供将自动生成 UUID4。

        Returns:
            str: 最终写入的记忆 ID。
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        memory_id = memory_id or str(uuid.uuid4())

        self._texts[memory_id] = text
        meta = dict(metadata or {})
        meta.setdefault("memory_id", memory_id)
        self._metadatas[memory_id] = meta

        # 初始化或增量添加到内存向量库（FAISS）
        if self._vectorstore is None:
            # 首次添加，构建索引
            self._vectorstore = FAISS.from_texts(
                texts=[text],
                embedding=self._embeddings_model,
                metadatas=[meta],
                ids=[memory_id],
            )
        else:
            self._vectorstore.add_texts(
                texts=[text],
                metadatas=[meta],
                ids=[memory_id],
            )

        return memory_id

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度检索，返回最相似的前 k 条记忆（不含分数）。

        Args:
            query (str): 查询文本。
            k (int): 返回数量（top-k）。

        Returns:
            List[Document]: 匹配到的文档列表。
        """
        ranked = self._rank_by_similarity(query, k)
        return [doc for doc, _ in ranked]

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        相似度检索，返回 (Document, score)。
        这里的 score 为 cosine 相似度，范围约为 [-1, 1]，越大越相似。

        Args:
            query (str): 查询文本。
            k (int): 返回数量（top-k）。

        Returns:
            List[Tuple[Document, float]]: (文档, 距离) 列表。
        """
        return self._rank_by_similarity(query, k)

    def count(self) -> int:
        """
        返回当前记忆条目数量。

        Args:
            None

        Returns:
            int: 记忆数量。
        """
        return len(self._texts)

    def clear(self) -> None:
        """
        清空所有记忆。

        Args:
            None

        Returns:
            None
        """
        self._texts.clear()
        self._metadatas.clear()
        self._vectorstore = None

    # 内部方法
    def _rank_by_similarity(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        计算查询与所有记忆的相关性分数并排序，返回前 k。

        Args:
            query (str): 查询文本。
            k (int): 返回数量。

        Returns:
            List[Tuple[Document, float]]: (Document, score) 列表；score 为 cosine 相似度。
        """
        if not isinstance(query, str) or not query.strip():
            return []

        if self._vectorstore is None:
            return []

        try:
            results = self._vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"vectorstore similarity_search_with_score failed: {e}")
            return []

        return results


def _run_demo() -> None:
    """
    运行示例：添加较多记忆，演示两种检索迭代方式：
    1) 稳定模式：每轮都返回当前 top-1 结果
    2) 唯一模式：每轮返回未出现过的新结果

    Args:
        None

    Returns:
        None
    """
    # 固定配置（无命令行参数）
    embedding_model_key = "mmm_beta_text_embedding_3_small"
    k = 3
    rounds = 3
    queries = [
        "最终结论",
        "性能优化",
        "发布与回滚",
        "安全与鉴权",
        "一致性策略",
    ]

    logger.info("Initializing ReasoningMemory…")
    memory = ReasoningMemory(embedding_model_key=embedding_model_key)

    logger.info("Adding demo memories…")
    demo_docs = [
        ("今天的推理得到：系统A在条件X下性能提升明显。", {"step": 1, "tag": "analysis"}),
        ("我们进一步推断：瓶颈可能出现在数据库连接池。", {"step": 2, "tag": "hypothesis"}),
        ("最终结论：需要扩大连接池并增加读写分离策略。", {"step": 3, "tag": "conclusion"}),
        ("为保障高并发读写，建议引入缓存层与异步队列。", {"step": 4, "tag": "architecture"}),
        ("监控数据显示延迟主要集中在磁盘IO与网络抖动。", {"step": 5, "tag": "observation"}),
        ("实验对照：在相同负载下，新索引方案降低查询耗时25%。", {"step": 6, "tag": "experiment"}),
        ("模型推断：用户行为高峰发生在工作日午间与晚间时段。", {"step": 7, "tag": "pattern"}),
        ("根因分析：服务间重试风暴导致级联超时。", {"step": 8, "tag": "rca"}),
        ("改进建议：限流熔断与重试退避策略需要细化配置。", {"step": 9, "tag": "mitigation"}),
        ("安全考量：鉴权模块需引入细粒度权限与审计日志。", {"step": 10, "tag": "security"}),
        ("数据一致性：采用最终一致策略并辅以补偿任务。", {"step": 11, "tag": "consistency"}),
        ("上线方案：灰度发布并实时回滚监控阈值。", {"step": 12, "tag": "release"}),
        ("评估指标：P95 响应时间与错误率需持续跟踪。", {"step": 13, "tag": "metrics"}),
        ("团队协作：建立跨服务故障应急演练机制。", {"step": 14, "tag": "process"}),
    ]
    for txt, meta in demo_docs:
        memory.add_memory(txt, meta)

    logger.info(f"Memory count: {memory.count()}")

    # 对每个查询做多轮迭代检索：先稳定模式、再唯一模式
    sep = "=" * 40
    for q in queries:
        # 稳定模式
        logger.info(sep)
        logger.info(f"Stable mode | query: {q}")
        for round_idx in range(1, rounds + 1):
            k_eff = max(1, k)
            candidates = memory.similarity_search_with_score(q, k=k_eff)
            if not candidates:
                logger.info(f"Round {round_idx}: no results")
                break
            doc, score = candidates[0]
            logger.info(f"Round {round_idx}: score={score:.4f}")
            logger.info(f"Content: {doc.page_content}")
            logger.info(f"Metadata: {doc.metadata}")

        # 唯一模式
        logger.info(sep)
        logger.info(f"Unique mode | query: {q}")
        seen_ids = set()
        for round_idx in range(1, rounds + 1):
            k_eff = max(k, memory.count())
            candidates = memory.similarity_search_with_score(q, k=k_eff)

            picked = None
            for doc, score in candidates:
                mem_id = doc.metadata.get("memory_id")
                if mem_id not in seen_ids:
                    picked = (doc, score)
                    seen_ids.add(mem_id)
                    break

            if picked is None:
                logger.info(f"Round {round_idx}: no new results (all seen)")
                break

            doc, score = picked
            logger.info(f"Round {round_idx}: score={score:.4f}")
            logger.info(f"Content: {doc.page_content}")
            logger.info(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    _run_demo()