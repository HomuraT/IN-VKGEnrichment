import os
import json
from datetime import datetime
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from tqdm import tqdm

from src.config.api_and_models import apis, api_model_configs
from src.config.logging_config import get_logger, log_function_call

logger = get_logger(__name__)


@log_function_call(include_args=False, include_result=False, log_level="INFO")
def build_vector_database(input_json_path: str, 
                         output_directory: str,
                         database_name: str = None,
                         embedding_model_key: str = "mmm_beta_text_embedding_3_small",
                         mode: str = "full",
                         batch_size: int = 256,
                         resume: bool = True) -> str:
    """
    构建向量数据库
    
    Args:
        input_json_path (str): 输入JSON知识库文件路径
        output_directory (str): 输出目录路径
        database_name (str, optional): 数据库名称，默认基于输入文件名生成
        embedding_model_key (str): 嵌入模型配置键名，默认为"mmm_beta_text_embedding_3_small"
        mode (str): 运行模式，"test"表示只使用前100条数据，"full"表示使用全部数据，默认为"full"
        
    Returns:
        str: 输出数据库路径
    """
    logger.info("开始构建向量数据库...")
    logger.info(f"输入文件: {input_json_path}")
    logger.info(f"输出目录: {output_directory}")
    
    # 1. 初始化嵌入模型
    api_config = api_model_configs.get(embedding_model_key, {})
    api_url_and_key = apis[api_config['api_name']]
    embeddings = OpenAIEmbeddings(
        model=api_config.get("model"),
        openai_api_base=api_url_and_key.get("base_url"),
        openai_api_key=api_url_and_key.get("api_key")
    )
    
    # 2. 加载知识库数据
    logger.info("加载知识库数据...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = data.get("examples", [])
    
    # 根据mode参数限制数据量
    if mode == "test":
        examples = examples[:100]
        logger.info("测试模式: 只使用前100条示例")
    
    logger.info(f"加载了 {len(examples)} 条示例")
    
    # 3. 创建文档对象
    logger.info("创建文档对象...")
    documents = []
    
    for i, example in enumerate(tqdm(examples, desc="处理示例")):
        text = example.get("text", "")
        sparql = example.get("SPARQL", "")
        
        if not text or not sparql:
            continue
        
        # 组合内容
        combined_content = f"Question:\n{text}\n\nSPARQL Query:\n{sparql}"
        
        # 基础metadata
        base_metadata = {
            "example_id": i,
            "source": "text_to_sparql_kb",
            "combined_content": combined_content
        }
        
        # 创建三个文档：组合内容、纯文本、纯SPARQL查询
        # 1. 组合内容文档
        doc_combined = Document(
            page_content=combined_content,
            metadata={
                **base_metadata,
                "content_type": "combined"
            }
        )
        documents.append(doc_combined)
        
        # 2. 纯文本文档
        doc_text = Document(
            page_content=text,
            metadata={
                **base_metadata,
                "content_type": "text_only"
            }
        )
        documents.append(doc_text)
        
        # 3. 纯SPARQL查询文档
        doc_sparql = Document(
            page_content=sparql,
            metadata={
                **base_metadata,
                "content_type": "sparql_only"
            }
        )
        documents.append(doc_sparql)
    
    logger.info(f"创建了 {len(documents)} 个文档 (每个示例3个文档: combined_content、text_only、sparql_only)")
    
    # 4. 计算并打印平均文本长度
    total_length = 0
    text_count = 0
    
    for doc in documents:
        content_length = len(doc.page_content)
        total_length += content_length
        text_count += 1
    
    if text_count > 0:
        average_length = total_length / text_count
        logger.info(f"平均文本长度: {average_length:.2f} 字符")
        logger.info(f"总文本长度: {total_length} 字符")
        logger.info(f"文档总数: {text_count}")
    else:
        logger.warning("没有找到有效的文档")
    
    # 5. 生成输出路径
    if database_name is None:
        input_filename = Path(input_json_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        database_name = f"{input_filename}_chroma_{timestamp}"
    
    output_path = os.path.join(output_directory, database_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 6. 构建Chroma向量数据库（批量嵌入与写入，带进度条）
    logger.info("构建向量数据库...")
    if len(documents) == 0:
        logger.warning("没有可写入的文档，跳过创建向量库")
        return output_path

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=output_path,
        collection_name="text_to_sparql"
    )

    # 断点续存：拉取已存在的 IDs
    existing_ids = set()
    if resume:
        try:
            # 避免取出 embeddings/text，减少IO
            raw = vectorstore.get(limit=None)
            if isinstance(raw, dict) and "ids" in raw:
                existing_ids = set(raw.get("ids", []))
        except Exception:
            try:
                # 兼容旧版本
                raw = vectorstore._collection.get(include=[], limit=None)  # type: ignore[attr-defined]
                existing_ids = set(raw.get("ids", []))
            except Exception:
                existing_ids = set()

    # 进度条：按文档数展示嵌入与写入进度（含跳过）
    total_docs = len(documents)
    bs = max(1, int(batch_size))
    added_total = 0
    skipped_total = 0
    with tqdm(total=total_docs, desc="嵌入与写入", unit="doc") as pbar:
        for start in range(0, total_docs, bs):
            end = min(start + bs, total_docs)
            batch = documents[start:end]
            # 基于稳定ID去重：example_id + content_type
            ids = []
            docs_to_add = []
            for d in batch:
                example_id = d.metadata.get("example_id")
                content_type = d.metadata.get("content_type", "unknown")
                stable_id = f"t2s:{example_id}:{content_type}"
                if (not resume) or (stable_id not in existing_ids):
                    ids.append(stable_id)
                    docs_to_add.append(d)
                else:
                    skipped_total += 1
            if docs_to_add:
                vectorstore.add_documents(docs_to_add, ids=ids)
                added_total += len(docs_to_add)
            pbar.set_postfix({"新增": added_total, "跳过": skipped_total})
            pbar.update(len(batch))

    # 在较新版本的 langchain_chroma.Chroma 中，指定 persist_directory 即会自动持久化；部分版本无 persist() 方法
    if hasattr(vectorstore, "persist"):
        try:
            vectorstore.persist()
        except Exception as e:
            logger.warning(f"调用 persist() 失败，将依赖自动持久化: {e}")
    else:
        logger.info("当前 Chroma 版本未提供 persist() 方法；已基于 persist_directory 自动持久化")
    
    logger.success("✅ 向量数据库构建完成!")
    logger.info(f"输出路径: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建Text-to-SPARQL向量数据库")
    parser.add_argument("--input", "-i", default= "resources/text_to_sparql_examples/text_to_sparql_kb_20250916_080521.json", help="输入JSON文件路径")
    parser.add_argument("--output", "-o", default="resources/vector_databases", help="输出目录")
    parser.add_argument("--name", "-n", default="text_to_sparql_vector_db.chroma", help="数据库名称")
    parser.add_argument("--embedding-model", "-e", default="mmm_beta_text_embedding_3_small", help="嵌入模型配置键名")
    parser.add_argument("--mode", "-m", default="full", choices=["test", "full"], help="运行模式: test只使用前100条数据，full使用全部数据")
    parser.add_argument("--batch-size", "-bs", type=int, default=256, help="嵌入与写入的批大小（默认256）")
    parser.add_argument("--resume", dest="resume", action="store_true", help="启用断点续存（默认启用）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续存")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    build_vector_database(args.input, args.output, args.name, args.embedding_model, args.mode, args.batch_size, args.resume)
