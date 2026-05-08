import os
import json

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config.api_and_models import apis, api_model_configs
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def build_ontology_vector_database(input_json_path: str, 
                                 output_directory: str,
                                 database_name: str = "textualized_ontology_elements.chroma",
                                 embedding_model_key: str = "mmm_beta_text_embedding_3_small",
                                 mode: str = "full",
                                 chunk_size: int = 500,
                                 chunk_overlap: int = 100,
                                 batch_size: int = 256,
                                 resume: bool = True) -> str:
    """
    构建本体元素向量数据库
    
    Args:
        input_json_path (str): 输入JSON文件路径
        output_directory (str): 输出目录路径
        database_name (str): 数据库名称，默认为"textualized_ontology_elements.chroma"
        embedding_model_key (str): 嵌入模型配置键名，默认为"mmm_beta_text_embedding_3_small"
        mode (str): 运行模式，"test"表示只使用前10个元素，"full"表示使用全部数据，默认为"full"
        chunk_size (int): 文本分段大小，默认500字符
        chunk_overlap (int): 分段重叠大小，默认100字符
        
    Returns:
        str: 输出数据库路径
    """
    logger.info("开始构建本体元素向量数据库...")
    logger.info(f"输入文件: {input_json_path}")
    logger.info(f"输出目录: {output_directory}")
    logger.info(f"分段参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # 1. 初始化嵌入模型
    api_config = api_model_configs.get(embedding_model_key, {})
    api_url_and_key = apis[api_config['api_name']]
    embeddings = OpenAIEmbeddings(
        model=api_config.get("model"),
        openai_api_base=api_url_and_key.get("base_url"),
        openai_api_key=api_url_and_key.get("api_key")
    )
    
    # 2. 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "。", "!", "！", "?", "？", " ", ""]
    )
    
    # 3. 加载本体元素数据
    logger.info("加载本体元素数据...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    elements = data.get("elements", {})
    
    # 根据mode参数限制数据量
    if mode == "test":
        elements = dict(list(elements.items())[:10])
        logger.info("测试模式: 只使用前10个元素")
    
    logger.info(f"加载了 {len(elements)} 个本体元素")
    
    # 4. 创建文档对象
    logger.info("创建文档对象...")
    documents = []
    
    # 统计信息
    total_detailed_segments = 0
    element_types = {"Class": 0, "ObjectProperty": 0, "DataProperty": 0, "Other": 0}
    
    for uri, element_data in tqdm(elements.items(), desc="处理本体元素"):
        element_type = element_data.get("type", "Unknown")
        original_data = element_data.get("original_data", {})
        descriptions = element_data.get("descriptions", {})
        
        # 统计元素类型
        if element_type in element_types:
            element_types[element_type] += 1
        else:
            element_types["Other"] += 1
        
        # 基础metadata（不包含时间字段）
        base_metadata = {
            "uri": uri,
            "element_type": element_type,
            "detailed_description": descriptions.get("detailed", ""),
            "brief_description": descriptions.get("brief", ""),
            "source": "ontology_elements",
            "original_entity_type": original_data.get("entity_type", ""),
        }
        
        # 添加其他原始数据字段（排除时间相关字段）
        for key, value in original_data.items():
            if key not in ["processed_at", "created_at", "last_updated"] and not key.endswith("_at"):
                if isinstance(value, (dict, list)):
                    # 对复杂数据结构进行字符串化处理
                    base_metadata[f"original_{key}"] = str(value)
                else:
                    base_metadata[f"original_{key}"] = value
        
        # 1. 创建URI文档
        doc_uri = Document(
            page_content=uri,
            metadata={
                **base_metadata,
                "content_type": "uri_only"
            }
        )
        documents.append(doc_uri)
        
        # 2. 创建Brief描述文档
        brief_desc = descriptions.get("brief", "")
        if brief_desc:
            doc_brief = Document(
                page_content=brief_desc,
                metadata={
                    **base_metadata,
                    "content_type": "brief_description"
                }
            )
            documents.append(doc_brief)
        
        # 3. 创建Detailed描述文档（使用LangChain分段处理）
        detailed_desc = descriptions.get("detailed", "")
        if detailed_desc:
            # 使用LangChain文本分割器进行分段
            detailed_chunks = text_splitter.split_text(detailed_desc)
            total_detailed_segments += len(detailed_chunks)
            
            for i, chunk in enumerate(detailed_chunks):
                doc_detailed = Document(
                    page_content=chunk,
                    metadata={
                        **base_metadata,
                        "content_type": "detailed_description_segment",
                        "segment_index": i,
                        "total_segments": len(detailed_chunks)
                    }
                )
                documents.append(doc_detailed)
    
    logger.info(f"创建了 {len(documents)} 个文档")
    logger.info(f"元素类型统计: {element_types}")
    logger.info(f"详细描述总分段数: {total_detailed_segments}")
    
    # 4. 计算并打印平均文本长度
    total_length = 0
    text_count = 0
    content_type_stats = {}
    
    for doc in documents:
        content_length = len(doc.page_content)
        total_length += content_length
        text_count += 1
        
        content_type = doc.metadata.get("content_type", "unknown")
        if content_type not in content_type_stats:
            content_type_stats[content_type] = {"count": 0, "total_length": 0}
        content_type_stats[content_type]["count"] += 1
        content_type_stats[content_type]["total_length"] += content_length
    
    if text_count > 0:
        average_length = total_length / text_count
        logger.info(f"平均文本长度: {average_length:.2f} 字符")
        logger.info(f"总文本长度: {total_length} 字符")
        logger.info(f"文档总数: {text_count}")
        
        logger.info("各内容类型统计:")
        for content_type, stats in content_type_stats.items():
            avg_len = stats["total_length"] / stats["count"]
            logger.info(f"  {content_type}: {stats['count']} 个文档, 平均长度 {avg_len:.2f} 字符")
    else:
        logger.warning("没有找到有效的文档")
    
    # 5. 生成输出路径
    output_path = os.path.join(output_directory, database_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 6. 构建Chroma向量数据库（批量嵌入与写入，带进度条 + 断点续存）
    logger.info("构建向量数据库...")
    if len(documents) == 0:
        logger.warning("没有可写入的文档，跳过创建向量库")
        return output_path

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=output_path,
        collection_name="ontology_elements"
    )

    # 断点续存：读取已存在的 IDs
    existing_ids = set()
    if resume:
        try:
            raw = vectorstore.get(limit=None)
            if isinstance(raw, dict) and "ids" in raw:
                existing_ids = set(raw.get("ids", []))
        except Exception:
            try:
                raw = vectorstore._collection.get(include=[], limit=None)  # type: ignore[attr-defined]
                existing_ids = set(raw.get("ids", []))
            except Exception:
                existing_ids = set()

    total_docs = len(documents)
    bs = max(1, int(batch_size))
    added_total = 0
    skipped_total = 0
    with tqdm(total=total_docs, desc="嵌入与写入", unit="doc") as pbar:
        for start in range(0, total_docs, bs):
            end = min(start + bs, total_docs)
            batch = documents[start:end]
            ids = []
            docs_to_add = []
            for d in batch:
                uri = d.metadata.get("uri")
                content_type = d.metadata.get("content_type", "unknown")
                segment_index = d.metadata.get("segment_index")
                stable_id = f"onto:{uri}:{content_type}"
                if segment_index is not None:
                    stable_id = f"{stable_id}:{segment_index}"
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
    
    logger.success("✅ 本体元素向量数据库构建完成!")
    logger.info(f"输出路径: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建本体元素向量数据库")
    parser.add_argument("--input", "-i", 
                       default="resources/textualized_ontology_elements/textualized_bgee_v14_genex_test.json", 
                       help="输入JSON文件路径")
    parser.add_argument("--output", "-o", 
                       default="resources/vector_databases", 
                       help="输出目录")
    parser.add_argument("--name", "-n", 
                       default="textualized_ontology_elements.chroma", 
                       help="数据库名称")
    parser.add_argument("--embedding-model", "-e", 
                       default="mmm_beta_text_embedding_3_small", 
                       help="嵌入模型配置键名")
    parser.add_argument("--mode", "-m", 
                       default="full", 
                       choices=["test", "full"], 
                       help="运行模式: test只使用前10个元素，full使用全部数据")
    parser.add_argument("--chunk-size", "-cs", 
                       type=int,
                       default=500, 
                       help="文本分段大小，默认500字符")
    parser.add_argument("--chunk-overlap", "-co", 
                       type=int,
                       default=100, 
                       help="分段重叠大小，默认100字符")
    parser.add_argument("--batch-size", "-bs", 
                       type=int,
                       default=256,
                       help="嵌入与写入的批大小（默认256）")
    parser.add_argument("--resume", dest="resume", action="store_true", help="启用断点续存（默认启用）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续存")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    build_ontology_vector_database(
        args.input, 
        args.output, 
        args.name, 
        args.embedding_model, 
        args.mode,
        args.chunk_size,
        args.chunk_overlap,
        args.batch_size,
        args.resume
    )
