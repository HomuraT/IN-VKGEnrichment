import os
import json
from typing import List
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config.api_and_models import apis, api_model_configs
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def build_vkg_mappings_vector_database(input_json_path: str, 
                                     output_directory: str,
                                     database_name: str = "textualized_vkg_mappings.chroma",
                                     embedding_model_key: str = "mmm_beta_text_embedding_3_small",
                                     mode: str = "full",
                                     chunk_size: int = 500,
                                     chunk_overlap: int = 100,
                                     batch_size: int = 256,
                                     resume: bool = True) -> str:
    """
    构建VKG映射向量数据库
    
    Args:
        input_json_path (str): 输入JSON文件路径
        output_directory (str): 输出目录路径
        database_name (str): 数据库名称，默认为"textualized_vkg_mappings.chroma"
        embedding_model_key (str): 嵌入模型配置键名，默认为"mmm_beta_text_embedding_3_small"
        mode (str): 运行模式，"test"表示只使用前5个映射，"full"表示使用全部数据，默认为"full"
        chunk_size (int): 文本分段大小，默认500字符
        chunk_overlap (int): 分段重叠大小，默认100字符
        
    Returns:
        str: 输出数据库路径
    """
    logger.info("开始构建VKG映射向量数据库...")
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
    
    # 3. 加载VKG映射数据（支持JSON和JSONL格式）
    logger.info("加载VKG映射数据...")
    
    # 判断文件格式
    if input_json_path.endswith('.jsonl'):
        # JSONL格式
        mappings = {}
        with open(input_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "mapping_id" in data:
                    # 这是一个mapping记录
                    mapping_id = data["mapping_id"]
                    mappings[mapping_id] = data
        # JSONL格式没有metadata，从文件名提取VKG名称
        # 文件名格式: <vkg_name>.<llm_model>.textualized_vkg_mappings.<mode>.jsonl
        filename = os.path.basename(input_json_path)
        vkg_name = filename.split('.')[0] if '.' in filename else "unknown"
        metadata = {
            "vkg_name": vkg_name,
            "input_file": input_json_path,
            "original_metadata": {"vkg_name": vkg_name},
            "original_prefixes": {}
        }
    else:
        # JSON格式
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mappings = data.get("mappings", {})
        metadata = data.get("metadata", {})
    
    # 根据mode参数限制数据量
    if mode == "test":
        mappings = dict(list(mappings.items())[:5])
        logger.info("测试模式: 只使用前5个映射")
    
    logger.info(f"加载了 {len(mappings)} 个VKG映射")
    
    # 4. 创建文档对象
    logger.info("创建文档对象...")
    documents = []
    
    # 统计信息
    total_detailed_segments = 0
    mapping_types = {}
    
    for mapping_id, mapping_data in tqdm(mappings.items(), desc="处理VKG映射"):
        original_data = mapping_data.get("original_data", {})
        descriptions = mapping_data.get("descriptions", {})
        
        # 从原始数据中提取映射信息
        target = original_data.get("target", "")
        source = original_data.get("source", "")
        
        # 统计映射类型（基于mapping_id的模式）
        mapping_category = _categorize_mapping(mapping_id)
        mapping_types[mapping_category] = mapping_types.get(mapping_category, 0) + 1
        
        # 基础metadata（不包含时间字段）
        base_metadata = {
            "mapping_id": mapping_id,
            "mapping_category": mapping_category,
            "target_pattern": target,
            "source_query": source,
            "detailed_description": descriptions.get("detailed", ""),
            "brief_description": descriptions.get("brief", ""),
            "source": "vkg_mappings",
            "vkg_name": metadata.get("original_metadata", {}).get("vkg_name", "unknown"),
            "input_file": metadata.get("input_file", ""),
        }
        
        # 添加前缀信息（复杂结构序列化为JSON字符串）
        prefixes = metadata.get("original_prefixes", {})
        if prefixes:
            base_metadata["prefixes"] = json.dumps(prefixes, ensure_ascii=False)
            base_metadata["prefix_count"] = len(prefixes)
        
        # 添加其他原始数据字段（排除时间相关字段；dict/list 统一序列化）
        for key, value in original_data.items():
            if key not in ["processed_at", "created_at", "last_updated"] and not key.endswith("_at"):
                if isinstance(value, (dict, list)):
                    base_metadata[f"original_{key}"] = json.dumps(value, ensure_ascii=False)
                else:
                    base_metadata[f"original_{key}"] = value
        
        # 1. 创建映射ID文档
        doc_mapping_id = Document(
            page_content=mapping_id,
            metadata={
                **base_metadata,
                "content_type": "mapping_id_only",
                "data_source": "raw_data_mapping_id"
            }
        )
        documents.append(doc_mapping_id)
        
        # 2. 创建Target模式文档
        if target:
            doc_target = Document(
                page_content=target,
                metadata={
                    **base_metadata,
                    "content_type": "target_pattern",
                    "data_source": "raw_data_rdf_target"
                }
            )
            documents.append(doc_target)
        
        # 3. 创建Source查询文档
        if source:
            doc_source = Document(
                page_content=source,
                metadata={
                    **base_metadata,
                    "content_type": "source_query",
                    "data_source": "raw_data_sql_source"
                }
            )
            documents.append(doc_source)
        
        # 4. 创建Brief描述文档
        brief_desc = descriptions.get("brief", "")
        if brief_desc:
            doc_brief = Document(
                page_content=brief_desc,
                metadata={
                    **base_metadata,
                    "content_type": "brief_description",
                    "data_source": "brief_description"
                }
            )
            documents.append(doc_brief)
        
        # 5. 创建Detailed描述文档（使用LangChain分段处理）
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
                        "data_source": "detailed_description",
                        "segment_index": i,
                        "total_segments": len(detailed_chunks)
                    }
                )
                documents.append(doc_detailed)
        
        # 6. 创建复合文档：映射ID + 简要描述
        if brief_desc:
            combined_content = f"Mapping: {mapping_id}\nDescription: {brief_desc}"
            doc_combined = Document(
                page_content=combined_content,
                metadata={
                    **base_metadata,
                    "content_type": "mapping_with_brief_description",
                    "data_source": "combined_id_brief"
                }
            )
            documents.append(doc_combined)
        
        # 7. 创建技术规格文档：Target + Source
        if target and source:
            tech_spec = f"Target RDF Pattern:\n{target}\n\nSource SQL Query:\n{source}"
            doc_tech = Document(
                page_content=tech_spec,
                metadata={
                    **base_metadata,
                    "content_type": "technical_specification",
                    "data_source": "combined_technical_spec"
                }
            )
            documents.append(doc_tech)
    
    logger.info(f"创建了 {len(documents)} 个文档")
    logger.info(f"映射类型统计: {mapping_types}")
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
        collection_name="vkg_mappings"
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
                mapping_id = d.metadata.get("mapping_id")
                content_type = d.metadata.get("content_type", "unknown")
                segment_index = d.metadata.get("segment_index")
                stable_id = f"vkg:{mapping_id}:{content_type}"
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

    # 在较新版本的 langchain_chroma.Chroma 中，指定 persist_directory 即会自动持久化；有的版本无 persist() 方法
    if hasattr(vectorstore, "persist"):
        try:
            vectorstore.persist()
        except Exception as e:
            logger.warning(f"调用 persist() 失败，将依赖自动持久化: {e}")
    else:
        logger.info("当前 Chroma 版本未提供 persist() 方法；已基于 persist_directory 自动持久化")
    
    logger.success("✅ VKG映射向量数据库构建完成!")
    logger.info(f"输出路径: {output_path}")
    
    return output_path


def _categorize_mapping(mapping_id: str) -> str:
    """
    根据mapping_id对映射进行分类
    
    Args:
        mapping_id (str): 映射标识符
        
    Returns:
        str: 映射类别
    """
    mapping_id_lower = mapping_id.lower()
    
    # 基于映射ID的模式识别映射类别
    if "expression" in mapping_id_lower:
        if "condition" in mapping_id_lower:
            return "expression_condition"
        elif "absence" in mapping_id_lower or "not_expressed" in mapping_id_lower:
            return "absence_expression"
        else:
            return "expression"
    elif "gene" in mapping_id_lower:
        return "gene"
    elif "species" in mapping_id_lower:
        return "species"
    elif "anatomical" in mapping_id_lower or "anatentity" in mapping_id_lower:
        return "anatomical_entity"
    elif "stage" in mapping_id_lower:
        return "developmental_stage"
    else:
        return "other"


def build_multiple_vkg_mappings_databases(input_directory: str, 
                                        output_directory: str,
                                        database_prefix: str = "textualized_vkg_mappings",
                                        embedding_model_key: str = "mmm_beta_text_embedding_3_small",
                                        mode: str = "full",
                                        chunk_size: int = 500,
                                        chunk_overlap: int = 100) -> List[str]:
    """
    批量构建多个VKG映射向量数据库
    
    Args:
        input_directory (str): 输入目录路径（包含多个映射JSON文件）
        output_directory (str): 输出目录路径
        database_prefix (str): 数据库名称前缀
        embedding_model_key (str): 嵌入模型配置键名
        mode (str): 运行模式
        chunk_size (int): 文本分段大小
        chunk_overlap (int): 分段重叠大小
        
    Returns:
        List[str]: 所有输出数据库路径列表
    """
    input_dir = Path(input_directory)
    if not input_dir.exists():
        logger.error(f"❌ 输入目录不存在: {input_directory}")
        return []
    
    # 查找所有textualized映射文件（支持JSON和JSONL）
    mapping_files = list(input_dir.glob("*.textualized_vkg_mappings.*.json*"))
    
    if not mapping_files:
        logger.error(f"❌ 在目录 {input_directory} 中未找到文本化映射文件")
        return []
    
    logger.info(f"🔍 发现 {len(mapping_files)} 个文本化映射文件:")
    for f in mapping_files:
        logger.info(f"  - {f.name}")
    
    output_paths = []
    
    # 处理每个映射文件
    for mapping_file in mapping_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 开始构建向量数据库: {mapping_file.name}")
        logger.info(f"{'='*60}")
        
        try:
            # 生成数据库名称
            file_stem = mapping_file.stem
            database_name = f"{database_prefix}_{file_stem}.chroma"
            
            output_path = build_vkg_mappings_vector_database(
                input_json_path=str(mapping_file),
                output_directory=output_directory,
                database_name=database_name,
                embedding_model_key=embedding_model_key,
                mode=mode,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=256
            )
            
            if output_path:
                output_paths.append(output_path)
                logger.success(f"✅ 成功构建: {mapping_file.name} -> {database_name}")
            else:
                logger.error(f"❌ 构建失败: {mapping_file.name}")
                
        except Exception as e:
            logger.error(f"❌ 处理文件 {mapping_file.name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.success(f"🎉 批量构建完成!")
    logger.info(f"成功构建: {len(output_paths)}/{len(mapping_files)} 个向量数据库")
    logger.info("输出数据库:")
    for output_path in output_paths:
        logger.info(f"  - {output_path}")
    
    return output_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建VKG映射向量数据库")
    parser.add_argument("--input", "-i", 
                       help="输入JSON文件路径或包含映射文件的目录路径")
    parser.add_argument("--output", "-o", 
                       default="resources/vector_databases", 
                       help="输出目录")
    parser.add_argument("--name", "-n", 
                       default="textualized_vkg_mappings.chroma", 
                       help="数据库名称（单文件模式）")
    parser.add_argument("--prefix", "-p", 
                       default="textualized_vkg_mappings", 
                       help="数据库名称前缀（批量模式）")
    parser.add_argument("--embedding-model", "-e", 
                       default="mmm_beta_text_embedding_3_small", 
                       help="嵌入模型配置键名")
    parser.add_argument("--mode", "-m", 
                       default="full", 
                       choices=["test", "full"], 
                       help="运行模式: test只使用前5个映射，full使用全部数据")
    parser.add_argument("--chunk-size", "-cs", 
                       type=int,
                       default=500, 
                       help="文本分段大小，默认500字符")
    parser.add_argument("--chunk-overlap", "-co", 
                       type=int,
                       default=100, 
                       help="分段重叠大小，默认100字符")
    parser.add_argument("--batch", "-b",
                       action='store_true',
                       help="批量处理目录中的所有映射文件")
    parser.add_argument("--batch-size", "-bs",
                       type=int,
                       default=256,
                       help="嵌入与写入的批大小（默认256）")
    parser.add_argument("--resume", dest="resume", action="store_true", help="启用断点续存（默认启用）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续存")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # 如果未指定输入，使用默认的文本化映射目录
    if not args.input:
        args.input = "resources/textualized_vkg_mappings"
        args.batch = True
        logger.info(f"🔄 未指定输入，使用默认文本化映射目录: {args.input}")
    
    # 检查输入路径是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 错误：输入路径不存在: {args.input}")
        exit(1)
    
    try:
        if args.batch or input_path.is_dir():
            # 批量处理目录中的所有映射文件
            output_paths = build_multiple_vkg_mappings_databases(
                input_directory=args.input,
                output_directory=args.output,
                database_prefix=args.prefix,
                embedding_model_key=args.embedding_model,
                mode=args.mode,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                resume=args.resume
            )
            
            if output_paths:
                logger.success(f"\n🎉 批量构建成功完成!")
                logger.info(f"构建了 {len(output_paths)} 个向量数据库")
            else:
                logger.error(f"\n❌ 批量构建失败或没有数据库被构建")
        else:
            # 处理单个映射文件
            output_path = build_vkg_mappings_vector_database(
                input_json_path=args.input,
                output_directory=args.output,
                database_name=args.name,
                embedding_model_key=args.embedding_model,
                mode=args.mode,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                resume=args.resume
            )
            
            if output_path:
                logger.success(f"\n🎉 构建成功完成!")
                logger.info(f"输出数据库: {output_path}")
            else:
                logger.error(f"\n❌ 构建失败")
        
    except Exception as e:
        logger.error(f"❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
