import os
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config.api_and_models import apis, api_model_configs
from src.config.logging_config import get_logger


logger = get_logger(__name__)


def _format_po_summary(po_list: List[Tuple[str, str]]) -> str:
    """
    生成 (predicate, object) 列表的可读摘要文本。

    Args:
        po_list (List[Tuple[str, str]]): (谓语, 宾语) 列表

    Returns:
        str: 摘要文本
    """
    lines: List[str] = []
    for predicate, obj in po_list:
        lines.append(f"Predicate: {predicate} -> Object: {obj}")
    return "\n".join(lines)


def build_textualized_aggregated_triples_vector_database(input_json_path: str,
                                                         output_directory: str,
                                                         vkg_name: str = "bgee_v14_genex",
                                                         database_name: str = None,
                                                         embedding_model_key: str = "local_qwen_3_8b_embedding",
                                                         mode: str = "full",
                                                         chunk_size: int = 500,
                                                         chunk_overlap: int = 100,
                                                         batch_size: int = 256,
                                                         resume: bool = True) -> str:
    """
    构建"按主语聚合后已文本化"的向量数据库。

    Args:
        input_json_path (str): 输入JSON文件路径（形如 textualize_aggregated_triples.py 的输出）
        output_directory (str): 输出目录路径
        vkg_name (str): VKG名称，用于构建数据库名称，默认为"bgee_v14_genex"
        database_name (str): 数据库名称，如果为None则根据vkg_name和embedding_model_key自动生成
        embedding_model_key (str): 嵌入模型配置键名，默认为"local_qwen_3_8b_embedding"
        mode (str): 运行模式，"test"可用于限制数量（前100个subject），"full"使用全部
        chunk_size (int): 文本分段大小，默认500字符
        chunk_overlap (int): 分段重叠大小，默认100字符
        batch_size (int): 嵌入与写入的批大小，默认256
        resume (bool): 是否启用断点续存，默认True

    Returns:
        str: 输出数据库路径
    """
    logger.info("开始构建文本化聚合三元组向量数据库…")
    logger.info(f"输入文件: {input_json_path}")
    logger.info(f"输出目录: {output_directory}")
    logger.info(f"VKG名称: {vkg_name}")
    logger.info(f"分段参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # 生成数据库名称（如果未指定）
    if database_name is None:
        # 对嵌入与文本化 LLM 键名做简单安全化（替换 / 和 空格 为 _）
        embedding_model_safe = embedding_model_key.replace("/", "_").replace(" ", "_")
        # 从输入文件读取元数据中的 LLM 键（如果存在）
        try:
            with open(input_json_path, 'r', encoding='utf-8') as _f_meta:
                _meta_probe = json.load(_f_meta)
            llm_model_key = (_meta_probe.get("metadata") or {}).get("llm_model") or ""
        except Exception:
            llm_model_key = ""
        llm_model_safe = str(llm_model_key).replace("/", "_").replace(" ", "_") if llm_model_key else ""
        if llm_model_safe:
            database_name = f"{vkg_name}.{llm_model_safe}.{embedding_model_safe}.textualized_aggregated_triples.chroma"
        else:
            # 回退到仅含嵌入模型（兼容旧文件）
            database_name = f"{vkg_name}.{embedding_model_safe}.textualized_aggregated_triples.chroma"
    
    logger.info(f"数据库名称: {database_name}")

    # 1) 初始化嵌入模型
    api_config = api_model_configs.get(embedding_model_key, {})
    api_url_and_key = apis[api_config['api_name']]
    embeddings = OpenAIEmbeddings(
        model=api_config.get("model"),
        openai_api_base=api_url_and_key.get("base_url"),
        openai_api_key=api_url_and_key.get("api_key")
    )

    # 2) 初始化文本分割器（用于详细描述和长摘要）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "。", "!", "！", "?", "？", " ", ""]
    )

    # 3) 加载数据
    logger.info("加载文本化聚合数据…")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    metadata = data.get("metadata", {})
    subjects: Dict[str, Any] = data.get("subjects", {})

    subject_items: List[Tuple[str, Dict[str, Any]]] = list(subjects.items())
    if mode == "test":
        subject_items = subject_items[:100]
        logger.info("测试模式: 只使用前100个 subject")

    logger.info(f"加载了 {len(subject_items)} 个 subjects")

    # 4) 创建文档
    documents: List[Document] = []

    total_detailed_segments = 0
    content_type_stats: Dict[str, Dict[str, int]] = {}

    for entry_id, item in tqdm(subject_items, desc="处理 entries"):
        original_data: Dict[str, Any] = item.get("original_data", {})
        po_list: List[Tuple[str, str]] = original_data.get("po", [])
        triples_count: int = int(original_data.get("triples_count", len(po_list) or 0))
        descriptions: Dict[str, str] = item.get("descriptions", {})
        detailed_desc: str = descriptions.get("detailed", "")
        brief_desc: str = descriptions.get("brief", "")
        subject_uri: str = item.get("subject", "")
        chunk_info: Dict[str, Any] = item.get("chunk_info", {})

        # base_metadata 包含该聚合对象的完整上下文（复杂结构统一序列化为JSON字符串）
        base_metadata: Dict[str, Any] = {
            "entry_id": entry_id,
            "subject_uri": subject_uri,
            "triples_count": triples_count,
            "source": "textualized_aggregated_triples",
            "input_file": metadata.get("input_file", ""),
            "item_json": json.dumps(item, ensure_ascii=False),
            "original_data_json": json.dumps(original_data, ensure_ascii=False),
            "descriptions_json": json.dumps(descriptions, ensure_ascii=False),
            "processed_at": item.get("processed_at", ""),
            # 继承输入文件级元数据中的部分字段（如需要可扩展）
            "kb_mode": metadata.get("mode", ""),
            "kb_llm_model": metadata.get("llm_model", ""),
            "chunk_info": json.dumps(chunk_info, ensure_ascii=False),
        }

        # 1. subject URI 文档
        doc_subject = Document(
            page_content=subject_uri,
            metadata={**base_metadata, "content_type": "subject_uri_only"}
        )
        documents.append(doc_subject)

        # 2. 简要描述
        if brief_desc:
            doc_brief = Document(
                page_content=brief_desc,
                metadata={**base_metadata, "content_type": "brief_description"}
            )
            documents.append(doc_brief)

        # 3. 详细描述分段
        if detailed_desc:
            detailed_chunks = text_splitter.split_text(detailed_desc)
            total_detailed_segments += len(detailed_chunks)
            for i, chunk in enumerate(detailed_chunks):
                doc_detailed = Document(
                    page_content=chunk,
                    metadata={
                        **base_metadata,
                        "content_type": "detailed_description_segment",
                        "segment_index": i,
                        "total_segments": len(detailed_chunks),
                    }
                )
                documents.append(doc_detailed)

        # 4. 组合文档（Subject + Brief）
        if brief_desc:
            combined_content = f"Subject: {subject_uri}\nDescription: {brief_desc}"
            doc_combined = Document(
                page_content=combined_content,
                metadata={**base_metadata, "content_type": "subject_with_brief_description"}
            )
            documents.append(doc_combined)

        # 5. PO 摘要（必要时分段）
        if po_list:
            po_summary = _format_po_summary(po_list)
            po_chunks = text_splitter.split_text(po_summary) if len(po_summary) > chunk_size else [po_summary]
            for i, chunk in enumerate(po_chunks):
                doc_po = Document(
                    page_content=chunk,
                    metadata={
                        **base_metadata,
                        "content_type": "po_summary_segment",
                        "segment_index": i,
                        "total_segments": len(po_chunks),
                    }
                )
                documents.append(doc_po)

    logger.info(f"创建了 {len(documents)} 个文档")
    logger.info(f"详细描述总分段数: {total_detailed_segments}")

    # 5) 输出路径
    output_path = os.path.join(output_directory, database_name)
    os.makedirs(output_path, exist_ok=True)

    # 6) 构建向量数据库（Chroma，批量嵌入与写入，带进度条 + 断点续存）
    logger.info("构建向量数据库…")
    if len(documents) == 0:
        logger.warning("没有可写入的文档，跳过创建向量库")
        return output_path

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=output_path,
        collection_name="aggregated_triples",
    )

    # 断点续存：读取已存在的 IDs
    existing_ids = set()
    if resume:
        raw = vectorstore.get(limit=None)
        if isinstance(raw, dict) and "ids" in raw:
            existing_ids = set(raw.get("ids", []))

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
                entry_id = d.metadata.get("entry_id")
                content_type = d.metadata.get("content_type", "unknown")
                segment_index = d.metadata.get("segment_index")
                stable_id = f"agg:{entry_id}:{content_type}"
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

    # 在较新版本的 langchain_chroma.Chroma 中，持久化在指定 persist_directory 时会自动完成，且可能不存在 persist() 方法
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    else:
        logger.info("当前 Chroma 版本未提供 persist() 方法；已基于 persist_directory 自动持久化")

    logger.success("✅ 文本化聚合三元组向量数据库构建完成!")
    logger.info(f"输出路径: {output_path}")

    return output_path


def build_multiple_textualized_aggregated_triples_databases(input_directory: str,
                                                            output_directory: str,
                                                            vkg_name: str = "bgee_v14_genex",
                                                            embedding_model_key: str = "local_qwen_3_8b_embedding",
                                                            mode: str = "full",
                                                            chunk_size: int = 500,
                                                            chunk_overlap: int = 100,
                                                            batch_size: int = 256,
                                                            resume: bool = True) -> List[str]:
    """
    批量构建"按主语聚合后已文本化"的多个向量数据库。

    Args:
        input_directory (str): 输入目录路径（包含多个文本化JSON文件）
        output_directory (str): 输出目录路径
        vkg_name (str): VKG名称，用于构建数据库名称，默认为"bgee_v14_genex"
        embedding_model_key (str): 嵌入模型配置键名，默认为"local_qwen_3_8b_embedding"
        mode (str): 运行模式
        chunk_size (int): 文本分段大小
        chunk_overlap (int): 分段重叠大小
        batch_size (int): 嵌入与写入的批大小，默认256
        resume (bool): 是否启用断点续存，默认True

    Returns:
        List[str]: 所有输出数据库路径列表
    """
    input_dir = Path(input_directory)
    if not input_dir.exists():
        logger.error(f"❌ 输入目录不存在: {input_directory}")
        return []

    # textualize_aggregated_triples.py 的输出通常形如 textualized_*.json
    files = list(input_dir.glob("textualized_*.json"))
    if not files:
        logger.error(f"❌ 在目录 {input_directory} 中未找到文本化聚合文件")
        return []

    logger.info(f"🔍 发现 {len(files)} 个文本化聚合文件:")
    for f in files:
        logger.info(f"  - {f.name}")

    output_paths: List[str] = []
    for file_path in files:
        logger.info("\n" + "=" * 60)
        logger.info(f"🔄 开始构建向量数据库: {file_path.name}")
        logger.info("=" * 60)
        # 数据库名称将由主函数根据vkg_name和embedding_model_key自动生成
        output_path = build_textualized_aggregated_triples_vector_database(
            input_json_path=str(file_path),
            output_directory=output_directory,
            vkg_name=vkg_name,
            database_name=None,  # 让主函数自动生成名称
            embedding_model_key=embedding_model_key,
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            resume=resume,
        )
        if output_path:
            output_paths.append(output_path)
            db_name = Path(output_path).name
            logger.success(f"✅ 成功构建: {file_path.name} -> {db_name}")
        else:
            logger.error(f"❌ 构建失败: {file_path.name}")

    logger.info("\n" + "=" * 60)
    logger.success("🎉 批量构建完成!")
    logger.info(f"成功构建: {len(output_paths)}/{len(files)} 个向量数据库")
    for output_path in output_paths:
        logger.info(f"  - {output_path}")

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建文本化聚合三元组向量数据库")
    parser.add_argument("--input", "-i",
                        help="输入JSON文件路径或包含文本化文件的目录路径",
                        default="resources/textualized_aggregated_triples/bgee_v14_genex.local_qwen_2_5_7b.textualized_aggregated_triples.full.json")
    parser.add_argument("--output", "-o",
                        default="resources/vector_databases",
                        help="输出目录")
    parser.add_argument("--vkg-name", "-v",
                        default="bgee_v14_genex",
                        help="VKG名称，用于生成数据库名称")
    parser.add_argument("--name", "-n",
                        default=None,
                        help="数据库名称（单文件模式），如果未指定则根据vkg-name和embedding-model自动生成")
    parser.add_argument("--embedding-model", "-e",
                        default="local_qwen_3_8b_embedding",
                        help="嵌入模型配置键名")
    parser.add_argument("--mode", "-m",
                        default="full",
                        choices=["test", "full"],
                        help="运行模式: test仅使用前100个subject，full使用全部数据")
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
                        help="批量处理目录中的所有文本化文件")
    parser.add_argument("--batch-size", "-bs",
                        type=int,
                        default=256,
                        help="嵌入与写入的批大小（默认256）")
    parser.add_argument("--resume", dest="resume", action="store_true", help="启用断点续存（默认启用）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续存")
    parser.set_defaults(resume=True)

    args = parser.parse_args()

    # 若未指定输入，则默认处理目录
    if not args.input:
        args.input = "resources/textualized_aggregated_triples"
        args.batch = True
        logger.info(f"🔄 未指定输入，使用默认目录: {args.input}")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 错误：输入路径不存在: {args.input}")
        raise SystemExit(1)

    if args.batch or input_path.is_dir():
        output_paths = build_multiple_textualized_aggregated_triples_databases(
            input_directory=str(input_path),
            output_directory=args.output,
            vkg_name=getattr(args, 'vkg_name'),
            embedding_model_key=args.embedding_model,
            mode=args.mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        if output_paths:
            logger.success("\n🎉 批量构建成功完成!")
            logger.info(f"构建了 {len(output_paths)} 个向量数据库")
        else:
            logger.error("\n❌ 批量构建失败或没有数据库被构建")
    else:
        output_path = build_textualized_aggregated_triples_vector_database(
            input_json_path=str(input_path),
            output_directory=args.output,
            vkg_name=getattr(args, 'vkg_name'),
            database_name=args.name,
            embedding_model_key=args.embedding_model,
            mode=args.mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        if output_path:
            logger.success("\n🎉 构建成功完成!")
            logger.info(f"输出数据库: {output_path}")
        else:
            logger.error("\n❌ 构建失败")


