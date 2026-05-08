import os
import json
import argparse
import threading
from datetime import datetime
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from src.config.api_and_models import get_api_configuration
from src.vkg_agent import prompts
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# 全局锁，用于线程安全的文件操作
_file_lock = threading.Lock()


def _strip_think_blocks(text: str) -> str:
    """
    移除形如 <think>...</think> 的思考内容块（大小写不敏感，多行）。
    """
    return re.sub(r"(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", text).strip()

def textualize_vkg_mapping(mapping_data: Dict[str, Any], 
                          llm: ChatOpenAI,
                          description_type: str = "detailed") -> str:
    """
    使用LLM将单个VKG映射文本化
    
    Args:
        mapping_data (Dict[str, Any]): VKG映射数据
        llm (ChatOpenAI): 语言模型实例
        description_type (str): 描述类型（"detailed" 或 "brief"）
        
    Returns:
        str: 文本化描述
    """
    # 根据描述类型选择模板
    template_key = f"textualize_vkg_mapping_{description_type}"
    
    # 使用Jinja2模板
    template = PromptTemplate.from_template(
        prompts.vkg_mapping_prompts[template_key],
        template_format="jinja2"
    )
    
    # 渲染提示模板
    prompt_text = template.format(
        mapping_data=json.dumps(mapping_data, ensure_ascii=False, indent=2)
    )
    
    # 调用语言模型
    response = llm.invoke(prompt_text)
    
    # 如果响应是AIMessage对象，提取content
    if hasattr(response, 'content'):
        return _strip_think_blocks(response.content.strip())
    else:
        return _strip_think_blocks(str(response).strip())


def load_existing_results(output_path: str) -> Dict[str, Any]:
    """
    加载已存在的JSONL结果文件
    
    Args:
        output_path (str): 输出文件路径（.jsonl格式）
        
    Returns:
        Dict[str, Any]: 已存在的结果数据，如果文件不存在则返回空结构
    """
    if os.path.exists(output_path):
        mappings = {}
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "mapping_id" in data:
                    # 这是一个mapping记录
                    mapping_id = data["mapping_id"]
                    mappings[mapping_id] = data
        return {"metadata": {}, "mappings": mappings}
    else:
        return {"metadata": {}, "mappings": {}}


def append_mapping_result(mapping_id: str, mapping_result: Dict[str, Any], output_path: str):
    """
    追加单个mapping结果到JSONL文件（线程安全）
    
    Args:
        mapping_id (str): 映射ID
        mapping_result (Dict[str, Any]): 映射结果数据
        output_path (str): 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 构建JSONL行数据
    line_data = {
        "mapping_id": mapping_id,
        **mapping_result
    }
    
    # 追加到文件
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(line_data, ensure_ascii=False) + '\n')


def process_single_mapping(mapping_id: str, 
                          mapping_data: Dict[str, Any], 
                          llm: ChatOpenAI,
                          output_data: Dict[str, Any],
                          output_path: str) -> Optional[Dict[str, Any]]:
    """
    处理单个VKG映射，生成详细和简略两种描述
    
    Args:
        mapping_id (str): 映射ID
        mapping_data (Dict[str, Any]): 映射数据
        llm (ChatOpenAI): LLM实例
        output_data (Dict[str, Any]): 输出数据结构（用于统计）
        output_path (str): 输出文件路径
        
    Returns:
        Optional[Dict[str, Any]]: 处理结果，包含详细和简略描述，失败返回None
    """
    logger.info(f"🔄 处理映射: {mapping_id}")
    
    # 生成详细描述
    detailed_description = textualize_vkg_mapping(
        mapping_data, llm, "detailed"
    )
    
    # 生成简略描述
    brief_description = textualize_vkg_mapping(
        mapping_data, llm, "brief"
    )
    
    # 构建结果数据
    mapping_result = {
        "original_data": mapping_data,
        "descriptions": {
            "detailed": detailed_description,
            "brief": brief_description
        },
        "processed_at": datetime.now().isoformat()
    }
    
    # 线程安全地追加到JSONL文件
    with _file_lock:
        append_mapping_result(mapping_id, mapping_result, output_path)
        output_data["mappings"][mapping_id] = mapping_result
        output_data["metadata"]["total_mappings"] = len(output_data["mappings"])
    
    logger.success(f"✅ 完成映射: {mapping_id}")
    return mapping_result


def process_mapping_batch_threaded(mappings: List[Dict[str, Any]], 
                                  llm_config: Dict[str, Any],
                                  existing_mappings: Dict[str, Any],
                                  output_data: Dict[str, Any],
                                  output_path: str,
                                  max_workers: int = 5) -> int:
    """
    使用多线程处理一批VKG映射
    
    Args:
        mappings (List[Dict[str, Any]]): 要处理的映射列表
        llm_config (Dict[str, Any]): LLM配置
        existing_mappings (Dict[str, Any]): 已存在的映射
        output_data (Dict[str, Any]): 输出数据结构
        output_path (str): 输出文件路径
        max_workers (int): 最大线程数
        
    Returns:
        int: 成功处理的映射数量
    """
    # 过滤掉已经处理过的映射
    mappings_to_process = []
    for mapping in mappings:
        mapping_id = mapping.get("mapping_id")
        if mapping_id not in existing_mappings:
            mappings_to_process.append(mapping)
        else:
            logger.debug(f"⏭️ 跳过已处理的映射: {mapping_id}")
    
    if not mappings_to_process:
        logger.info("📋 所有映射已处理完成")
        return 0
    
    logger.info(f"🚀 开始多线程处理 {len(mappings_to_process)} 个映射，线程数: {max_workers}")
    
    processed_count = 0
    
    # 使用线程池处理映射
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个任务创建独立的LLM实例
        future_to_mapping_id = {}
        for mapping in mappings_to_process:
            mapping_id = mapping.get("mapping_id")
            llm = ChatOpenAI(**llm_config)  # 每个线程使用独立的LLM实例
            future = executor.submit(
                process_single_mapping,
                mapping_id, mapping, llm, output_data, output_path
            )
            future_to_mapping_id[future] = mapping_id
        
        # 使用tqdm显示进度
        with tqdm(total=len(mappings_to_process), desc="处理VKG映射") as pbar:
            for future in as_completed(future_to_mapping_id):
                mapping_id = future_to_mapping_id[future]
                result = future.result()
                if result is not None:
                    processed_count += 1
                    pbar.set_postfix({"已完成": processed_count})
                pbar.update(1)
    
    logger.success(f"🎉 映射批处理完成，成功处理 {processed_count}/{len(mappings_to_process)} 个映射")
    return processed_count


def process_vkg_mappings(input_json_path: str, 
                        output_directory: str, 
                        mode: str = "test",
                        llm_model_key: str = "mmm_beta_gpt_4o_mini",
                        max_workers: int = 5,
                        resume: bool = True) -> str:
    """
    处理VKG映射并生成文本化描述
    
    Args:
        input_json_path (str): 输入JSON文件路径
        output_directory (str): 输出目录路径
        mode (str): 运行模式，"test"处理前5个，"full"处理全部
        llm_model_key (str): LLM模型配置键名
        max_workers (int): 最大线程数
        
    Returns:
        str: 输出文件路径
    """
    logger.info("开始处理VKG映射文本化...")
    logger.info(f"输入文件: {input_json_path}")
    logger.info(f"输出目录: {output_directory}")
    logger.info(f"运行模式: {mode}")
    logger.info(f"最大线程数: {max_workers}")
    
    # 1. 获取LLM配置（不创建实例，在线程中创建）
    llm_config = get_api_configuration(llm_model_key)
    logger.info(f"已获取LLM配置: {llm_model_key}")
    
    # 2. 加载输入数据
    logger.info("正在加载VKG映射数据...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        mapping_file_data = json.load(f)
    
    # 提取映射列表
    mappings = mapping_file_data.get("mappings", [])
    if not mappings:
        logger.error("❌ 输入文件中未找到映射数据")
        return ""
    
    logger.info(f"✅ 成功加载 {len(mappings)} 个映射")
    
    # 3. 准备输出文件路径（包含 VKG 名称与 LLM 模型，统一命名风格）
    input_filename = Path(input_json_path).stem  # 形如 <vkg_name>_mappings
    # VKG 名称通常等于去掉后缀 "_mappings" 的文件名
    vkg_name = input_filename[:-9] if input_filename.endswith("_mappings") else input_filename
    llm_model_safe = llm_model_key.replace('/', '_').replace(' ', '_')
    output_filename = f"{vkg_name}.{llm_model_safe}.textualized_vkg_mappings.{mode}.jsonl"
    output_path = os.path.join(output_directory, output_filename)
    
    # 4. 加载已有结果（支持断点继续）
    logger.info("检查已有结果文件...")
    existing_data = load_existing_results(output_path)
    existing_mappings = existing_data.get("mappings", {})
    
    if not resume and existing_mappings:
        logger.info("⛔ 已禁用续存，本次将忽略已有结果并从头开始")
        existing_mappings = {}

    # 4.1 若开启续存且本次应处理的映射均已存在，则直接跳过文本化
    # 稳定集合：按映射 ID 排序后再截取 test 前 N
    ids_all = sorted({m.get("mapping_id") for m in mappings if m.get("mapping_id")})
    if mode == "test":
        ids_all = ids_all[:5]
    target_ids = set(ids_all)
    missing = [mid for mid in target_ids if mid not in existing_mappings]

    completed_flag = existing_data.get("metadata", {}).get("completed_at") is not None
    if resume and completed_flag and len(target_ids) > 0 and len(missing) == 0:
        logger.success("✅ 本次应处理的所有映射均已存在，跳过文本化步骤")
        logger.info(f"目标映射数: {len(target_ids)}，已存在: {len(target_ids)}")
        logger.info(f"输出文件: {output_path}")
        return output_path
    
    if existing_mappings:
        logger.info(f"🔄 发现已有结果文件，包含 {len(existing_mappings)} 个已处理映射")
        logger.info("将从上次中断处继续...")
    
    # 5. 准备输出数据结构
    output_data = {
        "metadata": {
            "created_at": existing_data.get("metadata", {}).get("created_at", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat(),
            "input_file": input_json_path,
            "mode": mode,
            "llm_model": llm_model_key,
            "description": f"Textualized VKG mappings from {input_filename}",
            "original_metadata": mapping_file_data.get("metadata", {}),
            "original_prefixes": mapping_file_data.get("prefixes", {})
        },
        "mappings": existing_mappings
    }
    
    # 6. 确定处理数量
    if mode == "test":
        # 稳定顺序：按映射 ID 排序后选前 5
        mappings = sorted(mappings, key=lambda m: m.get("mapping_id", ""))[:5]
        logger.info(f"🧪 测试模式: 仅处理前 {len(mappings)} 个映射")
    
    # 7. 处理VKG映射
    logger.info(f"\n🗺️ 正在处理VKG映射...")
    processed_count = process_mapping_batch_threaded(
        mappings, llm_config, existing_mappings, output_data, output_path, max_workers
    )
    
    # 8. 更新统计信息（不保存到文件，只用于日志输出）
    output_data["metadata"]["total_mappings"] = len(output_data["mappings"])
    output_data["metadata"]["completed_at"] = datetime.now().isoformat()
    output_data["metadata"]["processing_stats"] = {
        "mappings_processed": processed_count,
        "total_processed_this_run": processed_count,
        "max_workers": max_workers
    }
    
    logger.success(f"\n🎉 处理完成！")
    logger.info(f"本次处理了 {processed_count} 个新映射")
    logger.info(f"总共包含 {len(output_data['mappings'])} 个VKG映射")
    logger.info(f"输出文件: {output_path}")
    
    return output_path


def process_all_mappings_in_directory(mappings_directory: str,
                                     output_directory: str, 
                                     mode: str = "test",
                                     llm_model_key: str = "mmm_beta_gpt_4o_mini",
                                     max_workers: int = 5,
                                     resume: bool = True) -> List[str]:
    """
    处理目录中所有的VKG映射文件
    
    Args:
        mappings_directory (str): 映射文件目录路径
        output_directory (str): 输出目录路径
        mode (str): 运行模式
        llm_model_key (str): LLM模型配置键名
        max_workers (int): 最大线程数
        
    Returns:
        List[str]: 所有输出文件路径列表
    """
    mappings_dir = Path(mappings_directory)
    if not mappings_dir.exists():
        logger.error(f"❌ 映射目录不存在: {mappings_directory}")
        return []
    
    # 查找所有JSON映射文件
    mapping_files = list(mappings_dir.glob("*_mappings.json"))
    
    if not mapping_files:
        logger.error(f"❌ 在目录 {mappings_directory} 中未找到映射文件")
        return []
    
    logger.info(f"🔍 发现 {len(mapping_files)} 个映射文件:")
    for f in mapping_files:
        logger.info(f"  - {f.name}")
    
    output_files = []
    
    # 处理每个映射文件
    for mapping_file in mapping_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 开始处理映射文件: {mapping_file.name}")
        logger.info(f"{'='*60}")
        
        try:
            output_path = process_vkg_mappings(
                input_json_path=str(mapping_file),
                output_directory=output_directory,
                mode=mode,
                llm_model_key=llm_model_key,
                max_workers=max_workers,
                resume=resume
            )
            
            if output_path:
                output_files.append(output_path)
                logger.success(f"✅ 成功处理: {mapping_file.name}")
            else:
                logger.error(f"❌ 处理失败: {mapping_file.name}")
                
        except Exception as e:
            logger.error(f"❌ 处理文件 {mapping_file.name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*60}")
    logger.success(f"🎉 批量处理完成!")
    logger.info(f"成功处理: {len(output_files)}/{len(mapping_files)} 个文件")
    logger.info("输出文件:")
    for output_file in output_files:
        logger.info(f"  - {output_file}")
    
    return output_files


def main():
    """
    主函数，处理命令行参数并执行VKG映射文本化处理
    """
    parser = argparse.ArgumentParser(description="将VKG映射通过LLM生成文本化描述")
    
    parser.add_argument(
        "--input", 
        type=str, 
        help="输入的VKG映射JSON文件路径，或者包含映射文件的目录路径"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./resources/textualized_vkg_mappings",
        help="输出目录路径 (默认: ./resources/textualized_vkg_mappings)"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["test", "full"],
        default="test",
        help="运行模式：test处理前5个，full处理全部 (默认: test)"
    )
    
    parser.add_argument(
        "--llm_model", 
        type=str,
        default="mmm_beta_gpt_4o_mini",
        help="LLM模型配置键名 (默认: mmm_beta_gpt_4o_mini)"
    )
    
    parser.add_argument(
        "--max_workers", 
        type=int,
        default=5,
        help="最大线程数 (默认: 5)"
    )
    
    parser.add_argument(
        "--batch",
        action='store_true',
        help="批量处理目录中的所有映射文件"
    )
    parser.add_argument("--resume", dest="resume", action="store_true", help="启用断点续存（默认启用）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续存")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # 如果未指定输入，使用默认的映射目录
    if not args.input:
        args.input = "./resources/vkg_mappings_parsed"
        args.batch = True
        logger.info(f"🔄 未指定输入，使用默认映射目录: {args.input}")
    
    # 检查输入路径是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 错误：输入路径不存在: {args.input}")
        return
    
    if args.batch or input_path.is_dir():
        # 批量处理目录中的所有映射文件
        output_files = process_all_mappings_in_directory(
            mappings_directory=args.input,
            output_directory=args.output_dir,
            mode=args.mode,
            llm_model_key=args.llm_model,
            max_workers=args.max_workers,
            resume=args.resume
        )
        
        if output_files:
            logger.success(f"\n🎉 批量处理成功完成!")
            logger.info(f"处理了 {len(output_files)} 个文件")
        else:
            logger.error(f"\n❌ 批量处理失败或没有文件被处理")
    else:
        # 处理单个映射文件
        output_path = process_vkg_mappings(
            input_json_path=args.input,
            output_directory=args.output_dir,
            mode=args.mode,
            llm_model_key=args.llm_model,
            max_workers=args.max_workers,
            resume=args.resume
        )
        
        if output_path:
            logger.success(f"\n🎉 处理成功完成!")
            logger.info(f"输出文件: {output_path}")
        else:
            logger.error(f"\n❌ 处理失败")


if __name__ == "__main__":
    main()
