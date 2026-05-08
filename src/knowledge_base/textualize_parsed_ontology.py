import os
import json
import argparse
import threading
from datetime import datetime
import re
from typing import List, Dict, Any, Tuple, Optional
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

def textualize_ontology_element(element_data: Dict[str, Any], 
                               element_type: str, 
                               llm: ChatOpenAI,
                               description_type: str = "detailed") -> str:
    """
    使用LLM将单个本体元素文本化
    
    Args:
        element_data (Dict[str, Any]): 本体元素数据
        element_type (str): 元素类型（Class, ObjectProperty, DataProperty）
        llm (ChatOpenAI): 语言模型实例
        description_type (str): 描述类型（"detailed" 或 "brief"）
        
    Returns:
        str: 文本化描述
    """
    # 根据描述类型选择模板
    template_key = f"textualize_ontology_element_{description_type}"
    
    # 使用Jinja2模板
    template = PromptTemplate.from_template(
        prompts.ontology_prompts[template_key],
        template_format="jinja2"
    )
    
    # 渲染提示模板
    prompt_text = template.format(
        element_type=element_type,
        element_data=json.dumps(element_data, ensure_ascii=False, indent=2)
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
    加载已存在的结果文件
    
    Args:
        output_path (str): 输出文件路径
        
    Returns:
        Dict[str, Any]: 已存在的结果数据，如果文件不存在则返回空结构
    """
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {"metadata": {}, "elements": {}}


def save_results(output_data: Dict[str, Any], output_path: str):
    """
    线程安全地保存结果到文件
    
    Args:
        output_data (Dict[str, Any]): 要保存的数据
        output_path (str): 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 更新时间戳
    output_data["metadata"]["last_updated"] = datetime.now().isoformat()

    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def process_single_element(uri: str, 
                          element_data: Dict[str, Any], 
                          element_type: str,
                          llm: ChatOpenAI,
                          output_data: Dict[str, Any],
                          output_path: str) -> Optional[Dict[str, Any]]:
    """
    处理单个本体元素，生成详细和简略两种描述
    
    Args:
        uri (str): 元素URI
        element_data (Dict[str, Any]): 元素数据
        element_type (str): 元素类型
        llm (ChatOpenAI): LLM实例
        output_data (Dict[str, Any]): 输出数据结构
        output_path (str): 输出文件路径
        
    Returns:
        Optional[Dict[str, Any]]: 处理结果，包含详细和简略描述，失败返回None
    """
    logger.info(f"🔄 处理 {element_type}: {uri}")
    
    # 生成详细描述
    detailed_description = textualize_ontology_element(
        element_data, element_type, llm, "detailed"
    )
    
    # 生成简略描述
    brief_description = textualize_ontology_element(
        element_data, element_type, llm, "brief"
    )
    
    # 构建结果数据
    element_result = {
        "type": element_type,
        "original_data": element_data,
        "descriptions": {
            "detailed": detailed_description,
            "brief": brief_description
        },
        "processed_at": datetime.now().isoformat()
    }
    
    # 线程安全地更新输出数据并保存
    with _file_lock:
        output_data["elements"][uri] = element_result
        output_data["metadata"]["total_elements"] = len(output_data["elements"])
        save_results(output_data, output_path)
    
    logger.success(f"✅ 完成 {element_type}: {uri}")
    return element_result


def process_element_batch_threaded(items: List[Tuple[str, Dict]], 
                                  element_type: str,
                                  llm_config: Dict[str, Any],
                                  existing_elements: Dict[str, Any],
                                  output_data: Dict[str, Any],
                                  output_path: str,
                                  max_workers: int = 5) -> int:
    """
    使用多线程处理一批本体元素
    
    Args:
        items (List[Tuple[str, Dict]]): 要处理的元素列表
        element_type (str): 元素类型
        llm_config (Dict[str, Any]): LLM配置
        existing_elements (Dict[str, Any]): 已存在的元素
        output_data (Dict[str, Any]): 输出数据结构
        output_path (str): 输出文件路径
        max_workers (int): 最大线程数
        
    Returns:
        int: 成功处理的元素数量
    """
    # 过滤掉已经处理过的元素
    items_to_process = []
    for uri, element_data in items:
        if uri not in existing_elements:
            items_to_process.append((uri, element_data))
        else:
            logger.debug(f"⏭️ 跳过已处理的{element_type}: {uri}")
    
    if not items_to_process:
        logger.info(f"📋 所有 {element_type} 已处理完成")
        return 0
    
    logger.info(f"🚀 开始多线程处理 {len(items_to_process)} 个 {element_type}，线程数: {max_workers}")
    
    processed_count = 0
    
    # 使用线程池处理元素
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个任务创建独立的LLM实例
        future_to_uri = {}
        for uri, element_data in items_to_process:
            llm = ChatOpenAI(**llm_config)  # 每个线程使用独立的LLM实例
            future = executor.submit(
                process_single_element,
                uri, element_data, element_type, llm, output_data, output_path
            )
            future_to_uri[future] = uri
        
        # 使用tqdm显示进度
        with tqdm(total=len(items_to_process), desc=f"处理{element_type}") as pbar:
            for future in as_completed(future_to_uri):
                uri = future_to_uri[future]
                result = future.result()
                if result is not None:
                    processed_count += 1
                    pbar.set_postfix({"已完成": processed_count})
                pbar.update(1)
    
    logger.success(f"🎉 {element_type} 批处理完成，成功处理 {processed_count}/{len(items_to_process)} 个元素")
    return processed_count


def process_ontology_elements(input_json_path: str, 
                            output_directory: str, 
                            mode: str = "test",
                            llm_model_key: str = "mmm_beta_gpt_4o_mini",
                            max_workers: int = 5,
                            resume: bool = True) -> str:
    """
    处理本体元素并生成文本化描述
    
    Args:
        input_json_path (str): 输入JSON文件路径
        output_directory (str): 输出目录路径
        mode (str): 运行模式，"test"处理前10个，"full"处理全部
        llm_model_key (str): LLM模型配置键名
        max_workers (int): 最大线程数
        
    Returns:
        str: 输出文件路径
    """
    logger.info("开始处理本体元素文本化...")
    logger.info(f"输入文件: {input_json_path}")
    logger.info(f"输出目录: {output_directory}")
    logger.info(f"运行模式: {mode}")
    logger.info(f"最大线程数: {max_workers}")
    
    # 1. 获取LLM配置（不创建实例，在线程中创建）
    llm_config = get_api_configuration(llm_model_key)
    logger.info(f"已获取LLM配置: {llm_model_key}")
    
    # 2. 加载输入数据
    logger.info("正在加载本体数据...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)
    
    # 3. 准备输出文件路径（包含本体名与LLM模型，统一命名风格）
    input_filename = Path(input_json_path).stem  # 通常等于本体名称
    llm_model_safe = llm_model_key.replace('/', '_').replace(' ', '_')
    output_filename = f"{input_filename}.{llm_model_safe}.textualized_ontology_elements.{mode}.json"
    output_path = os.path.join(output_directory, output_filename)
    
    # 4. 加载已有结果（支持断点继续）
    logger.info("检查已有结果文件...")
    existing_data = load_existing_results(output_path)
    existing_elements = existing_data.get("elements", {})
    
    if not resume and existing_elements:
        logger.info("⛔ 已禁用续存，本次将忽略已有结果并从头开始")
        existing_elements = {}

    # 4.1 若开启续存且本次应处理的元素均已存在，则直接跳过文本化
    classes_all = ontology_data.get("classes", {}) or {}
    obj_props_all = ontology_data.get("object_properties", {}) or {}
    data_props_all = ontology_data.get("data_properties", {}) or {}
    limit_n = 10 if mode == "test" else None

    # 为保证 test 模式下的“前 N 个”集合稳定，按键名排序后再截取
    class_keys = sorted(classes_all.keys())
    obj_keys = sorted(obj_props_all.keys())
    data_keys = sorted(data_props_all.keys())
    if limit_n is not None:
        class_keys = class_keys[:limit_n]
        obj_keys = obj_keys[:limit_n]
        data_keys = data_keys[:limit_n]
    target_uris = set(class_keys + obj_keys + data_keys)
    missing = [uri for uri in target_uris if uri not in existing_elements]

    completed_flag = existing_data.get("metadata", {}).get("completed_at") is not None
    if resume and completed_flag and len(target_uris) > 0 and len(missing) == 0:
        logger.success("✅ 本次应处理的所有元素均已存在，跳过文本化步骤")
        logger.info(f"目标元素数: {len(target_uris)}，已存在: {len(target_uris)}")
        logger.info(f"输出文件: {output_path}")
        return output_path
    
    if existing_elements:
        logger.info(f"🔄 发现已有结果文件，包含 {len(existing_elements)} 个已处理元素")
        logger.info("将从上次中断处继续...")
    
    # 5. 准备输出数据结构
    output_data = {
        "metadata": {
            "created_at": existing_data.get("metadata", {}).get("created_at", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat(),
            "input_file": input_json_path,
            "mode": mode,
            "llm_model": llm_model_key,
            "description": f"Textualized ontology elements from {input_filename}"
        },
        "elements": existing_elements
    }
    
    # 6. 确定处理数量
    limit = 10 if mode == "test" else None
    
    # 7. 处理各类本体元素
    total_processed = 0
    
    # 处理Classes
    logger.info("\n📚 正在处理Classes...")
    classes = ontology_data.get("classes", {})
    # 稳定顺序：按 URI 排序
    class_items = sorted(classes.items(), key=lambda x: x[0])
    if limit:
        class_items = class_items[:limit]
    
    processed_classes = process_element_batch_threaded(
        class_items, "Class", llm_config, existing_elements, output_data, output_path, max_workers
    )
    total_processed += processed_classes
    
    # 处理Object Properties
    logger.info("\n🔗 正在处理Object Properties...")
    object_properties = ontology_data.get("object_properties", {})
    # 稳定顺序：按 URI 排序
    obj_prop_items = sorted(object_properties.items(), key=lambda x: x[0])
    if limit:
        obj_prop_items = obj_prop_items[:limit]
    
    processed_obj_props = process_element_batch_threaded(
        obj_prop_items, "ObjectProperty", llm_config, existing_elements, output_data, output_path, max_workers
    )
    total_processed += processed_obj_props
    
    # 处理Data Properties
    logger.info("\n📊 正在处理Data Properties...")
    data_properties = ontology_data.get("data_properties", {})
    # 稳定顺序：按 URI 排序
    data_prop_items = sorted(data_properties.items(), key=lambda x: x[0])
    if limit:
        data_prop_items = data_prop_items[:limit]
    
    processed_data_props = process_element_batch_threaded(
        data_prop_items, "DataProperty", llm_config, existing_elements, output_data, output_path, max_workers
    )
    total_processed += processed_data_props
    
    # 8. 最终保存结果和统计信息
    with _file_lock:
        output_data["metadata"]["total_elements"] = len(output_data["elements"])
        output_data["metadata"]["completed_at"] = datetime.now().isoformat()
        output_data["metadata"]["processing_stats"] = {
            "classes_processed": processed_classes,
            "object_properties_processed": processed_obj_props,
            "data_properties_processed": processed_data_props,
            "total_processed_this_run": total_processed,
            "max_workers": max_workers
        }
        save_results(output_data, output_path)
    
    logger.success(f"\n🎉 处理完成！")
    logger.info(f"本次处理了 {total_processed} 个新元素")
    logger.info(f"总共包含 {len(output_data['elements'])} 个本体元素")
    logger.info(f"- Classes: {processed_classes}")
    logger.info(f"- Object Properties: {processed_obj_props}")
    logger.info(f"- Data Properties: {processed_data_props}")
    logger.info(f"输出文件: {output_path}")
    
    return output_path


def main():
    """
    主函数，处理命令行参数并执行文本化处理
    """
    parser = argparse.ArgumentParser(description="将聚合后的本体元素通过LLM生成文本化描述")
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="输入的聚合本体JSON文件路径"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="./resources/textualized_ontology_elements",
        help="输出目录路径 (默认: ./resources/textualized_ontology_elements)"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["test", "full"],
        default="test",
        help="运行模式：test处理前10个，full处理全部 (默认: test)"
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
    parser.add_argument("--resume", dest="resume", action="store_true", help="启用断点续存（默认启用）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点续存")
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"❌ 错误：输入文件不存在: {args.input}")
        return
    
    # 执行处理
    output_path = process_ontology_elements(
        input_json_path=args.input,
        output_directory=args.output_dir,
        mode=args.mode,
        llm_model_key=args.llm_model,
        max_workers=args.max_workers,
        resume=args.resume
    )
    
    logger.success(f"\n🎉 处理成功完成!")
    logger.info(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
