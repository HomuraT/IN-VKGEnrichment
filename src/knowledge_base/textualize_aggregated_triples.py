import os
import json
import argparse
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import re

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
    try:
        return re.sub(r"(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", text).strip()
    except Exception:
        return text



# 每个样本最多包含的 URI 对象 (predicate, object) 数量；非 URI 对象不受此限制
MAX_PO_PER_SAMPLE: int = 5


def _chunk_po_list(po_list: List[Tuple[str, str]], max_po: int = MAX_PO_PER_SAMPLE) -> List[List[Tuple[str, str]]]:
    """
    将 (predicate, object) 列表按固定大小切分为多个子列表。

    Args:
        po_list (List[Tuple[str, str]]): (谓语, 宾语) 列表
        max_po (int): 每个样本的最大 (谓语, 宾语) 数量

    Returns:
        List[List[Tuple[str, str]]]: 切分后的子列表集合（最后一组可少于 max_po）
    """
    return [po_list[i:i + max_po] for i in range(0, len(po_list), max_po)]


def _is_uri_value(value: str) -> bool:
    """
    判断一个对象值是否为 URI 形式（N3 序列化形如 <http://...>）。

    Args:
        value (str): 对象字符串（N3 格式）

    Returns:
        bool: 是否为 URI 值
    """
    v = value.strip()
    return v.startswith("<") and v.endswith(">")


def _select_po_with_uri_limit(po_list: List[Tuple[str, str]], max_uri_po: int = MAX_PO_PER_SAMPLE) -> List[Tuple[str, str]]:
    """
    选择 PO：
    - 保留所有“对象为非URI”的PO（如字符串文字、带语言/数据类型的文字）
    - 仅保留按出现顺序的前 max_uri_po 个“对象为URI”的PO

    Args:
        po_list (List[Tuple[str, str]]): (谓语, 宾语) 列表（N3 序列化）
        max_uri_po (int): 允许的 URI 对象 PO 的最大数量

    Returns:
        List[Tuple[str, str]]: 选择后的 PO 列表
    """
    selected: List[Tuple[str, str]] = []
    uri_kept: int = 0
    for predicate, obj in po_list:
        if _is_uri_value(obj):
            if uri_kept < max_uri_po:
                selected.append((predicate, obj))
                uri_kept += 1
        else:
            selected.append((predicate, obj))
    return selected


def load_existing_results(output_path: str) -> Dict[str, Any]:
    """
    加载已存在的结果文件

    Args:
        output_path (str): 输出文件路径

    Returns:
        Dict[str, Any]: 已存在的结果数据，如果文件不存在则返回空结构
    """
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 加载已有结果文件失败: {e}")
            return {"metadata": {}, "subjects": {}}
    else:
        return {"metadata": {}, "subjects": {}}


def save_results(output_data: Dict[str, Any], output_path: str) -> None:
    """
    线程安全地保存结果到文件

    Args:
        output_data (Dict[str, Any]): 要保存的数据
        output_path (str): 输出文件路径

    Returns:
        None
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 更新时间戳
    output_data["metadata"]["last_updated"] = datetime.now().isoformat()

    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def textualize_aggregated_subject(subject_uri: str,
                                  po_list: List[Tuple[str, str]],
                                  llm: ChatOpenAI,
                                  description_type: str = "detailed") -> str:
    """
    使用LLM将聚合的 subject+(p,o) 列表文本化

    Args:
        subject_uri (str): 主语 URI
        po_list (List[Tuple[str, str]]): (谓语, 宾语) 列表
        llm (ChatOpenAI): 语言模型实例
        description_type (str): 描述类型（"detailed" 或 "brief"）

    Returns:
        str: 文本化描述
    """
    try:
        template_key = f"textualize_aggregated_subject_{description_type}"
        template = PromptTemplate.from_template(
            prompts.aggregated_triples_prompts[template_key],
            template_format="jinja2",
        )
        subject_data = {
            "subject": subject_uri,
            "po": po_list,
            "triples_count": len(po_list),
        }
        prompt_text = template.format(
            subject_data=json.dumps(subject_data, ensure_ascii=False, indent=2)
        )
        response = llm.invoke(prompt_text)
        if hasattr(response, "content"):
            return _strip_think_blocks(response.content.strip())
        return _strip_think_blocks(str(response).strip())
    except Exception as e:
        logger.error(f"❌ LLM 调用失败: {e}")
        return f"Failed to generate {description_type} description: {str(e)}"


def process_single_subject(subject_uri: str,
                           po_list: List[Tuple[str, str]],
                           llm: ChatOpenAI,
                           output_data: Dict[str, Any],
                           output_path: str,
                           progress_callback: Optional[Callable[[str, int, int], None]] = None) -> Optional[int]:
    """
    处理单个 subject：仅取按顺序出现的前5个 (p,o)，生成单条文本化结果。

    Args:
        subject_uri (str): 主语 URI（原始 subject）
        po_list (List[Tuple[str, str]]): 原始 (谓语, 宾语) 列表
        llm (ChatOpenAI): 语言模型实例
        output_data (Dict[str, Any]): 输出数据结构（含 subjects 映射）
        output_path (str): 输出文件路径
        progress_callback (Optional[Callable[[str, int, int], None]]): 可选进度回调（此处仅在完成时回调一次）

    Returns:
        Optional[int]: 本次新增/写入的条目数（0或1）；失败返回 None
    """
    try:
        entry_id = subject_uri

        # 断点续存：若该 subject 已存在则跳过
        with _file_lock:
            if entry_id in output_data["subjects"]:
                return 0

        limited_po = _select_po_with_uri_limit(po_list, MAX_PO_PER_SAMPLE)

        detailed_description = textualize_aggregated_subject(
            subject_uri, limited_po, llm, "detailed"
        )
        brief_description = textualize_aggregated_subject(
            subject_uri, limited_po, llm, "brief"
        )

        result = {
            "id": entry_id,
            "subject": subject_uri,
            "original_data": {
                "po": limited_po,
                "triples_count": len(limited_po),
            },
            "descriptions": {
                "detailed": detailed_description,
                "brief": brief_description,
            },
            "processed_at": datetime.now().isoformat(),
        }

        with _file_lock:
            output_data["subjects"][entry_id] = result
            output_data["metadata"]["total_subjects"] = len(output_data["subjects"])  # 语义：条目=subject数
            save_results(output_data, output_path)

        if progress_callback is not None:
            progress_callback(subject_uri, 1, 1)

        return 1
    except Exception as e:
        logger.error(f"❌ 处理 subject 失败: {subject_uri}，错误: {e}")
        return None


def process_subjects_threaded(items: List[Tuple[str, List[Tuple[str, str]]]],
                              llm_config: Dict[str, Any],
                              existing_subjects: Dict[str, Any],
                              output_data: Dict[str, Any],
                              output_path: str,
                              max_workers: int = 5) -> int:
    """
    使用多线程处理一批 subjects。

    Args:
        items (List[Tuple[str, List[Tuple[str, str]]]]): (subject, po_list) 列表
        existing_subjects (Dict[str, Any]): 已存在的 subject 结果，用于断点续存
        output_data (Dict[str, Any]): 输出数据结构
        output_path (str): 输出文件路径
        max_workers (int): 最大线程数

    Returns:
        int: 新增的条目（entry）数量
    """
    # 过滤已处理
    items_to_process = []
    for subject_uri, po_list in items:
        if subject_uri not in existing_subjects:
            items_to_process.append((subject_uri, po_list))
        else:
            logger.debug(f"⏭️ 跳过已处理 subject: {subject_uri}")

    if not items_to_process:
        logger.info("📋 所有 subjects 已处理完成")
        return 0

    logger.info(f"🚀 开始多线程处理 {len(items_to_process)} 个 subjects，线程数: {max_workers}")

    processed_entries = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subject = {}
        for subject_uri, po_list in items_to_process:
            llm = ChatOpenAI(**llm_config)
            future = executor.submit(
                process_single_subject, subject_uri, po_list, llm, output_data, output_path, None
            )
            future_to_subject[future] = subject_uri

        with tqdm(total=len(items_to_process), desc="处理Subjects") as pbar:
            for future in as_completed(future_to_subject):
                subject_uri = future_to_subject[future]
                try:
                    new_entries = future.result()
                    if isinstance(new_entries, int):
                        processed_entries += new_entries
                        pbar.set_postfix({"新增条目": processed_entries})
                except Exception as e:
                    logger.error(f"❌ 线程处理 subject 异常: {subject_uri}，错误: {e}")
                finally:
                    pbar.update(1)


    logger.success(f"🎉 批处理完成，新增条目 {processed_entries}（subjects: {len(items_to_process)}）")
    return processed_entries


def load_aggregated_jsonl(jsonl_path: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """
    读取按主语聚合的 JSONL 文件，返回 (subject, po_list) 列表。

    Args:
        jsonl_path (str): 输入 JSONL 文件路径

    Returns:
        List[Tuple[str, List[Tuple[str, str]]]]: (subject, po_list) 列表
    """
    items: List[Tuple[str, List[Tuple[str, str]]]] = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            subject = obj.get("subject")
            po = obj.get("po", [])
            items.append((subject, po))
    return items


def textualize_aggregated_subjects(input_jsonl_path: str,
                                   output_directory: str,
                                   mode: str = "test",
                                   llm_model_key: str = "mmm_beta_gpt_4o_mini",
                                   max_workers: int = 5) -> str:
    """
    将聚合后的 subjects 文本化（不依赖 LLM），支持 test/full、断点续存和多线程。

    Args:
        input_jsonl_path (str): 输入 JSONL 文件
        output_directory (str): 输出目录
        mode (str): 运行模式，"test" 仅处理前 10 个，"full" 处理全部
        max_workers (int): 最大线程数

    Returns:
        str: 输出文件路径
    """
    logger.info("开始处理聚合三元组文本化…")
    logger.info(f"输入文件: {input_jsonl_path}")
    logger.info(f"输出目录: {output_directory}")
    logger.info(f"运行模式: {mode}")
    logger.info(f"最大线程数: {max_workers}")

    # 1. 获取LLM配置（在线程中创建实例）
    llm_config = get_api_configuration(llm_model_key)
    logger.info(f"已获取LLM配置: {llm_model_key}")

    # 2. 加载输入
    items = load_aggregated_jsonl(input_jsonl_path)
    logger.info(f"✅ 已加载 {len(items)} 个 subjects")

    # 3. 输出文件路径（包含 VKG 名称与 LLM 模型，统一命名风格）
    input_name = Path(input_jsonl_path).stem  # 形如 <vkg_name>-materialized.by_subject
    # 取 VKG 名称：按约定，'-' 之前部分即 vkg 名（如 bgee_v14_genex）
    vkg_name = input_name.split('-')[0] if '-' in input_name else input_name
    llm_model_safe = llm_model_key.replace('/', '_').replace(' ', '_')
    output_filename = f"{vkg_name}.{llm_model_safe}.textualized_aggregated_triples.{mode}.json"
    output_path = os.path.join(output_directory, output_filename)

    # 4. 断点续存
    logger.info("检查已有结果文件…")
    existing_data = load_existing_results(output_path)
    existing_subjects = existing_data.get("subjects", {})
    if existing_subjects:
        logger.info(f"🔄 发现已有结果文件，包含 {len(existing_subjects)} 个已处理 subjects，将继续…")

    # 5. 输出数据骨架
    output_data: Dict[str, Any] = {
        "metadata": {
            "created_at": existing_data.get("metadata", {}).get("created_at", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat(),
            "input_file": input_jsonl_path,
            "mode": mode,
            "llm_model": llm_model_key,
            "description": f"Textualized aggregated triples from {input_name}",
        },
        "subjects": existing_subjects,
    }

    # 6. 模式控制
    if mode == "test":
        items = items[:10]
        logger.info(f"🧪 测试模式：仅处理前 {len(items)} 个 subjects")

    # 7. 处理
    processed_count = process_subjects_threaded(
        items=items,
        llm_config=llm_config,
        existing_subjects=existing_subjects,
        output_data=output_data,
        output_path=output_path,
        max_workers=max_workers,
    )

    # 8. 收尾
    with _file_lock:
        output_data["metadata"]["total_subjects"] = len(output_data["subjects"])
        output_data["metadata"]["completed_at"] = datetime.now().isoformat()
        output_data["metadata"]["processing_stats"] = {
            "new_entries": processed_count,
            "subjects_attempted": len(items),
            "max_workers": max_workers,
        }
        save_results(output_data, output_path)

    logger.success("🎉 文本化处理完成！")
    logger.info(f"输出文件: {output_path}")
    return output_path


def main() -> None:
    """
    主函数：文本化聚合结果（支持 test/full、断点续存、多线程）。
    """
    parser = argparse.ArgumentParser(description="Textualize aggregated triples (by subject) using LLM")
    parser.add_argument(
        "--input",
        type=str,
        default="resources/vkg_triples_aggregated/bgee_v14_genex-materialized.by_subject.jsonl",
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./resources/textualized_aggregated_triples",
        help="输出目录 (默认: ./resources/textualized_aggregated_triples)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "full"],
        default="test",
        help="运行模式：test 处理前10个，full 处理全部 (默认: test)"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="local_qwen_2_5_7b",
        help="LLM模型配置键名 (默认: mmm_beta_gpt_4o_mini)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="最大线程数 (默认: 5)"
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"❌ 错误：输入文件不存在: {args.input}")
        return

    try:
        output_path = textualize_aggregated_subjects(
            input_jsonl_path=args.input,
            output_directory=args.output_dir,
            mode=args.mode,
            llm_model_key=args.llm_model,
            max_workers=args.max_workers,
        )
        logger.success("✅ 处理成功")
        logger.info(f"输出文件: {output_path}")
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


