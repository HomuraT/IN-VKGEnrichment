"""
批量分析实验结果并生成汇总表
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import csv

from loguru import logger


def parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    从文件名中提取参数信息
    
    示例文件名: 
    - local_qwen_2_5_7b.bgee_v14_genex.ont10.map10.tri10.cand3.iter1.eval_report.json
    - mmm_beta_gpt_4o_mini.bgee_v14_genex.ont10.map10.tri5.cand3.eval_report.json
    
    Args:
        filename: 文件名（不含路径）
    
    Returns:
        包含解析参数的字典，如果解析失败返回 None
    """
    # 移除 .eval_report.json 后缀
    if not filename.endswith('.eval_report.json'):
        return None
    
    base = filename[:-len('.eval_report.json')]
    
    # 分割为各个部分
    parts = base.split('.')
    if len(parts) < 2:
        logger.warning(f"文件名格式不符合规范（至少需要模型名和数据集名）: {filename}")
        return None
    
    # 提取模型名和数据集名
    model_name = parts[0]
    dataset_name = parts[1]
    
    # 提取 ont_k, map_k, tri_k, cand_k, iter_rounds
    ont_k = None
    map_k = None
    tri_k = None
    cand_k = None
    iter_rounds = None
    
    for part in parts[2:]:
        ont_match = re.match(r'ont(\d+)', part)
        if ont_match:
            ont_k = int(ont_match.group(1))
            continue
        
        map_match = re.match(r'map(\d+)', part)
        if map_match:
            map_k = int(map_match.group(1))
            continue
        
        tri_match = re.match(r'tri(\d+)', part)
        if tri_match:
            tri_k = int(tri_match.group(1))
            continue
        
        cand_match = re.match(r'cand(\d+)', part)
        if cand_match:
            cand_k = int(cand_match.group(1))
            continue
        
        iter_match = re.match(r'iter(\d+)', part)
        if iter_match:
            iter_rounds = int(iter_match.group(1))
            continue
    
    # 对于 baseline 实验，参数可能缺失，用空值填充
    if ont_k is None or map_k is None or tri_k is None or cand_k is None:
        logger.info(f"文件名缺少参数（ont/map/tri/cand），视为 baseline 实验: {filename}")
    
    return {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'ont_k': ont_k if ont_k is not None else '',
        'map_k': map_k if map_k is not None else '',
        'tri_k': tri_k if tri_k is not None else '',
        'cand_k': cand_k if cand_k is not None else '',
        'iter_rounds': iter_rounds if iter_rounds is not None else ''
    }


def extract_metrics(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 eval_report JSON 数据中提取关键指标
    
    Args:
        report_data: 从 JSON 文件加载的数据
    
    Returns:
        包含指标的字典
    """
    summary = report_data.get('summary', {})
    
    # 提取关键指标
    overall_f1 = summary.get('avg_adj_macro_f1')
    conf_f1 = summary.get('avg_adj_macro_f1_has_final')
    empty_rate = summary.get('ratio_final_exec_empty')
    
    # 提取 with_references 和 without_references
    with_refs = summary.get('with_references', {})
    without_refs = summary.get('without_references', {})
    hybrid_f1 = with_refs.get('avg_adj_macro_f1')
    ont_only_f1 = without_refs.get('avg_adj_macro_f1')
    
    # 提取 token 使用情况
    token_usage = summary.get('token_usage', {})
    avg_token = token_usage.get('avg_total_tokens')
    
    # 提取完整模型名称
    model = summary.get('model', '')
    
    return {
        'overall_f1': round(overall_f1, 4) if overall_f1 is not None else 'N/A',
        'conf_f1': round(conf_f1, 4) if conf_f1 is not None else 'N/A',
        'empty_rate': f"{empty_rate * 100:.2f}" if empty_rate is not None else 'N/A',
        'hybrid_f1': round(hybrid_f1, 4) if hybrid_f1 is not None else 'N/A',
        'ont_only_f1': round(ont_only_f1, 4) if ont_only_f1 is not None else 'N/A',
        'avg_token': int(avg_token) if avg_token is not None else 'N/A',
        'full_model_name': model
    }


def process_file(file_path: Path, project_root: Path) -> Optional[Dict[str, Any]]:
    """
    处理单个 eval_report.json 文件
    
    Args:
        file_path: 文件路径
        project_root: 项目根目录路径
    
    Returns:
        包含所有信息的字典，如果处理失败返回 None
    """
    filename = file_path.name
    
    # 解析文件名
    params = parse_filename(filename)
    if params is None:
        return None
    
    # 读取 JSON 文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
    except Exception as e:
        logger.error(f"读取 JSON 文件失败: {file_path}, 错误: {e}")
        return None
    
    # 提取指标
    metrics = extract_metrics(report_data)
    
    # 计算相对路径
    try:
        rel_path = file_path.relative_to(project_root)
    except ValueError:
        rel_path = file_path
    
    # 合并所有信息
    result = {
        'file_path': str(rel_path),
        'model': params['model_name'],
        'dataset': params['dataset_name'],
        'ont_k': params['ont_k'],
        'map_k': params['map_k'],
        'tri_k': params['tri_k'],
        'cand_k': params['cand_k'],
        'iter': params['iter_rounds'],
        'overall_f1': metrics['overall_f1'],
        'conf_f1': metrics['conf_f1'],
        'empty_rate': metrics['empty_rate'],
        'hybrid_f1': metrics['hybrid_f1'],
        'ont_only_f1': metrics['ont_only_f1'],
        'avg_token': metrics['avg_token'],
        'full_model_name': metrics['full_model_name']
    }
    
    return result


def scan_directory(input_dir: Path, project_root: Path) -> List[Dict[str, Any]]:
    """
    扫描目录下的所有 eval_report.json 文件（不递归）
    
    Args:
        input_dir: 输入目录
        project_root: 项目根目录
    
    Returns:
        包含所有实验结果的列表
    """
    results = []
    
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return results
    
    # 只扫描直接子文件（不递归）
    for item in input_dir.iterdir():
        if item.is_file() and item.name.endswith('.eval_report.json'):
            logger.info(f"处理文件: {item.name}")
            result = process_file(item, project_root)
            if result is not None:
                results.append(result)
            else:
                logger.warning(f"跳过文件: {item.name}")
    
    return results


def write_csv(results: List[Dict[str, Any]], output_path: Path, sort_by: str) -> None:
    """
    将结果写入 CSV 文件
    
    Args:
        results: 实验结果列表
        output_path: 输出文件路径
        sort_by: 排序列名
    """
    if not results:
        logger.warning("没有可写入的结果")
        return
    
    # 排序（按指定列降序）
    # 将列名映射到字典键
    sort_key_map = {
        'Overall F1': 'overall_f1',
        'Conf. F1': 'conf_f1',
        'Empty Rate (%)': 'empty_rate',
        'Hybrid F1': 'hybrid_f1',
        'Ont. Only F1': 'ont_only_f1',
        'Avg Token': 'avg_token'
    }
    
    sort_key = sort_key_map.get(sort_by, 'overall_f1')
    
    # 排序时将 N/A 视为最小值
    def sort_value(x):
        val = x[sort_key]
        if val == 'N/A':
            return float('-inf')
        # 如果是字符串（如 empty_rate 的百分比），转换为浮点数
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return float('-inf')
        return val
    
    results_sorted = sorted(results, key=sort_value, reverse=True)
    
    # 写入 CSV
    fieldnames = [
        'file_path', 'model', 'dataset', 'ont_k', 'map_k', 'tri_k', 'cand_k', 'iter',
        'overall_f1', 'conf_f1', 'empty_rate', 'hybrid_f1', 'ont_only_f1', 'avg_token', 'full_model_name'
    ]
    
    # CSV 列头（用户友好的名称）
    header_map = {
        'file_path': 'File Path',
        'model': 'Model',
        'dataset': 'Dataset',
        'ont_k': 'Ont K',
        'map_k': 'Map K',
        'tri_k': 'Tri K',
        'cand_k': 'Cand K',
        'iter': 'Iter',
        'overall_f1': 'Overall F1',
        'conf_f1': 'Conf. F1',
        'empty_rate': 'Empty Rate (%)',
        'hybrid_f1': 'Hybrid F1',
        'ont_only_f1': 'Ont. Only F1',
        'avg_token': 'Avg Token',
        'full_model_name': 'Full Model Name'
    }
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 写入表头（使用友好名称）
        writer.writerow({k: header_map[k] for k in fieldnames})
        
        # 写入数据
        writer.writerows(results_sorted)
    
    logger.info(f"CSV 文件已保存至: {output_path}")
    logger.info(f"共处理 {len(results_sorted)} 个实验结果")


def main():
    parser = argparse.ArgumentParser(
        description='批量分析实验结果并生成汇总表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m src.experiment.summarize_experiments --input evaluations/easy_bgee_new_all --output summary.csv
  python -m src.experiment.summarize_experiments --input evaluations/bgee --output bgee_summary.csv --sort-by "Conf. F1"
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='实验文件夹路径（必需）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='experiment_summary.csv',
        help='输出 CSV 文件路径（默认: experiment_summary.csv）'
    )
    
    parser.add_argument(
        '--sort-by',
        type=str,
        default='Overall F1',
        choices=['Overall F1', 'Conf. F1', 'Empty Rate (%)', 'Hybrid F1', 'Ont. Only F1', 'Avg Token'],
        help='排序列（默认: Overall F1，降序）'
    )
    
    args = parser.parse_args()
    
    # 获取项目根目录（假设脚本在 src/experiment/ 下）
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    
    # 解析输入和输出路径
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"输入目录: {input_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"排序列: {args.sort_by}")
    
    # 扫描目录
    results = scan_directory(input_path, project_root)
    
    # 写入 CSV
    write_csv(results, output_path, args.sort_by)
    
    logger.info("处理完成！")


if __name__ == '__main__':
    main()

