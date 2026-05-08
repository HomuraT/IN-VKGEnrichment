#!/bin/bash
# 提取推理报告的便捷脚本

set -e

# 默认参数
PREDICTIONS_FILE=""
OUTPUT_DIR=""
NO_INDIVIDUAL=false
NO_SUMMARY=false

# 帮助信息
show_help() {
    cat << EOF
用法: $0 <predictions.jsonl> [选项]

提取 predictions.jsonl 文件中所有样例的推理报告

参数:
    predictions.jsonl    predictions.jsonl 文件路径

选项:
    -o, --output-dir DIR    输出目录（默认：analysis/reasoning_reports/<实验名称>）
    --no-individual         不保存单个报告文件（只保存汇总）
    --no-summary            不保存汇总文件（只保存单个报告）
    -h, --help              显示此帮助信息

示例:
    # 基本用法
    $0 runs/easy_bgee_new_1217_vkg_agent/yunwu_gpt_4o_mini.bgee_v14_genex.ont10.map10.tri10.cand3.iter3.predictions.jsonl
    
    # 指定输出目录
    $0 runs/xxx/predictions.jsonl -o analysis/my_reports
    
    # 只保存汇总文件
    $0 runs/xxx/predictions.jsonl --no-individual

输出文件:
    <output_dir>/
    ├── INDEX.md              # 索引文件（包含所有样例列表）
    ├── summary.json          # JSON 格式汇总
    ├── reports.jsonl         # JSONL 格式（每行一个报告）
    ├── statistics.txt        # 统计信息
    └── <sample_id>.md        # 单个样例的推理报告（如果启用）

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-individual)
            NO_INDIVIDUAL=true
            shift
            ;;
        --no-summary)
            NO_SUMMARY=true
            shift
            ;;
        -*)
            echo "错误: 未知选项 $1" >&2
            show_help
            exit 1
            ;;
        *)
            if [ -z "$PREDICTIONS_FILE" ]; then
                PREDICTIONS_FILE="$1"
            else
                echo "错误: 多余的参数 $1" >&2
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查必需参数
if [ -z "$PREDICTIONS_FILE" ]; then
    echo "错误: 缺少 predictions.jsonl 文件路径" >&2
    show_help
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "错误: 文件不存在: $PREDICTIONS_FILE" >&2
    exit 1
fi

# 构建命令
CMD="python -m analysis.extract_reasoning_reports \"$PREDICTIONS_FILE\""

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

if [ "$NO_INDIVIDUAL" = true ]; then
    CMD="$CMD --no-individual"
fi

if [ "$NO_SUMMARY" = true ]; then
    CMD="$CMD --no-summary"
fi

# 执行命令
echo "执行命令: $CMD"
echo ""
eval $CMD

