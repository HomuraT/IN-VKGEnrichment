#!/usr/bin/env bash
set -euo pipefail

# 参数实验脚本
# 对多个 LLM 模型和不同的检索/生成参数进行系统性实验

# Resolve repository root (one level up from this script directory)
ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR"
echo "工作目录: $ROOT_DIR"

# ===========================
# 固定配置
# ===========================
RUN_TAG=easy_bgee_new_1217_vkg_agent_3_simple_prompt
DATASET=resources/datasets/bgee_dataset_1217.jsonl
VKG_NAME=bgee_v14_genex
EMBEDDING_MODEL=local_qwen_3_8b_embedding
TEXTUALIZE_LLM_MODEL_KEY=local_qwen_2_5_7b
USE_SPARQL_DB=true
EXEC_SPARQL=true
WORKERS=2
EVAL_TIMEOUT=60
SKIP_BLOCKS="queries"

# ===========================
# 实验配置：待测模型
# ===========================
LLM_MODELS=(
  "yunwu_gpt_4o"
)

# ===========================
# 实验配置：精选参数组合（基于 easy_bgee_new_small_memory_fast_slow 结果）
# ===========================
# 每组配置格式：ont_k:map_k:tri_k:cand:iter
# 设计理念：
# 1. 高效配置（低token，快速）- iter1, 平衡检索
# 2. 高精度配置（高F1）- iter5, 精细检索
# 3. 高召回配置（低Empty Rate）- 丰富检索
# 4. 极简配置（最小token）- 最少检索
PARAMETER_CONFIGS=(
  # === 组1: GPT-4o-mini 最优配置 ===
  "10:10:10:3:5"   # F1=0.2081, Token=23655, 已验证最佳
  "10:10:10:3:1"   # F1=0.1306, Token=10356, 快速基线
  "30:30:30:3:1"   # F1=0.1305, Token=10151, 丰富检索
  
  # === 组2: Qwen-2.5-7B 高效配置 ===
  "30:30:30:3:1"   # F1=0.1828, Token=11317, 最佳本地模型
  "10:10:10:3:1"   # F1=0.1234, Token=11988, 轻量基线
  "20:20:20:3:1"   # F1=0.1334, Token=11519, 平衡配置
  
  # === 组3: 极简高效配置 ===
  "0:30:30:3:1"    # F1=0.1376, Token=9147, 无ontology测试
  "0:30:0:3:1"     # F1=0.1297, Token=6049, 仅mapping
  
  # === 组4: 迭代优化实验 ===
  "10:10:10:3:2"   # F1=0.1283, Token=18576, 2轮迭代
  "10:10:10:3:3"   # F1=0.1404, Token=25972, 3轮迭代
  "30:30:30:3:3"   # F1=0.1057, Token=24643, 3轮+丰富检索
  
  # === 组5: 候选数量实验 ===
  "10:10:10:5:1"   # F1=0.1277, Token=12276, 5候选基线
  "10:30:10:5:1"   # F1=0.1299, Token=12399, 5候选+增强mapping
  "30:30:30:5:1"   # F1=0.0934, Token=12110, 5候选+丰富检索
  
  # === 组6: 检索增强实验 ===
  "30:50:30:3:1"   # F1=0.1539, Token=11577, 高mapping
  "10:10:30:3:1"   # F1=0.1390, Token=11438, 高triples
  "10:10:60:3:1"   # F1=0.1001, Token=11900, 超高triples
  
  # === 组7: 单参数变体 - 基于 10:10:10:3:1 ===
  "11:10:10:3:1"   # 微调ontology +1
  "9:10:10:3:1"    # 微调ontology -1
  "10:11:10:3:1"   # 微调mapping +1
  "10:9:10:3:1"    # 微调mapping -1
  "10:10:11:3:1"   # 微调triples +1
  "10:10:9:3:1"    # 微调triples -1
  "10:10:10:4:1"   # 微调候选数 +1
  "10:10:10:2:1"   # 微调候选数 -1
  
  # === 组8: 单参数变体 - 基于 30:30:30:3:1 ===
  "31:30:30:3:1"   # 微调ontology +1
  "29:30:30:3:1"   # 微调ontology -1
  "30:31:30:3:1"   # 微调mapping +1
  "30:29:30:3:1"   # 微调mapping -1
  "30:30:31:3:1"   # 微调triples +1
  "30:30:29:3:1"   # 微调triples -1
  
  # === 组9: 单参数变体 - 基于 20:20:20:3:1 ===
  "21:20:20:3:1"   # 微调ontology +1
  "19:20:20:3:1"   # 微调ontology -1
  "20:21:20:3:1"   # 微调mapping +1
  "20:19:20:3:1"   # 微调mapping -1
  "20:20:21:3:1"   # 微调triples +1
  "20:20:19:3:1"   # 微调triples -1
  
  # === 组10: 迭代轮数细粒度变体 ===
  "10:10:10:3:4"   # 4轮迭代（填补3轮和5轮之间的空白）
  "30:30:30:3:2"   # 2轮迭代+丰富检索
  "20:20:20:3:2"   # 2轮迭代+平衡配置
  "20:20:20:3:3"   # 3轮迭代+平衡配置
  
  # === 组11: 高Triples策略（基于10:10:60最优发现）===
  "10:10:50:3:1"   # 高triples变体-1
  "10:10:40:3:1"   # 高triples变体-2
  "10:10:60:3:2"   # 高triples+迭代
  "10:10:60:3:3"   # 高triples+更多迭代
  "10:20:60:3:1"   # 高triples+增强mapping
  "10:30:60:3:1"   # 高triples+高mapping
  
  # === 组12: 高Mapping策略（基于10:30:10:5:1发现）===
  "10:30:10:3:1"   # 高mapping+标准候选
  "10:40:10:3:1"   # 更高mapping
  "10:40:10:5:1"   # 更高mapping+多候选
  "10:30:10:4:1"   # 高mapping+中等候选
  "10:30:20:5:1"   # 高mapping+中等triples+多候选
  "10:30:30:5:1"   # 高mapping+高triples+多候选
  
  # === 组13: 最优迭代探索（iter=3最优，测试周边）===
  "10:10:10:3:4"   # 已在组10，重要所以重复
  "10:10:20:3:3"   # 低ont+中tri+3轮
  "10:20:20:3:3"   # 中等增强+3轮
  "10:20:30:3:3"   # 中等ont/map+高tri+3轮
  "10:30:40:3:3"   # 中高map+高tri+3轮
  
  # === 组14: 黄金组合探索（综合最优模式）===
  "10:20:40:3:3"   # 低ont+中map+高tri+3轮
  "10:20:50:3:3"   # 低ont+中map+超高tri+3轮
  "10:30:50:3:3"   # 低ont+高map+超高tri+3轮
  "10:30:60:3:2"   # 高map+超高tri+2轮
  "10:30:60:3:3"   # 高map+超高tri+3轮
  "10:40:60:3:3"   # 超高map+超高tri+3轮
)

# ===========================
# 实验计数器
# ===========================
TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0

# 计算总实验数（模型数 × 配置数）
TOTAL_EXPERIMENTS=$((${#LLM_MODELS[@]} * ${#PARAMETER_CONFIGS[@]}))

echo "=========================================="
echo "参数实验启动（精选配置策略）"
echo "=========================================="
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "模型数量: ${#LLM_MODELS[@]}"
echo "配置数量: ${#PARAMETER_CONFIGS[@]}"
echo ""
echo "测试模型: ${LLM_MODELS[*]}"
echo ""
echo "配置列表 (ont:map:tri:cand:iter):"
for config in "${PARAMETER_CONFIGS[@]}"; do
  echo "  - $config"
done
echo "=========================================="
echo ""

# ===========================
# 通用实验执行函数
# ===========================
run_single_experiment() {
  local llm=$1
  local onto_k=$2
  local map_k=$3
  local trip_k=$4
  local cand=$5
  local iter=$6
  
  COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
  echo ""
  echo "[$COMPLETED_EXPERIMENTS/$TOTAL_EXPERIMENTS] 实验: ${llm}_onto${onto_k}_map${map_k}_trip${trip_k}_cand${cand}_iter${iter}"
  echo "  模型: $llm"
  echo "  ONTOLOGY_K=$onto_k, MAPPINGS_K=$map_k, TRIPLES_K=$trip_k"
  echo "  NUM_CANDIDATES=$cand, ITER_ROUNDS=$iter"
  
  # 设置输出路径（所有实验共享同一个 RUN_TAG）
  OUT="runs/${RUN_TAG}/predictions.jsonl"
  OUT_DECISION_ONLY="runs/${RUN_TAG}/decisions.jsonl"
  EVAL_OUT_DIR="evaluations/${RUN_TAG}"
  
  # 构建评估文件前缀（与 02_run_experiment_and_evaluate.sh 保持一致）
  safe_key() { echo "$1" | sed 's/[\/ ]/_/g'; }
  LLM_SAFE=$(safe_key "$llm")
  VKG_SAFE=$(safe_key "$VKG_NAME")
  EVAL_PREFIX="${LLM_SAFE}.${VKG_SAFE}.ont${onto_k}.map${map_k}.tri${trip_k}.cand${cand}.iter${iter}"
  
  # 检查评估文件是否已存在（检查 eval_report.json 或 eval_details.csv）
  EVAL_REPORT="${EVAL_OUT_DIR}/${EVAL_PREFIX}.eval_report.json"
  EVAL_DETAILS="${EVAL_OUT_DIR}/${EVAL_PREFIX}.eval_details.csv"
  
  if [ -f "$EVAL_REPORT" ] || [ -f "$EVAL_DETAILS" ]; then
    echo "  ⏭️  跳过（评估文件已存在）"
    echo "     文件: ${EVAL_PREFIX}.eval_report.json"
    SKIPPED_EXPERIMENTS=$((SKIPPED_EXPERIMENTS + 1))
    return 0
  fi
  
  # 执行实验
  START_TIME=$(date +%s)
  if RUN_TAG=$RUN_TAG \
     DATASET=$DATASET \
     VKG_NAME=$VKG_NAME \
     EMBEDDING_MODEL=$EMBEDDING_MODEL \
     LLM_MODEL_KEY=$llm \
     TEXTUALIZE_LLM_MODEL_KEY=$TEXTUALIZE_LLM_MODEL_KEY \
     ONTOLOGY_K=$onto_k \
     MAPPINGS_K=$map_k \
     TRIPLES_K=$trip_k \
     NUM_CANDIDATES=$cand \
     ITER_ROUNDS=$iter \
     USE_SPARQL_DB=$USE_SPARQL_DB \
     EXEC_SPARQL=$EXEC_SPARQL \
     WORKERS=$WORKERS \
     EVAL_TIMEOUT=$EVAL_TIMEOUT \
     OUT=$OUT \
     OUT_DECISION_ONLY=$OUT_DECISION_ONLY \
     SKIP_BLOCKS=$SKIP_BLOCKS \
     EVAL_OUT_DIR=$EVAL_OUT_DIR \
     bash script/02_run_experiment_and_evaluate.sh; then
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "  ✅ 完成 (耗时: ${DURATION}s)"
  else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
    echo "  ❌ 失败 (耗时: ${DURATION}s)"
    echo "  继续下一个实验..."
  fi
}

# ===========================
# 实验循环（精选配置）
# ===========================
for llm in "${LLM_MODELS[@]}"; do
  echo ""
  echo "=========================================="
  echo "🔧 当前模型: $llm"
  echo "=========================================="
  
  # 遍历所有配置
  for config in "${PARAMETER_CONFIGS[@]}"; do
    # 解析配置字符串 "ont:map:tri:cand:iter"
    IFS=':' read -r onto_k map_k trip_k cand iter <<< "$config"
    
    # 执行实验
    run_single_experiment "$llm" "$onto_k" "$map_k" "$trip_k" "$cand" "$iter"
  done
  
  echo ""
  echo "=========================================="
  echo "✅ 模型 $llm 所有实验完成"
  echo "=========================================="
  echo ""
done

# ===========================
# 实验总结
# ===========================
ACTUALLY_RUN=$((TOTAL_EXPERIMENTS - SKIPPED_EXPERIMENTS))
SUCCESS=$((ACTUALLY_RUN - FAILED_EXPERIMENTS))

echo ""
echo "=========================================="
echo "参数实验完成"
echo "=========================================="
echo "总实验数: $TOTAL_EXPERIMENTS"
echo "跳过: $SKIPPED_EXPERIMENTS（评估文件已存在）"
echo "实际运行: $ACTUALLY_RUN"
echo "成功: $SUCCESS"
echo "失败: $FAILED_EXPERIMENTS"
echo "=========================================="

echo ""
echo "🎉 所有实验流程完成！"
echo ""
echo "所有结果已保存在同一目录下："
echo "  运行结果: runs/${RUN_TAG}/"
echo "  评估结果: evaluations/${RUN_TAG}/"

# ===========================
# 生成汇总报告
# ===========================
echo ""
echo "生成实验汇总报告..."

mkdir -p ./summary

uv run python -m src.experiment.summarize_experiments \
  --input "evaluations/${RUN_TAG}" \
  --output "./summary/${RUN_TAG}.csv"

echo "✅ 汇总报告已生成: ./summary/${RUN_TAG}.csv"

