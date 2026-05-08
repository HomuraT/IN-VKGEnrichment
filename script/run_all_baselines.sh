#!/bin/bash

# 一键运行所有基线实验并收集结果
# 用法: bash script/run_all_baselines.sh <MODEL_KEY> [DATASET_TYPE]
# 示例: bash script/run_all_baselines.sh yunwu_gpt_4o_mini bgee
# 示例: bash script/run_all_baselines.sh yunwu_gpt_4o_mini npd
# 示例: bash script/run_all_baselines.sh yunwu_gpt_4o_mini all

set -e

# 检查参数
if [ -z "$1" ]; then
    echo "错误: 请提供模型 KEY"
    echo "用法: bash script/run_all_baselines.sh <MODEL_KEY> [DATASET_TYPE]"
    echo "DATASET_TYPE: bgee | npd | all (默认: all)"
    echo "可用模型: yunwu_gpt_4o_mini, yunwu_gpt_4o, yunwu_gpt_5_1, local_qwen_2_5_7b"
    exit 1
fi

MODEL_KEY=$1
DATASET_TYPE=${2:-all}

echo "=========================================="
echo "开始运行所有基线实验"
echo "模型: $MODEL_KEY"
echo "数据集类型: $DATASET_TYPE"
echo "=========================================="

# ==================== Bgee 数据集 ====================
if [ "$DATASET_TYPE" = "bgee" ] || [ "$DATASET_TYPE" = "all" ]; then
    echo ""
    echo "==================== Bgee 数据集 ===================="
    
    # 1. baseline-vanilla LLM (simple_onto2sparql)
    echo ""
    echo "[1/5] 运行 baseline-vanilla LLM (simple_onto2sparql)..."
    EVAL_FILE="evaluations/baselines/simple_onto2sparql/easy_bgee_new_1217_vkg_agent/${MODEL_KEY}.bgee_v14_genex.eval_report.json"
    if [ -f "$EVAL_FILE" ]; then
        echo "⏭️  跳过: 评估文件已存在 ($EVAL_FILE)"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=bgee_v14_genex \
        RUN_TAG=easy_bgee_new_1217_vkg_agent \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/bgee_dataset_1217.jsonl \
        ONTOLOGY=resources/vkg_ontologies/bgee_v14_genex.ttl \
        EVAL_K=1 \
        EVAL_TIMEOUT=45 \
        KEEP_DECISIONS_PATH=true \
        SAVE_JSON= \
        SAVE_CSV= \
        bash baselines/simple_onto2sparql/run_experiment.sh
    fi
    
    # 2. baseline-CoT (cot_onto2sparql)
    echo ""
    echo "[2/5] 运行 baseline-CoT (cot_onto2sparql)..."
    EVAL_FILE="evaluations/baselines/cot_onto2sparql/easy_bgee_new_1217_vkg_agent/${MODEL_KEY}.bgee_v14_genex.eval_report.json"
    if [ -f "$EVAL_FILE" ]; then
        echo "⏭️  跳过: 评估文件已存在 ($EVAL_FILE)"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=bgee_v14_genex \
        RUN_TAG=easy_bgee_new_1217_vkg_agent \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/bgee_dataset_1217.jsonl \
        ONTOLOGY=resources/vkg_ontologies/bgee_v14_genex.ttl \
        EVAL_K=1 \
        EVAL_TIMEOUT=45 \
        bash baselines/cot_onto2sparql/run_experiment.sh
    fi
    
    # 3. baseline-RAG Ontology (rag_onto2sparql)
    echo ""
    echo "[3/5] 运行 baseline-RAG Ontology (rag_onto2sparql)..."
    EVAL_PATTERN="evaluations/baselines/rag_onto2sparql/easy_bgee_new_1217_vkg_agent/${MODEL_KEY}.bgee_v14_genex.*.eval_report.json"
    if ls $EVAL_PATTERN 1> /dev/null 2>&1; then
        echo "⏭️  跳过: 评估文件已存在 ($(ls $EVAL_PATTERN | head -1))"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=bgee_v14_genex \
        ONTOLOGY_FILE=resources/vkg_ontologies/bgee_v14_genex.ttl \
        EMBEDDING_MODEL=local_qwen_3_8b_embedding \
        RUN_TAG=easy_bgee_new_1217_vkg_agent \
        RETRIEVAL_K=100 \
        CHUNK_SIZE=1000 \
        CHUNK_OVERLAP=200 \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/bgee_dataset_1217.jsonl \
        EVAL_TIMEOUT=45 \
        bash baselines/rag_onto2sparql/run_experiment.sh
    fi
    
    # 4. baseline-RAG Mapping (mapping_rag_onto2sparql)
    echo ""
    echo "[4/5] 运行 baseline-RAG Mapping (mapping_rag_onto2sparql)..."
    EVAL_PATTERN="evaluations/baselines/mapping_rag_onto2sparql/easy_bgee_new_1217_vkg_agent/${MODEL_KEY}.bgee_v14_genex.*.eval_report.json"
    if ls $EVAL_PATTERN 1> /dev/null 2>&1; then
        echo "⏭️  跳过: 评估文件已存在 ($(ls $EVAL_PATTERN | head -1))"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=bgee_v14_genex \
        MAPPING_FILE=resources/vkg_mappings/bgee_v14_genex.obda \
        EMBEDDING_MODEL=local_qwen_3_8b_embedding \
        RUN_TAG=easy_bgee_new_1217_vkg_agent \
        RETRIEVAL_K=10 \
        CHUNK_SIZE=1000 \
        CHUNK_OVERLAP=200 \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/bgee_dataset_1217.jsonl \
        EVAL_TIMEOUT=45 \
        bash baselines/mapping_rag_onto2sparql/run_experiment.sh
    fi
    
    # 5. baseline-Hybrid RAG (hybrid_rag_onto2sparql)
    echo ""
    echo "[5/5] 运行 baseline-Hybrid RAG (hybrid_rag_onto2sparql)..."
    EVAL_PATTERN="evaluations/baselines/hybrid_rag_onto2sparql/easy_bgee_new_1217_vkg_agent/${MODEL_KEY}.bgee_v14_genex.*.eval_report.json"
    if ls $EVAL_PATTERN 1> /dev/null 2>&1; then
        echo "⏭️  跳过: 评估文件已存在 ($(ls $EVAL_PATTERN | head -1))"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=bgee_v14_genex \
        ONTOLOGY_FILE=resources/vkg_ontologies/bgee_v14_genex.ttl \
        MAPPING_FILE=resources/vkg_mappings/bgee_v14_genex.obda \
        EMBEDDING_MODEL=local_qwen_3_8b_embedding \
        RUN_TAG=easy_bgee_new_1217_vkg_agent \
        ONTOLOGY_K=15 \
        MAPPING_K=10 \
        CHUNK_SIZE=1000 \
        CHUNK_OVERLAP=200 \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/bgee_dataset_1217.jsonl \
        EVAL_TIMEOUT=45 \
        bash baselines/hybrid_rag_onto2sparql/run_experiment.sh
    fi
    
    # 收集 Bgee 结果
    echo ""
    echo "==================== 收集 Bgee 结果 ===================="
    
    echo "收集 simple_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/simple_onto2sparql/easy_bgee_new_1217_vkg_agent \
        --output ./summary/easy_bgee_new_1217_simple_baseline.csv
    
    echo "收集 cot_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/cot_onto2sparql/easy_bgee_new_1217_vkg_agent \
        --output ./summary/easy_bgee_new_1217_cot_baseline.csv
    
    echo "收集 rag_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/rag_onto2sparql/easy_bgee_new_1217_vkg_agent \
        --output ./summary/easy_bgee_new_1217_rag_onto_baseline.csv
    
    echo "收集 mapping_rag_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/mapping_rag_onto2sparql/easy_bgee_new_1217_vkg_agent \
        --output ./summary/easy_bgee_new_1217_rag_mapping_baseline.csv
    
    echo "收集 hybrid_rag_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/hybrid_rag_onto2sparql/easy_bgee_new_1217_vkg_agent \
        --output ./summary/easy_bgee_new_1217_hybrid_rag_baseline.csv
fi

# ==================== NPD 数据集 ====================
if [ "$DATASET_TYPE" = "npd" ] || [ "$DATASET_TYPE" = "all" ]; then
    echo ""
    echo "==================== NPD 数据集 ===================="
    
    # 注意: NPD 数据集不运行 vanilla LLM 和 CoT (会超长)
    
    # 1. baseline-RAG Ontology (rag_onto2sparql)
    echo ""
    echo "[1/3] 运行 baseline-RAG Ontology (rag_onto2sparql)..."
    EVAL_PATTERN="evaluations/baselines/rag_onto2sparql/npd_dataset_1216/${MODEL_KEY}.npd.*.eval_report.json"
    if ls $EVAL_PATTERN 1> /dev/null 2>&1; then
        echo "⏭️  跳过: 评估文件已存在 ($(ls $EVAL_PATTERN | head -1))"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=npd \
        ONTOLOGY_FILE=resources/vkg_ontologies/npd.owl \
        EMBEDDING_MODEL=local_qwen_3_8b_embedding \
        RUN_TAG=npd_dataset_1216 \
        RETRIEVAL_K=30 \
        CHUNK_SIZE=1000 \
        CHUNK_OVERLAP=200 \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/npd_dataset_1216.jsonl \
        EVAL_TIMEOUT=45 \
        bash baselines/rag_onto2sparql/run_experiment.sh
    fi
    
    # 2. baseline-RAG Mapping (mapping_rag_onto2sparql)
    echo ""
    echo "[2/3] 运行 baseline-RAG Mapping (mapping_rag_onto2sparql)..."
    EVAL_PATTERN="evaluations/baselines/mapping_rag_onto2sparql/npd_dataset_1216/${MODEL_KEY}.npd.*.eval_report.json"
    if ls $EVAL_PATTERN 1> /dev/null 2>&1; then
        echo "⏭️  跳过: 评估文件已存在 ($(ls $EVAL_PATTERN | head -1))"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=npd \
        MAPPING_FILE=resources/vkg_mappings/npd.obda \
        EMBEDDING_MODEL=local_qwen_3_8b_embedding \
        RUN_TAG=npd_dataset_1216 \
        RETRIEVAL_K=30 \
        CHUNK_SIZE=1000 \
        CHUNK_OVERLAP=200 \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/npd_dataset_1216.jsonl \
        EVAL_TIMEOUT=45 \
        bash baselines/mapping_rag_onto2sparql/run_experiment.sh
    fi
    
    # 3. baseline-Hybrid RAG (hybrid_rag_onto2sparql)
    echo ""
    echo "[3/3] 运行 baseline-Hybrid RAG (hybrid_rag_onto2sparql)..."
    EVAL_PATTERN="evaluations/baselines/hybrid_rag_onto2sparql/npd_dataset_1216/${MODEL_KEY}.npd.*.eval_report.json"
    if ls $EVAL_PATTERN 1> /dev/null 2>&1; then
        echo "⏭️  跳过: 评估文件已存在 ($(ls $EVAL_PATTERN | head -1))"
    else
        MODEL_KEY=$MODEL_KEY \
        VKG_NAME=npd \
        ONTOLOGY_FILE=resources/vkg_ontologies/npd.owl \
        MAPPING_FILE=resources/vkg_mappings/npd.obda \
        EMBEDDING_MODEL=local_qwen_3_8b_embedding \
        RUN_TAG=npd_dataset_1216 \
        ONTOLOGY_K=10 \
        MAPPING_K=10 \
        CHUNK_SIZE=1000 \
        CHUNK_OVERLAP=200 \
        WORKERS=16 \
        LOG_LEVEL=INFO \
        FORCE_RESTART=false \
        DATASET=resources/datasets/npd_dataset_1216.jsonl \
        EVAL_TIMEOUT=45 \
        bash baselines/hybrid_rag_onto2sparql/run_experiment.sh
    fi
    
    # 收集 NPD 结果
    echo ""
    echo "==================== 收集 NPD 结果 ===================="
    
    echo "收集 rag_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/rag_onto2sparql/npd_dataset_1216 \
        --output ./summary/npd_dataset_1216_rag_onto_baseline.csv
    
    echo "收集 mapping_rag_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/mapping_rag_onto2sparql/npd_dataset_1216 \
        --output ./summary/npd_dataset_1216_rag_mapping_baseline.csv
    
    echo "收集 hybrid_rag_onto2sparql 结果..."
    uv run python -m src.experiment.summarize_experiments \
        --input evaluations/baselines/hybrid_rag_onto2sparql/npd_dataset_1216 \
        --output ./summary/npd_dataset_1216_hybrid_rag_baseline.csv
fi

# ==================== 完成 ====================
echo ""
echo "=========================================="
echo "所有基线实验完成！"
echo "模型: $MODEL_KEY"
echo ""
echo "结果汇总文件已保存到 ./summary/ 目录"
echo ""
if [ "$DATASET_TYPE" = "bgee" ] || [ "$DATASET_TYPE" = "all" ]; then
    echo "Bgee 数据集结果:"
    echo "  - easy_bgee_new_1217_simple_baseline.csv"
    echo "  - easy_bgee_new_1217_cot_baseline.csv"
    echo "  - easy_bgee_new_1217_rag_onto_baseline.csv"
    echo "  - easy_bgee_new_1217_rag_mapping_baseline.csv"
    echo "  - easy_bgee_new_1217_hybrid_rag_baseline.csv"
fi
if [ "$DATASET_TYPE" = "npd" ] || [ "$DATASET_TYPE" = "all" ]; then
    echo ""
    echo "NPD 数据集结果:"
    echo "  - npd_dataset_1216_rag_onto_baseline.csv"
    echo "  - npd_dataset_1216_rag_mapping_baseline.csv"
    echo "  - npd_dataset_1216_hybrid_rag_baseline.csv"
fi
echo "=========================================="

