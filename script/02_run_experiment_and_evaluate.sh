#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (one level up from this script directory)
ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR"
echo "move to root dir: $ROOT_DIR"

# Ensure Python can import the top-level `src` package when running modules
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ===========================
# Configurable parameters
# Override via environment or inline: VAR=value ./02_run_experiment_and_evaluate.sh
# ===========================

# Runner tool: uv

# --------- Experiment (runner) ---------
DATASET=${DATASET:-resources/datasets/easybgee_v14_2.jsonl}
VKG_NAME=${VKG_NAME:-bgee_v14_genex}
# Optional: defaults to VKG_NAME when empty
ONTOLOGY_NAME=${ONTOLOGY_NAME:-}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-local_qwen_3_8b_embedding}
LLM_MODEL_KEY=${LLM_MODEL_KEY:-mmm_beta_gpt_4o_mini}
# Textualization LLM key (used only for vector DB naming); keep default if unsure
TEXTUALIZE_LLM_MODEL_KEY=${TEXTUALIZE_LLM_MODEL_KEY:-local_qwen_2_5_7b}
# Unified tag for run/eval output directories (e.g., bgee1, expA)
RUN_TAG=${RUN_TAG:-bgee1}
START=${START:-0}
LIMIT=${LIMIT:-}
USE_SPARQL_DB=${USE_SPARQL_DB:-true}
EXEC_SPARQL=${EXEC_SPARQL:-true}
OUT=${OUT:-runs/${RUN_TAG}/predictions.jsonl}
OUT_DECISION_ONLY=${OUT_DECISION_ONLY:-runs/${RUN_TAG}/decisions.jsonl}
WORKERS=${WORKERS:-4}
# 检索参数
ONTOLOGY_K=${ONTOLOGY_K:-10}
MAPPINGS_K=${MAPPINGS_K:-10}
TRIPLES_K=${TRIPLES_K:-0}
NUM_CANDIDATES=${NUM_CANDIDATES:-3}
ITER_ROUNDS=${ITER_ROUNDS:-3}
LOG_LEVEL=${LOG_LEVEL:-DEBUG}
SKIP_BLOCKS=${SKIP_BLOCKS:-}
# 本体文件路径（可选，支持 .ttl/.owl/.rdf/.xml 等格式）
ONTOLOGY_FILE=${ONTOLOGY_FILE:-}

# --------- Evaluation ---------
EVAL_DECISIONS=${EVAL_DECISIONS:-$OUT_DECISION_ONLY}
EVAL_OUT_DIR=${EVAL_OUT_DIR:-evaluations/${RUN_TAG}}
EVAL_VKG_NAME=${EVAL_VKG_NAME:-$VKG_NAME}
EVAL_DATASET=${EVAL_DATASET:-$DATASET}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-60}
# 高级（评估器并发与超时细化）
EVAL_HARD_TIMEOUT_SECONDS=${EVAL_HARD_TIMEOUT_SECONDS:-60}
EVAL_HARD_TIMEOUT_WORKERS=${EVAL_HARD_TIMEOUT_WORKERS:-16}
EVAL_INNER_TIMEOUT_SECONDS=${EVAL_INNER_TIMEOUT_SECONDS:-}
SAVE_JSON=${SAVE_JSON:-}
SAVE_CSV=${SAVE_CSV:-}

# Ensure output directories exist
if [ -n "${OUT:-}" ]; then mkdir -p "$(dirname "$OUT")"; fi
if [ -n "${OUT_DECISION_ONLY:-}" ]; then mkdir -p "$(dirname "$OUT_DECISION_ONLY")"; fi
mkdir -p "$EVAL_OUT_DIR"

# Export VKG name for endpoint resolution inside evaluation (see evaluate_sparql_results.py)
export EVAL_VKG_NAME="${EVAL_VKG_NAME}"

# Always resolve decisions to the actual prefixed filename produced by runner:
# <llm>.<vkg>.ont<O>.map<M>.tri<T>.cand<C>.iter<R>.<basename>
safe_key() { echo "$1" | sed 's/[\/ ]/_/g'; }
LLM_SAFE=$(safe_key "$LLM_MODEL_KEY")
VKG_SAFE=$(safe_key "$VKG_NAME")
DEC_PREFIX="${LLM_SAFE}.${VKG_SAFE}.ont${ONTOLOGY_K}.map${MAPPINGS_K}.tri${TRIPLES_K}.cand${NUM_CANDIDATES}.iter${ITER_ROUNDS}"
DEC_BASE_DIR=$(dirname "$OUT_DECISION_ONLY")
DEC_BASE_NAME=$(basename "$OUT_DECISION_ONLY")
EVAL_DECISIONS="${DEC_BASE_DIR}/${DEC_PREFIX}.${DEC_BASE_NAME}"

echo "[1/2] Run experiment"
RUN_ARGS=(run -m src.experiment.run_experiment
  --dataset "$DATASET"
  --vkg-name "$VKG_NAME"
  --embedding-model "$EMBEDDING_MODEL"
  --llm-model-key "$LLM_MODEL_KEY"
  --ontology-k "$ONTOLOGY_K"
  --mappings-k "$MAPPINGS_K"
  --triples-k "$TRIPLES_K"
  --num-candidates "$NUM_CANDIDATES"
  --workers "$WORKERS"
  --iter-rounds "$ITER_ROUNDS"
  --out "$OUT"
  --out-decision-only "$OUT_DECISION_ONLY"
  --log-level "$LOG_LEVEL"
)

# Optional arguments
if [ -n "${ONTOLOGY_NAME}" ]; then RUN_ARGS+=(--ontology-name "$ONTOLOGY_NAME"); fi
if [ -n "${TEXTUALIZE_LLM_MODEL_KEY}" ]; then RUN_ARGS+=(--textualize-llm-model-key "$TEXTUALIZE_LLM_MODEL_KEY"); fi
if [ -n "${LIMIT}" ]; then RUN_ARGS+=(--limit "$LIMIT"); fi
if [ "${START}" != "" ] && [ "$START" != "0" ]; then RUN_ARGS+=(--start "$START"); fi
if [ "${USE_SPARQL_DB}" = "true" ]; then RUN_ARGS+=(--use-sparql-db); fi
if [ "${EXEC_SPARQL}" = "true" ]; then RUN_ARGS+=(--exec-sparql); fi
if [ -n "${SKIP_BLOCKS}" ]; then RUN_ARGS+=(--skip-blocks "$SKIP_BLOCKS"); fi
if [ -n "${ONTOLOGY_FILE}" ]; then RUN_ARGS+=(--ontology-file "$ONTOLOGY_FILE"); fi

uv "${RUN_ARGS[@]}"

echo "[2/2] Evaluate results"
echo "DEBUG: EVAL_DATASET=$EVAL_DATASET"
echo "DEBUG: EVAL_DECISIONS=$EVAL_DECISIONS"

EVAL_ARGS=(run -m src.experiment.evaluate_sparql_results
  --decisions "$EVAL_DECISIONS"
  --out-dir "$EVAL_OUT_DIR"
  --dataset "$EVAL_DATASET"
  --timeout "$EVAL_TIMEOUT"
)

if [ -n "${SAVE_JSON}" ]; then EVAL_ARGS+=(--save-json "$SAVE_JSON"); fi
if [ -n "${SAVE_CSV}" ]; then EVAL_ARGS+=(--save-csv "$SAVE_CSV"); fi
# 透传高级超时/并发
if [ -n "${EVAL_HARD_TIMEOUT_SECONDS}" ]; then EVAL_ARGS+=(--hard-timeout-seconds "$EVAL_HARD_TIMEOUT_SECONDS"); fi
if [ -n "${EVAL_HARD_TIMEOUT_WORKERS}" ]; then EVAL_ARGS+=(--hard-timeout-workers "$EVAL_HARD_TIMEOUT_WORKERS"); fi
if [ -n "${EVAL_INNER_TIMEOUT_SECONDS}" ]; then EVAL_ARGS+=(--inner-timeout-seconds "$EVAL_INNER_TIMEOUT_SECONDS"); fi

uv "${EVAL_ARGS[@]}"

echo "✅ Experiment run and evaluation completed."


