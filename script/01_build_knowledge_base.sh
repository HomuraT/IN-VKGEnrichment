#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (one level up from this script directory)
ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR"
echo "move to root dir: $ROOT_DIR"

# Ensure Python can import the top-level `src` package when running files by path
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ===========================
# Configurable parameters
# Override via environment or inline: VAR=value ./01_build_knowledge_base.sh
# ===========================

# Example configuration (all parameters) — copy, adjust, and uncomment if needed
# 也可以在运行时以内联方式传参：VAR=value VAR2=value script/01_build_knowledge_base.sh
#
# MODE=full
# ONTOLOGY_TTL="resources/vkg_ontologies/bgee_v14_genex.ttl"
# ONTOLOGY_NAME="bgee_v14_genex"
# OBDA_FILE="resources/vkg_mappings/bgee_v14_genex.obda"
# PARSED_ONTO_DIR="resources/parsed_ontologies"
# TEXT_ONTO_DIR="resources/textualized_ontology_elements"
# TEXT_VKG_DIR="resources/textualized_vkg_mappings"
# VECTOR_DB_DIR="resources/vector_databases"
# T2S_DIR="resources/text_to_sparql_examples"
# EMBEDDING_MODEL="mmm_beta_text_embedding_3_small"
# CHUNK_SIZE=500
# CHUNK_OVERLAP=100
# ONTO_VECTOR_DB_NAME="textualized_ontology_elements.chroma"
# VKG_VECTOR_DB_NAME="textualized_vkg_mappings.chroma"
# T2S_VECTOR_DB_NAME="text_to_sparql_vector_db.chroma"
# TEXTUALIZE_MODE="full"   # 可选，覆盖由 MODE 推导的值
# VECTOR_MODE="full"       # 可选，覆盖由 MODE 推导的值
# TEXTUALIZE_LLM_MODEL="mmm_beta_gpt_4o_mini"  # 可选，文本化使用的 LLM 模型键名

# One-line example (copy-paste):
# MODE=full ONTOLOGY_TTL="resources/vkg_ontologies/bgee_v14_genex.ttl" ONTOLOGY_NAME="bgee_v14_genex" OBDA_FILE="resources/vkg_mappings/bgee_v14_genex.obda" PARSED_ONTO_DIR="resources/parsed_ontologies" TEXT_ONTO_DIR="resources/textualized_ontology_elements" TEXT_VKG_DIR="resources/textualized_vkg_mappings" VECTOR_DB_DIR="resources/vector_databases" T2S_DIR="resources/text_to_sparql_examples" EMBEDDING_MODEL="mmm_beta_text_embedding_3_small" CHUNK_SIZE=500 CHUNK_OVERLAP=100 ONTO_VECTOR_DB_NAME="textualized_ontology_elements.chroma" VKG_VECTOR_DB_NAME="textualized_vkg_mappings.chroma" T2S_VECTOR_DB_NAME="text_to_sparql_vector_db.chroma" TEXTUALIZE_MODE="full" VECTOR_MODE="full" TEXTUALIZE_LLM_MODEL="mmm_beta_gpt_4o_mini" script/01_build_knowledge_base.sh

# Workflow mode
MODE=${MODE:-test}  # accepted: test | full

# Python interpreter
PYTHON=${PYTHON:-"${ROOT_DIR}/.venv/bin/python"}

# Inputs
ONTOLOGY_TTL=${ONTOLOGY_TTL:-resources/vkg_ontologies/bgee_v14_genex.ttl}
ONTOLOGY_NAME=${ONTOLOGY_NAME:-bgee_v14_genex}
OBDA_FILE=${OBDA_FILE:-resources/vkg_mappings/bgee_v14_genex.obda}

# Output/directories
PARSED_ONTO_DIR=${PARSED_ONTO_DIR:-resources/parsed_ontologies}
TEXT_ONTO_DIR=${TEXT_ONTO_DIR:-resources/textualized_ontology_elements}
TEXT_VKG_DIR=${TEXT_VKG_DIR:-resources/textualized_vkg_mappings}
VECTOR_DB_DIR=${VECTOR_DB_DIR:-resources/vector_databases}
T2S_DIR=${T2S_DIR:-resources/text_to_sparql_examples}

# Aggregated triples (optional)
AGG_NT_FILE=${AGG_NT_FILE:-resources/vkg_triples/${VKG_NAME}-materialized.nt}
AGG_JSONL_DIR=${AGG_JSONL_DIR:-resources/vkg_triples_aggregated}
TEXT_AGG_DIR=${TEXT_AGG_DIR:-resources/textualized_aggregated_triples}
# textualize aggregated triples concurrency
AGG_MAX_WORKERS=${AGG_MAX_WORKERS:-16}

# Vector DB configuration
EMBEDDING_MODEL=${EMBEDDING_MODEL:-mmm_beta_text_embedding_3_small}
CHUNK_SIZE=${CHUNK_SIZE:-500}
CHUNK_OVERLAP=${CHUNK_OVERLAP:-100}
# 向量库批大小与断点续存（可通过环境变量覆盖）
ONTO_BATCH_SIZE=${ONTO_BATCH_SIZE:-256}
VKG_BATCH_SIZE=${VKG_BATCH_SIZE:-256}
T2S_BATCH_SIZE=${T2S_BATCH_SIZE:-256}
ONTO_RESUME=${ONTO_RESUME:-true}
VKG_RESUME=${VKG_RESUME:-true}
T2S_RESUME=${T2S_RESUME:-true}
# Aggregated triples vector build
AGG_BATCH_SIZE=${AGG_BATCH_SIZE:-256}
AGG_RESUME=${AGG_RESUME:-true}
# 文本化阶段的断点续存（默认启用，可通过环境变量关闭）
TEXT_ONTO_RESUME=${TEXT_ONTO_RESUME:-true}
TEXT_VKG_RESUME=${TEXT_VKG_RESUME:-true}
# 根据 OBDA 文件推导 VKG 名称（可通过 VKG_NAME 覆盖）
VKG_NAME=${VKG_NAME:-$(basename "$OBDA_FILE" .obda)}

# 文本化所用的 LLM 模型（传给 --llm_model）
TEXTUALIZE_LLM_MODEL=${TEXTUALIZE_LLM_MODEL:-mmm_beta_gpt_4o_mini}
# 将文本化 LLM 与嵌入模型加入向量库名称，避免不同配置写入同一目录
# 对键名做简单安全化（替换 / 和 空格 为 _）
DB_SUFFIX_SAFE=${EMBEDDING_MODEL//\//_}
DB_SUFFIX_SAFE=${DB_SUFFIX_SAFE// /_}
LLM_SAFE=${TEXTUALIZE_LLM_MODEL//\//_}
LLM_SAFE=${LLM_SAFE// /_}

# 命名规则：
# - Ontology:   <ONTOLOGY_NAME>.<LLM_SAFE>.<DB_SUFFIX_SAFE>.textualized_ontology_elements.chroma
# - VKG map:    <VKG_NAME>.<LLM_SAFE>.<DB_SUFFIX_SAFE>.textualized_vkg_mappings.chroma
# - T2S:        <DB_SUFFIX_SAFE>.text_to_sparql_vector_db.chroma（无文本化阶段）
# - Aggregated: <VKG_NAME>.<LLM_SAFE>.<DB_SUFFIX_SAFE>.textualized_aggregated_triples.chroma
ONTO_VECTOR_DB_NAME=${ONTO_VECTOR_DB_NAME:-${ONTOLOGY_NAME}.${LLM_SAFE}.${DB_SUFFIX_SAFE}.textualized_ontology_elements.chroma}
VKG_VECTOR_DB_NAME=${VKG_VECTOR_DB_NAME:-${VKG_NAME}.${LLM_SAFE}.${DB_SUFFIX_SAFE}.textualized_vkg_mappings.chroma}
T2S_VECTOR_DB_NAME=${T2S_VECTOR_DB_NAME:-${DB_SUFFIX_SAFE}.text_to_sparql_vector_db.chroma}
AGG_VECTOR_DB_NAME=${AGG_VECTOR_DB_NAME:-${VKG_NAME}.${LLM_SAFE}.${DB_SUFFIX_SAFE}.textualized_aggregated_triples.chroma}

# Derived modes (overridable)
TEXTUALIZE_MODE=${TEXTUALIZE_MODE:-$([ "$MODE" = "test" ] && echo "test" || echo "full")}
VECTOR_MODE=${VECTOR_MODE:-$([ "$MODE" = "test" ] && echo "test" || echo "full")}

echo "[1/11] Parse ontology to structured JSON"
"$PYTHON" -m src.knowledge_base.preprocess_ontology_parser \
  -f "$ONTOLOGY_TTL" \
  -n "$ONTOLOGY_NAME" \
  -v || true

PARSED_ONTO_JSON="${PARSED_ONTO_DIR}/${ONTOLOGY_NAME}.json"
if [ ! -f "$PARSED_ONTO_JSON" ]; then
  echo "ERROR: Parsed ontology JSON not found at $PARSED_ONTO_JSON"
  exit 1
fi

echo "[2/11] Textualize ontology elements ($TEXTUALIZE_MODE)"
"$PYTHON" -m src.knowledge_base.textualize_parsed_ontology \
  --input "$PARSED_ONTO_JSON" \
  --output_dir "$TEXT_ONTO_DIR" \
  --mode "$TEXTUALIZE_MODE" \
  --llm_model "$TEXTUALIZE_LLM_MODEL" \
  $([ "${TEXT_ONTO_RESUME}" = "false" ] && echo "--no-resume" || echo "--resume")

# 确定性命名：<ONTOLOGY_NAME>.<LLM_SAFE>.textualized_ontology_elements.<mode>.json
LLM_SAFE=${TEXTUALIZE_LLM_MODEL//\//_}
LLM_SAFE=${LLM_SAFE// /_}
TEXT_ONTO_JSON="${TEXT_ONTO_DIR}/${ONTOLOGY_NAME}.${LLM_SAFE}.textualized_ontology_elements.${TEXTUALIZE_MODE}.json"
if [ ! -f "${TEXT_ONTO_JSON}" ]; then
  echo "ERROR: Textualized ontology JSON not found at ${TEXT_ONTO_JSON}"
  exit 1
fi
echo "Using textualized ontology: ${TEXT_ONTO_JSON}"

echo "[3/11] Build vector DB for ontology elements ($VECTOR_MODE)"
"$PYTHON" -m src.knowledge_base.build_vector_database_ontology_elements \
  --input "$TEXT_ONTO_JSON" \
  --output "$VECTOR_DB_DIR" \
  --name "$ONTO_VECTOR_DB_NAME" \
  --embedding-model "$EMBEDDING_MODEL" \
  --mode "$VECTOR_MODE" \
  --chunk-size "$CHUNK_SIZE" \
  --chunk-overlap "$CHUNK_OVERLAP" \
  --batch-size "$ONTO_BATCH_SIZE" \
  $([ "${ONTO_RESUME}" = "false" ] && echo "--no-resume" || echo "--resume")

echo "[4/11] Parse VKG mappings from OBDA"
"$PYTHON" -m src.knowledge_base.preprocess_vkg_mapping_parser \
  -f "$OBDA_FILE" \
  -v || true

PARSED_VKG_JSON="resources/vkg_mappings_parsed/$(basename "$OBDA_FILE" .obda)_mappings.json"
if [ ! -f "$PARSED_VKG_JSON" ]; then
  echo "ERROR: Parsed VKG mappings JSON not found at $PARSED_VKG_JSON"
  exit 1
fi

echo "[5/11] Textualize VKG mappings ($TEXTUALIZE_MODE)"
"$PYTHON" -m src.knowledge_base.textualize_vkg_mappings \
  --input "$PARSED_VKG_JSON" \
  --output_dir "$TEXT_VKG_DIR" \
  --mode "$TEXTUALIZE_MODE" \
  --llm_model "$TEXTUALIZE_LLM_MODEL" \
  $([ "${TEXT_VKG_RESUME}" = "false" ] && echo "--no-resume" || echo "--resume")

# 确定性命名：<VKG_NAME>.<LLM_SAFE>.textualized_vkg_mappings.<mode>.jsonl
TEXT_VKG_JSONL="${TEXT_VKG_DIR}/${VKG_NAME}.${LLM_SAFE}.textualized_vkg_mappings.${TEXTUALIZE_MODE}.jsonl"
if [ ! -f "${TEXT_VKG_JSONL}" ]; then
  echo "ERROR: Textualized VKG mappings JSONL not found at ${TEXT_VKG_JSONL}"
  exit 1
fi
echo "Using textualized VKG mappings: ${TEXT_VKG_JSONL}"

echo "[6/11] Build vector DB for VKG mappings ($VECTOR_MODE)"
"$PYTHON" -m src.knowledge_base.build_vector_database_vkg_mappings \
  --input "$TEXT_VKG_JSONL" \
  --output "$VECTOR_DB_DIR" \
  --name "$VKG_VECTOR_DB_NAME" \
  --embedding-model "$EMBEDDING_MODEL" \
  --mode "$VECTOR_MODE" \
  --chunk-size "$CHUNK_SIZE" \
  --chunk-overlap "$CHUNK_OVERLAP" \
  --batch-size "$VKG_BATCH_SIZE" \
  $([ "${VKG_RESUME}" = "false" ] && echo "--no-resume" || echo "--resume")

echo "[7/11] Aggregate triples by subject (.nt → .by_subject.jsonl)"
AGG_JSONL_FILE="${AGG_JSONL_DIR}/${VKG_NAME}-materialized.by_subject.jsonl"
"$PYTHON" -m src.knowledge_base.preprocess_aggregate_triples_by_subject \
  -i "$AGG_NT_FILE" \
  -o "$AGG_JSONL_FILE" || true

if [ ! -f "$AGG_JSONL_FILE" ]; then
  echo "WARNING: Aggregated JSONL not found at $AGG_JSONL_FILE; skipping textualization and vector DB build"
else
  echo "Using aggregated JSONL: ${AGG_JSONL_FILE}"

  # Ensure directories exist before subsequent steps
  mkdir -p "$AGG_JSONL_DIR" "$TEXT_AGG_DIR" "$VECTOR_DB_DIR"

  echo "[8/11] Textualize aggregated triples ($TEXTUALIZE_MODE)"
  "$PYTHON" -m src.knowledge_base.textualize_aggregated_triples \
    --input "$AGG_JSONL_FILE" \
    --output_dir "$TEXT_AGG_DIR" \
    --mode "$TEXTUALIZE_MODE" \
    --llm_model "$TEXTUALIZE_LLM_MODEL" \
    --max_workers "$AGG_MAX_WORKERS" || true

  # Probe textualized output file name (by convention)
  LLM_SAFE_TXT=${TEXTUALIZE_LLM_MODEL//\//_}
  LLM_SAFE_TXT=${LLM_SAFE_TXT// /_}
  TEXT_AGG_JSON_CANDIDATE="${TEXT_AGG_DIR}/${VKG_NAME}.${LLM_SAFE_TXT}.textualized_aggregated_triples.${TEXTUALIZE_MODE}.json"
  if [ ! -f "$TEXT_AGG_JSON_CANDIDATE" ]; then
    echo "WARNING: Textualized aggregated triples not found at $TEXT_AGG_JSON_CANDIDATE; will try batch directory mode for vector DB"
    TEXT_AGG_INPUT_FOR_DB="$TEXT_AGG_DIR"
  else
    echo "Using textualized aggregated triples file: $TEXT_AGG_JSON_CANDIDATE"
    TEXT_AGG_INPUT_FOR_DB="$TEXT_AGG_JSON_CANDIDATE"
  fi

  echo "[9/11] Build vector DB for aggregated triples ($VECTOR_MODE)"
  "$PYTHON" -m src.knowledge_base.build_vector_database_textualized_aggregated_triples \
    --input "$TEXT_AGG_INPUT_FOR_DB" \
    --output "$VECTOR_DB_DIR" \
    --vkg-name "$VKG_NAME" \
    --embedding-model "$EMBEDDING_MODEL" \
    --mode "$VECTOR_MODE" \
    --batch-size "$AGG_BATCH_SIZE" \
    $([ "${AGG_RESUME}" = "false" ] && echo "--no-resume" || echo "--resume")

  AGG_VECTOR_DB_PATH="${VECTOR_DB_DIR}/${AGG_VECTOR_DB_NAME}"
  if [ -d "$AGG_VECTOR_DB_PATH" ]; then
    echo "Aggregated triples vector DB ready at: $AGG_VECTOR_DB_PATH"
  else
    echo "WARNING: Aggregated triples vector DB not found at: $AGG_VECTOR_DB_PATH"
  fi
fi

echo "[10/11] Prepare Text-to-SPARQL knowledge base (dataset download may take time)"
"$PYTHON" -m src.knowledge_base.preprocess_text_to_sparql_datasets || true

LATEST_T2S_JSON=$(ls -t ${T2S_DIR}/text_to_sparql_kb_*.json 2>/dev/null | head -n 1 || true)
if [ -z "${LATEST_T2S_JSON}" ]; then
  echo "WARNING: No text-to-sparql KB found in $T2S_DIR; skipping vector DB for it"
else
  echo "Using text-to-sparql KB: $LATEST_T2S_JSON"
  echo "[11/11] Build vector DB for Text-to-SPARQL ($VECTOR_MODE)"
  "$PYTHON" -m src.knowledge_base.build_vector_database_text_to_sparql \
    --input "$LATEST_T2S_JSON" \
    --output "$VECTOR_DB_DIR" \
    --name "$T2S_VECTOR_DB_NAME" \
    --embedding-model "$EMBEDDING_MODEL" \
    --mode "$VECTOR_MODE" \
    --batch-size "$T2S_BATCH_SIZE" \
    $([ "${T2S_RESUME}" = "false" ] && echo "--no-resume" || echo "--resume")
fi

echo "✅ All knowledge bases built successfully."