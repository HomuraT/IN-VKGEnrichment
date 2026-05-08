# models
openai_gpt_4o_mini
openai_gpt_4o
local_qwen_2_5_7b
openai_gpt_5_1

# vkg-agent
```shell
# Bgee dataset
RUN_TAG=bgee_vkg_agent \
DATASET=resources/datasets/bgee_dataset_1217.jsonl \
VKG_NAME=bgee_v14_genex \
EMBEDDING_MODEL=local_qwen_3_8b_embedding \
LLM_MODEL_KEY=openai_gpt_4o \
TEXTUALIZE_LLM_MODEL_KEY=local_qwen_2_5_7b \
ONTOLOGY_K=10 \
MAPPINGS_K=10 \
TRIPLES_K=10 \
NUM_CANDIDATES=3 \
ITER_ROUNDS=3 \
USE_SPARQL_DB=true \
EXEC_SPARQL=true \
WORKERS=16 \
EVAL_TIMEOUT=60 \
OUT=runs/${RUN_TAG}/predictions.jsonl \
OUT_DECISION_ONLY=runs/${RUN_TAG}/decisions.jsonl \
OBDA_FILE=resources/vkg_mappings/${VKG_NAME}.obda \
SKIP_BLOCKS="queries" \
EVAL_OUT_DIR=evaluations/${RUN_TAG} \
uv run script/02_run_experiment_and_evaluate.sh

# npd 1216
RUN_TAG=npd_dataset_1216_vkg_agent_simple_prompt \
DATASET=resources/datasets/npd_dataset_1216.jsonl \
VKG_NAME=npd \
EMBEDDING_MODEL=local_qwen_3_8b_embedding \
LLM_MODEL_KEY=openai_gpt_4o_mini \
TEXTUALIZE_LLM_MODEL_KEY=local_qwen_2_5_7b \
ONTOLOGY_K=10 \
MAPPINGS_K=10 \
TRIPLES_K=10 \
NUM_CANDIDATES=3 \
ITER_ROUNDS=1 \
USE_SPARQL_DB=true \
EXEC_SPARQL=true \
WORKERS=16 \
EVAL_TIMEOUT=60 \
OUT=runs/${RUN_TAG}/predictions.jsonl \
OUT_DECISION_ONLY=runs/${RUN_TAG}/decisions.jsonl \
ONTOLOGY_FILE=resources/vkg_ontologies/npd.owl \
OBDA_FILE=resources/vkg_mappings/${VKG_NAME}.obda \
SKIP_BLOCKS="queries" \
EVAL_OUT_DIR=evaluations/${RUN_TAG} \
uv run script/02_run_experiment_and_evaluate.sh
```

# Collect results
```shell
# Bgee
EVAL_NAME=bgee_vkg_agent && uv run python -m src.experiment.summarize_experiments --input evaluations/${EVAL_NAME} --output ./summary/${EVAL_NAME}.csv

# NPD
EVAL_NAME=npd_vkg_agent && uv run python -m src.experiment.summarize_experiments --input evaluations/${EVAL_NAME} --output ./summary/${EVAL_NAME}.csv
```