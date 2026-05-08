## Information-Needs-Guided Virtual Knowledge Graph Enrichment via Large Language Models

### Quick Start

This project uses **uv** for environment management. Make sure you have uv installed before starting.

#### Step 1: Set Up Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

#### Step 2: Configure API Keys

```bash
cp src/config/api_and_models.py.sample src/config/api_and_models.py
# Edit src/config/api_and_models.py and add your API keys
```

#### Step 3: Build Knowledge Base

Choose test mode for quick validation or full mode for complete build:

```bash
# Test mode (quick validation)
MODE=test VKG_NAME=bgee_v14_genex \
  ONTOLOGY_TTL=resources/vkg_ontologies/bgee_v14_genex.ttl \
  OBDA_FILE=resources/vkg_mappings/bgee_v14_genex.obda \
  EMBEDDING_MODEL=local_qwen_3_8b_embedding \
  TEXTUALIZE_LLM_MODEL=local_qwen_2_5_7b \
  uv run script/01_build_knowledge_base.sh

# Full mode (complete build)
MODE=full VKG_NAME=bgee_v14_genex \
  ONTOLOGY_TTL=resources/vkg_ontologies/bgee_v14_genex.ttl \
  OBDA_FILE=resources/vkg_mappings/bgee_v14_genex.obda \
  EMBEDDING_MODEL=local_qwen_3_8b_embedding \
  TEXTUALIZE_LLM_MODEL=local_qwen_2_5_7b \
  uv run script/01_build_knowledge_base.sh
```

#### Step 4: Start Vector Database Services

```bash
bash script/start_vector_services.sh

# Verify services are running
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8004/health
```

**⚠️ Important**: Services must be running before you run experiments.

#### Step 5: Run Experiment

```bash
uv run -m src.experiment.run_experiment \
  --dataset resources/datasets/easybgee_v14_2.jsonl \
  --vkg-name bgee_v14_genex \
  --ontology-file resources/vkg_ontologies/bgee_v14_genex.ttl \
  --obda-file resources/vkg_mappings/bgee_v14_genex.obda \
  --embedding-model local_qwen_3_8b_embedding \
  --llm-model-key openai_gpt_4o \
  --iter-rounds 3 \
  --ontology-k 10 \
  --mappings-k 10 \
  --triples-k 0 \
  --num-candidates 3 \
  --use-sparql-db \
  --exec-sparql \
  --workers 4 \
  --out runs/bgee/predictions.jsonl \
  --out-decision-only runs/bgee/decisions.jsonl
```

#### Step 6: Evaluate Results

```bash
uv run -m src.experiment.evaluate_sparql_results \
  --decisions runs/bgee/openai_gpt_4o.bgee_v14_genex.ont10.map10.tri0.cand3.iter3.decisions.jsonl \
  --keep-decisions-path \
  --out-dir evaluations/bgee \
  --llm-model-key openai_gpt_4o \
  --vkg-name bgee_v14_genex
```

#### Step 7: Generate Summary

```bash
uv run -m src.experiment.summarize_experiments \
  --input evaluations/bgee \
  --output ./summary/bgee.csv
```

---

### Vector Database Services

#### Basic Usage

```bash
# Start all services
bash script/start_vector_services.sh

# Stop all services
bash script/stop_vector_services.sh

# Check service status
curl http://localhost:8001/info
```

Service locations (default):
- Ontology elements: http://localhost:8001
- VKG mappings: http://localhost:8002
- Text to SPARQL: http://localhost:8003

#### Advanced: Generate Custom Configuration

If you have multiple vector databases:

```bash
uv run script/generate_service_config.py \
  --base-dir resources/vector_databases \
  --config-output src/config/vector_db_services.py \
  --script-output script/start_vector_services_generated.sh \
  --embedding-model local_qwen_3_8b_embedding

bash script/start_vector_services_generated.sh
```

#### Troubleshooting Services

**Service won't start:**
```bash
# Check logs
cat logs/vector_services/ontology.log

# Common issues:
# 1. Port already in use → change port in src/config/vector_db_services.py
# 2. Vector database missing → check resources/vector_databases/
# 3. Embedding model error → check src/config/api_and_models.py
```

**Can't connect from experiment:**
```bash
# Verify service is running
curl http://localhost:8001/health

# Check configuration matches
cat src/config/vector_db_services.py
```

---

### Advanced Usage

#### Parameters Reference

**Dataset**: Input JSONL with fields: `id`, `vkg`, `question`, optional `sparql`

**Common Options**:
- `--workers N` - Parallel workers (default: 4)
- `--ontology-k N` - Retrieve N ontology elements (default: 10)
- `--mappings-k N` - Retrieve N mappings (default: 10)
- `--num-candidates N` - Generate N candidate SPARQLs per round
- `--iter-rounds N` - Number of iteration rounds
- `--exec-sparql` - Execute SPARQL queries in evaluation
- `--use-sparql-db` - Use SPARQL database

**Output**: Files are auto-prefixed with config details:
- Input: `runs/bgee/predictions.jsonl`
- Actual output: `runs/bgee/openai_gpt_4o.bgee_v14_genex.ont10.map10.tri0.cand3.iter3.predictions.jsonl`

#### One-Click Run & Evaluate

Run experiment and evaluate in one command:

```bash
RUN_TAG=bgee \
DATASET=resources/datasets/easybgee_v14_2.jsonl \
VKG_NAME=bgee_v14_genex \
ONTOLOGY_FILE=resources/vkg_ontologies/bgee_v14_genex.ttl \
OBDA_FILE=resources/vkg_mappings/bgee_v14_genex.obda \
EMBEDDING_MODEL=local_qwen_3_8b_embedding \
LLM_MODEL_KEY=openai_gpt_4o \
WORKERS=4 \
bash script/02_run_experiment_and_evaluate.sh
```

#### Experiment Parameters in Script

Key environment variables:
- `RUN_TAG` - Folder prefix (e.g., `runs/bgee`)
- `DATASET`, `VKG_NAME`, `ONTOLOGY_FILE`, `OBDA_FILE`
- `EMBEDDING_MODEL`, `LLM_MODEL_KEY`
- `ONTOLOGY_K`, `MAPPINGS_K` - Retrieval counts
- `NUM_CANDIDATES`, `ITER_ROUNDS`
- `WORKERS`, `EXEC_SPARQL`, `USE_SPARQL_DB`
- `EVAL_OUT_DIR` - Evaluation output folder

---

### Notes

- Vector database files are stored in `resources/vector_databases/`
- Output files use naming: `<model>.<vkg>.ont<k>.map<k>.cand<n>.iter<r>.<filename>`
- Logs: `logs/experiment.log` and `logs/vector_services/*.log`
- Auto-resume: If output file exists, completed items are skipped

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{lin2026invkge,
  title={Information-Needs-Guided Virtual Knowledge Graph Enrichment via Large Language Models},
  author={Lin, Ren and Xiao, Guohui and Qi, Guilin and Du, Wenjie and Geng, Yishuai and Xue, Haohan and Yue, Zhiyan and Li, Mingxuan and Di Panfilo, Marco and Lanti, Davide and Hamaz, Kamal and Ding, Linfang},
  booktitle={Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI)},
  year={2026}
}
```
