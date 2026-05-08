#!/bin/bash
# 向量数据库服务启动脚本
# 为每个向量库启动一个独立的服务实例

set -e

BASE_DIR="resources/vector_databases"
LOGS_DIR="logs/vector_services"
EMBEDDING_MODEL="local_qwen_3_8b_embedding"

# 创建日志目录
mkdir -p "$LOGS_DIR"

# 服务列表：名称|端口|日志前缀|集合名称
SERVICES=(
  "bgee_v14_genex.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_ontology_elements|8001|ontology|ontology_elements"
  "bgee_v14_genex.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_vkg_mappings|8002|mappings|vkg_mappings"
  "bgee_v14_genex.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_aggregated_triples|8003|triples|aggregated_triples"
  "npd.local_qwen_2_5_7b.local_qwen_3_8b_embedding.textualized_vkg_mappings|8101|npd_mappings|vkg_mappings"
)

echo "========================================="
echo "启动向量数据库服务"
echo "========================================="
echo "基础目录: $BASE_DIR"
echo "日志目录: $LOGS_DIR"
echo "嵌入模型: $EMBEDDING_MODEL"
echo ""

# 批量启动服务
for service in "${SERVICES[@]}"; do
  IFS='|' read -r db_name port log_prefix collection <<< "$service"
  
  echo "----------------------------------------"
  echo "启动服务: $log_prefix"
  echo "  数据库名称: $db_name"
  echo "  端口: $port"
  echo "  集合: $collection"
  
  # 检查端口是否已被占用
  if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "  ⚠️  端口 $port 已被占用，跳过启动"
    continue
  fi
  
  # 启动服务
  nohup uv run -m src.services.vector_db.service \
    --db-name "$db_name" \
    --port "$port" \
    --base-dir "$BASE_DIR" \
    --collection "$collection" \
    --embedding-model "$EMBEDDING_MODEL" \
    > "$LOGS_DIR/${log_prefix}.log" 2>&1 &
  
  pid=$!
  echo $pid > "$LOGS_DIR/${log_prefix}.pid"
  echo "  ✅ 已启动 (PID: $pid)"
  echo "  📝 日志: $LOGS_DIR/${log_prefix}.log"
  
  # 等待服务启动
  sleep 2
done

echo ""
echo "========================================="
echo "所有服务已启动！"
echo "========================================="
echo ""
echo "服务列表："
for service in "${SERVICES[@]}"; do
  IFS='|' read -r db_name port log_prefix collection <<< "$service"
  pid_file="$LOGS_DIR/${log_prefix}.pid"
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    if ps -p $pid > /dev/null 2>&1; then
      echo "  ✅ $log_prefix - http://localhost:$port (PID: $pid)"
    else
      echo "  ❌ $log_prefix - 启动失败，请查看日志: $LOGS_DIR/${log_prefix}.log"
    fi
  fi
done

echo ""
echo "查看日志: tail -f $LOGS_DIR/*.log"
echo "停止服务: bash script/stop_vector_services.sh"
echo "健康检查: curl http://localhost:8001/health"

