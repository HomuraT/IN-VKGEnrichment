#!/bin/bash
# 向量数据库服务停止脚本
# 停止所有运行中的向量数据库服务

# 注意：不使用 set -e，避免在停止单个服务失败时终止整个脚本

LOGS_DIR="logs/vector_services"

echo "========================================="
echo "停止向量数据库服务"
echo "========================================="
echo ""

if [ ! -d "$LOGS_DIR" ]; then
  echo "⚠️  日志目录不存在: $LOGS_DIR"
  echo "没有服务需要停止"
  exit 0
fi

# 停止所有服务
stopped_count=0
failed_count=0

for pid_file in "$LOGS_DIR"/*.pid; do
  if [ ! -f "$pid_file" ]; then
    continue
  fi
  
  pid=$(cat "$pid_file")
  service_name=$(basename "$pid_file" .pid)
  
  echo "停止服务: $service_name (PID: $pid)"
  
  # 检查进程是否存在
  if ps -p $pid > /dev/null 2>&1; then
    # 尝试优雅停止
    kill $pid 2>/dev/null || true
    
    # 等待最多 5 秒
    for i in {1..5}; do
      if ! ps -p $pid > /dev/null 2>&1; then
        echo "  ✅ 已停止"
        ((stopped_count++))
        break
      fi
      sleep 1
    done
    
    # 如果还在运行，强制停止
    if ps -p $pid > /dev/null 2>&1; then
      echo "  ⚠️  优雅停止失败，强制停止..."
      kill -9 $pid 2>/dev/null || true
      sleep 1
      if ! ps -p $pid > /dev/null 2>&1; then
        echo "  ✅ 已强制停止"
        ((stopped_count++))
      else
        echo "  ❌ 停止失败"
        ((failed_count++))
      fi
    fi
  else
    echo "  ℹ️  进程已不存在"
  fi
  
  # 删除 PID 文件
  rm "$pid_file"
done

echo ""
echo "========================================="
echo "停止完成"
echo "========================================="
echo "成功停止: $stopped_count 个服务"
if [ $failed_count -gt 0 ]; then
  echo "停止失败: $failed_count 个服务"
fi

