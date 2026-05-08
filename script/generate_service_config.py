#!/usr/bin/env python3
"""
向量数据库服务配置自动生成工具

扫描 resources/vector_databases/ 目录下的所有 .chroma 向量库，
自动生成服务配置文件和启动脚本。
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def scan_vector_databases(base_dir: str) -> List[str]:
    """
    扫描向量数据库目录
    
    Args:
        base_dir: 向量数据库基础目录
    
    Returns:
        向量库名称列表（不含.chroma后缀）
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ 目录不存在: {base_dir}")
        return []
    
    db_names = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.endswith(".chroma"):
            # 去掉 .chroma 后缀
            db_name = item.name[:-7]
            db_names.append(db_name)
    
    return sorted(db_names)


def classify_database(db_name: str) -> Tuple[str, str, int]:
    """
    根据向量库名称分类
    
    Args:
        db_name: 向量库名称
    
    Returns:
        (类型, 日志前缀, 推荐端口号)
    """
    if "ontology_elements" in db_name:
        return ("ontology", "ontology", 8001)
    elif "vkg_mappings" in db_name:
        return ("mappings", "mappings", 8002)
    elif "aggregated_triples" in db_name:
        return ("triples", "triples", 8003)
    elif "text_to_sparql" in db_name:
        return ("queries", "queries", 8004)
    else:
        return ("unknown", "unknown", 8099)


def infer_collection_name(db_name: str) -> str:
    """
    推断集合名称
    
    Args:
        db_name: 向量库名称
    
    Returns:
        集合名称
    """
    if "ontology_elements" in db_name:
        return "ontology_elements"
    elif "vkg_mappings" in db_name:
        return "vkg_mappings"
    elif "aggregated_triples" in db_name:
        return "aggregated_triples"
    elif "text_to_sparql" in db_name:
        return "text_to_sparql"
    else:
        return "default"


def generate_config_file(db_names: List[str], output_path: str, start_port: int = 8001):
    """
    生成配置文件
    
    Args:
        db_names: 向量库名称列表
        output_path: 输出文件路径
        start_port: 起始端口号
    """
    lines = []
    lines.append('"""')
    lines.append('向量数据库服务配置（自动生成）')
    lines.append('')
    lines.append('定义向量库名称到服务URL的映射关系。')
    lines.append('"""')
    lines.append('')
    lines.append('import os')
    lines.append('from typing import Dict')
    lines.append('')
    lines.append('# 默认服务地址映射')
    lines.append('VECTOR_DB_SERVICES: Dict[str, str] = {')
    
    for idx, db_name in enumerate(db_names):
        _, _, suggested_port = classify_database(db_name)
        if suggested_port == 8099:
            # 未知类型，使用递增端口
            port = start_port + idx
        else:
            port = suggested_port
        
        lines.append(f'    "{db_name}": "http://localhost:{port}",')
    
    lines.append('}')
    lines.append('')
    lines.append('')
    lines.append('def get_service_url(db_name: str) -> str:')
    lines.append('    """根据向量库名称获取服务URL"""')
    lines.append('    url = VECTOR_DB_SERVICES.get(db_name)')
    lines.append('    if not url:')
    lines.append('        raise ValueError(f"No service URL found for: {db_name}")')
    lines.append('    return url')
    lines.append('')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ 已生成配置文件: {output_path}")
    print(f"   包含 {len(db_names)} 个向量库")


def generate_start_script(db_names: List[str], output_path: str, base_dir: str, embedding_model: str, start_port: int = 8001):
    """
    生成启动脚本
    
    Args:
        db_names: 向量库名称列表
        output_path: 输出文件路径
        base_dir: 向量数据库基础目录
        embedding_model: 嵌入模型键名
        start_port: 起始端口号
    """
    lines = []
    lines.append('#!/bin/bash')
    lines.append('# 向量数据库服务启动脚本（自动生成）')
    lines.append('')
    lines.append('set -e')
    lines.append('')
    lines.append(f'BASE_DIR="{base_dir}"')
    lines.append('LOGS_DIR="logs/vector_services"')
    lines.append(f'EMBEDDING_MODEL="{embedding_model}"')
    lines.append('')
    lines.append('mkdir -p "$LOGS_DIR"')
    lines.append('')
    lines.append('# 服务列表：名称|端口|日志前缀|集合名称')
    lines.append('SERVICES=(')
    
    for idx, db_name in enumerate(db_names):
        db_type, log_prefix, suggested_port = classify_database(db_name)
        if suggested_port == 8099:
            port = start_port + idx
        else:
            port = suggested_port
        
        collection = infer_collection_name(db_name)
        lines.append(f'  "{db_name}|{port}|{log_prefix}_{idx}|{collection}"')
    
    lines.append(')')
    lines.append('')
    lines.append('echo "========================================="')
    lines.append('echo "启动向量数据库服务"')
    lines.append('echo "========================================="')
    lines.append('')
    lines.append('for service in "${SERVICES[@]}"; do')
    lines.append('  IFS=\'|\' read -r db_name port log_prefix collection <<< "$service"')
    lines.append('  echo "启动: $log_prefix (端口 $port)"')
    lines.append('  ')
    lines.append('  nohup uv run -m src.services.vector_db.service \\')
    lines.append('    --db-name "$db_name" \\')
    lines.append('    --port "$port" \\')
    lines.append('    --base-dir "$BASE_DIR" \\')
    lines.append('    --collection "$collection" \\')
    lines.append('    --embedding-model "$EMBEDDING_MODEL" \\')
    lines.append('    > "$LOGS_DIR/${log_prefix}.log" 2>&1 &')
    lines.append('  ')
    lines.append('  echo $! > "$LOGS_DIR/${log_prefix}.pid"')
    lines.append('  sleep 1')
    lines.append('done')
    lines.append('')
    lines.append('echo "所有服务已启动！"')
    lines.append('echo "停止服务: bash script/stop_vector_services.sh"')
    lines.append('')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    # 添加执行权限
    os.chmod(output_path, 0o755)
    
    print(f"✅ 已生成启动脚本: {output_path}")
    print(f"   包含 {len(db_names)} 个服务")


def main():
    parser = argparse.ArgumentParser(
        description="向量数据库服务配置自动生成工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="resources/vector_databases",
        help="向量数据库基础目录",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default="src/config/vector_db_services.py",
        help="配置文件输出路径",
    )
    parser.add_argument(
        "--script-output",
        type=str,
        default="script/start_vector_services_generated.sh",
        help="启动脚本输出路径",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="local_qwen_3_8b_embedding",
        help="嵌入模型键名",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=8001,
        help="起始端口号",
    )
    
    args = parser.parse_args()
    
    print("========================================")
    print("向量数据库服务配置生成工具")
    print("========================================")
    print(f"扫描目录: {args.base_dir}")
    print("")
    
    # 扫描向量数据库
    db_names = scan_vector_databases(args.base_dir)
    
    if not db_names:
        print("❌ 未找到任何向量数据库")
        sys.exit(1)
    
    print(f"找到 {len(db_names)} 个向量数据库:")
    for idx, db_name in enumerate(db_names, 1):
        db_type, log_prefix, port = classify_database(db_name)
        print(f"  {idx}. {db_name} ({db_type}, 端口 {port})")
    print("")
    
    # 生成配置文件
    generate_config_file(db_names, args.config_output, args.start_port)
    print("")
    
    # 生成启动脚本
    generate_start_script(
        db_names,
        args.script_output,
        args.base_dir,
        args.embedding_model,
        args.start_port
    )
    print("")
    
    print("========================================")
    print("生成完成！")
    print("========================================")
    print(f"配置文件: {args.config_output}")
    print(f"启动脚本: {args.script_output}")
    print("")
    print("下一步:")
    print(f"  1. 检查配置文件: cat {args.config_output}")
    print(f"  2. 启动服务: bash {args.script_output}")
    print("  3. 验证服务: curl http://localhost:8001/health")


if __name__ == "__main__":
    main()

