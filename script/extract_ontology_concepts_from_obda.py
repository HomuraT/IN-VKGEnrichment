#!/usr/bin/env python3
"""
从 OBDA 映射文件提取本体概念白名单

用于测试和独立生成本体概念白名单 JSON 文件。

使用示例:
    python script/extract_ontology_concepts_from_obda.py \
        -f resources/vkg_mappings/bgee_v14_genex.obda \
        -t resources/vkg_ontologies/bgee_v14_genex.ttl \
        -o resources/ontology_concept_whitelists/bgee_v14_genex_whitelist.json
"""

import argparse
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.ontology_concept_extractor import (
    extract_and_save_whitelist,
    OntologyConceptWhitelist
)


def main():
    """主函数：解析命令行参数并执行白名单提取"""
    parser = argparse.ArgumentParser(
        description="从 OBDA 映射文件提取本体概念白名单",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 提取 Bgee 数据集的白名单
  python script/extract_ontology_concepts_from_obda.py \\
    -f resources/vkg_mappings/bgee_v14_genex.obda \\
    -t resources/vkg_ontologies/bgee_v14_genex.ttl

  # 提取 NPD 数据集的白名单
  python script/extract_ontology_concepts_from_obda.py \\
    -f resources/vkg_mappings/npd.obda \\
    -t resources/vkg_ontologies/npd.ttl \\
    -o resources/ontology_concept_whitelists/npd_whitelist.json

  # 强制重新转换（忽略缓存）
  python script/extract_ontology_concepts_from_obda.py \\
    -f resources/vkg_mappings/bgee_v14_genex.obda \\
    --force
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help="OBDA 映射文件路径"
    )
    
    parser.add_argument(
        '-t', '--ontology',
        type=str,
        help="本体文件路径（可选，提高转换质量）"
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help="输出 JSON 文件路径（如果不指定，将自动生成）"
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help="强制重新转换 OBDA 到 R2RML（忽略缓存）"
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="显示详细日志信息"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("本体概念白名单提取工具")
    print("=" * 60)
    print(f"输入 OBDA 文件: {args.file}")
    if args.ontology:
        print(f"本体文件: {args.ontology}")
    print()
    
    # 检查输入文件是否存在
    obda_path = Path(args.file)
    if not obda_path.exists():
        print(f"❌ 错误: OBDA 文件不存在: {args.file}")
        sys.exit(1)
    
    if args.ontology:
        ontology_path = Path(args.ontology)
        if not ontology_path.exists():
            print(f"⚠️  警告: 本体文件不存在: {args.ontology}")
            print("   将继续转换，但可能影响质量")
            args.ontology = None
    
    # 生成输出文件名（如果未指定）
    if args.output is None:
        output_dir = Path("resources/ontology_concept_whitelists")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{obda_path.stem}_whitelist.json")
    
    print(f"输出 JSON 文件: {args.output}")
    print()
    
    # 执行提取
    try:
        print("正在提取本体概念...")
        
        # 如果需要强制重新转换，先删除缓存
        if args.force:
            r2rml_cache = Path("resources/vkg_mappings_r2rml") / f"{obda_path.stem}.r2rml.ttl"
            if r2rml_cache.exists():
                r2rml_cache.unlink()
                print(f"已删除 R2RML 缓存: {r2rml_cache}")
        
        whitelist = extract_and_save_whitelist(
            obda_file=args.file,
            output_json=args.output,
            ontology_file=args.ontology
        )
        
        # 打印统计信息
        stats = whitelist.get_statistics()
        print()
        print("=" * 60)
        print("✓ 提取完成！")
        print("=" * 60)
        print(f"总概念数:   {stats['total_concepts']}")
        print(f"  - 类:     {stats['classes']}")
        print(f"  - 属性:   {stats['properties']}")
        print()
        print(f"白名单已保存到: {args.output}")
        print()
        
        # 显示前几个类和属性示例
        if args.verbose:
            print("示例类 URI (前 10 个):")
            for uri in sorted(list(whitelist.classes))[:10]:
                print(f"  - {uri}")
            print()
            
            print("示例属性 URI (前 10 个):")
            for uri in sorted(list(whitelist.properties))[:10]:
                print(f"  - {uri}")
            print()
        
        print("=" * 60)
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 错误: {e}")
        print("=" * 60)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

