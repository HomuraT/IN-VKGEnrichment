from typing import List, Dict, Union, Optional
from pathlib import Path
import logging
import json
import argparse
import sys
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VKGMapping:
    """
    VKG映射数据结构
    
    Attributes:
        mapping_id: 映射标识符
        target: RDF目标三元组模板
        source: SQL查询源
    """
    mapping_id: str
    target: str
    source: str


class VKGMappingParser:
    """
    VKG映射解析器
    
    用于解析OBDA格式的VKG映射文件，提取MappingDeclaration中的映射信息。
    """
    
    def __init__(self):
        """
        初始化解析器
        """
        self.logger = logging.getLogger(__name__)
        self.prefixes: Dict[str, str] = {}
        self.mappings: List[VKGMapping] = []
    
    def load_obda_file(self, file_path: Union[str, Path]) -> bool:
        """
        加载OBDA映射文件
        
        Args:
            file_path: OBDA文件路径
            
        Returns:
            bool: 加载是否成功
        """
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"文件不存在: {file_path}")
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析前缀声明
        self._parse_prefixes(content)
        
        # 解析映射声明
        self._parse_mappings(content)
        
        self.logger.info(f"成功解析OBDA文件 {file_path}，包含 {len(self.prefixes)} 个前缀和 {len(self.mappings)} 个映射")
        return True
    
    def _parse_prefixes(self, content: str) -> None:
        """
        解析前缀声明
        
        Args:
            content: OBDA文件内容
        """
        # 查找PrefixDeclaration部分
        prefix_pattern = r'\[PrefixDeclaration\](.*?)\[MappingDeclaration\]'
        prefix_match = re.search(prefix_pattern, content, re.DOTALL)
        
        if not prefix_match:
            self.logger.warning("未找到PrefixDeclaration部分")
            return
        
        prefix_content = prefix_match.group(1)
        
        # 解析每行前缀
        prefix_lines = prefix_content.strip().split('\n')
        for line in prefix_lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # 解析前缀行格式：prefix:   <URI>
            prefix_line_pattern = r'^([^:]+):\s*(.+)$'
            match = re.match(prefix_line_pattern, line)
            if match:
                prefix = match.group(1).strip()
                uri = match.group(2).strip()
                self.prefixes[prefix] = uri
    
    def _parse_mappings(self, content: str) -> None:
        """
        解析映射声明
        
        Args:
            content: OBDA文件内容
        """
        # 查找MappingDeclaration部分
        mapping_pattern = r'\[MappingDeclaration\]\s*@collection\s*\[\[(.*?)\]\]'
        mapping_match = re.search(mapping_pattern, content, re.DOTALL)
        
        if not mapping_match:
            self.logger.error("未找到MappingDeclaration部分")
            return
        
        mapping_content = mapping_match.group(1)
        
        # 分割各个映射块
        mapping_blocks = self._split_mapping_blocks(mapping_content)
        
        for block in mapping_blocks:
            mapping = self._parse_single_mapping(block)
            if mapping:
                self.mappings.append(mapping)
    
    def _split_mapping_blocks(self, mapping_content: str) -> List[str]:
        """
        分割映射块
        
        Args:
            mapping_content: 映射内容
            
        Returns:
            List[str]: 映射块列表
        """
        # 使用mappingId作为分割标志
        blocks = []
        current_block = []
        lines = mapping_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 如果遇到mappingId且当前块不为空，保存当前块并开始新块
            if line.startswith('mappingId') and current_block:
                blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        
        # 添加最后一个块
        if current_block:
            blocks.append('\n'.join(current_block))
        
        return blocks
    
    def _parse_single_mapping(self, block: str) -> Optional[VKGMapping]:
        """
        解析单个映射块
        
        Args:
            block: 映射块内容
            
        Returns:
            Optional[VKGMapping]: 解析的映射对象，失败则返回None
        """
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        mapping_id = None
        target = None
        source = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('mappingId'):
                # 解析mappingId
                mapping_id = line.split('\t', 1)[1].strip() if '\t' in line else line.split(None, 1)[1].strip()
            
            elif line.startswith('target'):
                # 解析target（可能跨多行）
                target_parts = []
                target_line = line.split('\t', 1)[1].strip() if '\t' in line else line.split(None, 1)[1].strip()
                target_parts.append(target_line)
                
                # 检查后续行是否是target的延续（不以关键字开头）
                j = i + 1
                while j < len(lines) and not lines[j].startswith(('mappingId', 'target', 'source')):
                    target_parts.append(lines[j])
                    j += 1
                
                target = self._clean_target(' '.join(target_parts))
                i = j - 1  # 调整索引
            
            elif line.startswith('source'):
                # 解析source（可能跨多行）
                source_parts = []
                source_line = line.split('\t', 1)[1].strip() if '\t' in line else line.split(None, 1)[1].strip()
                source_parts.append(source_line)
                
                # 检查后续行是否是source的延续
                j = i + 1
                while j < len(lines) and not lines[j].startswith(('mappingId', 'target', 'source')):
                    source_parts.append(lines[j])
                    j += 1
                
                source = self._clean_source(' '.join(source_parts))
                i = j - 1  # 调整索引
            
            i += 1
        
        if mapping_id and target and source:
            return VKGMapping(
                mapping_id=mapping_id,
                target=target,
                source=source
            )
        else:
            self.logger.warning(f"映射块解析不完整: mappingId={mapping_id}, target={bool(target)}, source={bool(source)}")
            return None
    
    def _clean_target(self, target: str) -> str:
        """
        清理target字符串
        
        Args:
            target: 原始target字符串
            
        Returns:
            str: 清理后的target字符串
        """
        # 保留完整的 target pattern（包括占位符如 {geneId}），只做基本的空白清理
        # 占位符对理解映射结构和变量绑定规则至关重要
        return ' '.join(target.split())
    
    def _clean_source(self, source: str) -> str:
        """
        清理source字符串
        
        Args:
            source: 原始source字符串
            
        Returns:
            str: 清理后的source字符串
        """
        # 移除多余的空白字符，但保留SQL结构
        return ' '.join(source.split())
    
    def get_mappings(self) -> List[VKGMapping]:
        """
        获取所有映射
        
        Returns:
            List[VKGMapping]: 映射列表
        """
        return self.mappings
    
    def get_prefixes(self) -> Dict[str, str]:
        """
        获取所有前缀
        
        Returns:
            Dict[str, str]: 前缀字典
        """
        return self.prefixes
    
    def get_mapping_count(self) -> int:
        """
        获取映射数量
        
        Returns:
            int: 映射总数
        """
        return len(self.mappings)
    
    def save_mappings_to_json(self, output_path: Union[str, Path], 
                             include_prefixes: bool = True) -> bool:
        """
        将映射保存到JSON文件
        
        Args:
            output_path: 输出文件路径
            include_prefixes: 是否包含前缀信息
            
        Returns:
            bool: 保存是否成功
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建输出数据
        data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_mappings": len(self.mappings),
                "description": "VKG mappings from OBDA file",
                "include_prefixes": include_prefixes
            }
        }
        
        # 添加前缀信息（如果需要）
        if include_prefixes:
            data["prefixes"] = self.prefixes
        
        # 添加映射信息
        mappings_data = []
        for mapping in self.mappings:
            mapping_dict = {
                "mapping_id": mapping.mapping_id,
                "target": mapping.target,
                "source": mapping.source
            }
            mappings_data.append(mapping_dict)
        
        data["mappings"] = mappings_data
        
        # 保存到JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"成功保存 {len(self.mappings)} 个映射到 {output_path}")
        return True


def parse_obda_to_json(obda_file: Union[str, Path], 
                      output_file: Optional[Union[str, Path]] = None,
                      include_prefixes: bool = True) -> List[VKGMapping]:
    """
    从OBDA文件解析映射并保存到JSON的便捷函数
    
    Args:
        obda_file: OBDA文件路径
        output_file: 输出JSON文件路径，如果不指定则自动生成
        include_prefixes: 是否包含前缀信息
        
    Returns:
        List[VKGMapping]: 解析的映射列表
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建解析器
    parser = VKGMappingParser()
    
    # 加载OBDA文件
    if not parser.load_obda_file(obda_file):
        return []
    
    # 生成输出文件名（如果未指定）
    if output_file is None:
        obda_path = Path(obda_file)
        output_file = f"resources/vkg_mappings_parsed/{obda_path.stem}_mappings.json"
    
    # 保存到JSON文件
    success = parser.save_mappings_to_json(output_file, include_prefixes)
    
    if success:
        print(f"\n=== 解析完成 ===")
        print(f"前缀: {len(parser.get_prefixes())} 个")
        print(f"映射: {len(parser.get_mappings())} 个")
        print(f"数据已保存到: {output_file}")
    else:
        print("保存JSON文件失败")
    
    return parser.get_mappings()


def main():
    """
    主函数：解析命令行参数并执行OBDA映射解析
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="VKG映射解析工具 - 将OBDA格式的映射文件转换为JSON格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python preprocess_vkg_mapping_parser.py -f resources/vkg_mappings/bgee_v14_genex.obda
  python preprocess_vkg_mapping_parser.py --file mappings.obda --output my_mappings.json
  python preprocess_vkg_mapping_parser.py  # 使用默认参数
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        default="resources/vkg_mappings/bgee_v14_genex.obda",
        help="OBDA映射文件路径 (默认: resources/vkg_mappings/bgee_v14_genex.obda)"
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help="输出JSON文件路径 (如果不指定，将自动生成)"
    )
    
    parser.add_argument(
        '--no-prefixes',
        action='store_true',
        help="不包含前缀信息"
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=== VKG映射解析工具 ===")
    print(f"输入文件: {args.file}")
    print(f"输出文件: {args.output or '自动生成'}")
    print()
    
    # 检查输入文件是否存在
    obda_path = Path(args.file)
    if not obda_path.exists():
        print(f"错误: OBDA文件不存在: {args.file}")
        sys.exit(1)
    
    # 执行映射解析
    mappings = parse_obda_to_json(
        args.file, 
        args.output,
        include_prefixes=not args.no_prefixes
    )
    
    # 检查解析结果
    if not mappings:
        print("警告: 未解析到任何映射信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
