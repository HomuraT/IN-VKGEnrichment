"""
本体概念提取工具

从 OBDA 映射文件中提取所有被使用的本体类和属性 URI，构建白名单过滤器。

流程：
1. 使用 ontop 将 OBDA 转换为 R2RML (Turtle 格式)
2. 使用 RDFLib 解析 R2RML 图
3. 提取 rr:class 和 rr:predicate 中的本体 URI
4. 构建白名单集合，支持过滤和验证
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Set, Tuple, Dict, List, Optional, Any
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

logger = logging.getLogger(__name__)

# R2RML 命名空间
RR = Namespace("http://www.w3.org/ns/r2rml#")

# XSD 数据类型前缀（需要过滤）
XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"

# RDF/RDFS/OWL 内置词汇（总是允许）
BUILTIN_VOCABULARY = {
    # RDF
    str(RDF.type),
    str(RDF.Property),
    str(RDF.Statement),
    str(RDF.subject),
    str(RDF.predicate),
    str(RDF.object),
    # RDFS
    str(RDFS.Class),
    str(RDFS.label),
    str(RDFS.comment),
    str(RDFS.subClassOf),
    str(RDFS.subPropertyOf),
    str(RDFS.domain),
    str(RDFS.range),
    str(RDFS.seeAlso),
    str(RDFS.isDefinedBy),
    # OWL
    str(OWL.Class),
    str(OWL.ObjectProperty),
    str(OWL.DatatypeProperty),
    str(OWL.AnnotationProperty),
    str(OWL.Thing),
    str(OWL.Nothing),
    str(OWL.equivalentClass),
    str(OWL.equivalentProperty),
    str(OWL.sameAs),
    str(OWL.differentFrom),
}


def obda_to_r2rml(
    obda_file: str,
    output_ttl: Optional[str] = None,
    ontology_file: Optional[str] = None,
    force: bool = False
) -> str:
    """
    使用 ontop 将 OBDA 转换为 R2RML TTL 格式
    
    Args:
        obda_file: OBDA 文件路径
        output_ttl: 输出 TTL 文件路径（可选，默认自动生成到 resources/vkg_mappings_r2rml/）
        ontology_file: 本体文件路径（可选，提高转换质量）
        force: 强制重新转换，即使缓存存在
    
    Returns:
        str: 生成的 TTL 文件路径
    
    Raises:
        FileNotFoundError: OBDA 文件不存在
        RuntimeError: ontop 转换失败
    """
    obda_path = Path(obda_file)
    if not obda_path.exists():
        raise FileNotFoundError(f"OBDA file not found: {obda_file}")
    
    # 自动生成输出路径
    if output_ttl is None:
        output_dir = Path("resources/vkg_mappings_r2rml")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ttl = output_dir / f"{obda_path.stem}.r2rml.ttl"
    else:
        output_ttl = Path(output_ttl)
        output_ttl.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查缓存（如果 TTL 存在且较新，直接使用）
    if output_ttl.exists() and not force:
        if output_ttl.stat().st_mtime > obda_path.stat().st_mtime:
            logger.info(f"Using cached R2RML file: {output_ttl}")
            return str(output_ttl)
    
    # 构建 ontop 命令
    cmd = ["ontop", "mapping", "to-r2rml", "-i", str(obda_path), "-o", str(output_ttl), "--force"]
    
    if ontology_file:
        ontology_path = Path(ontology_file)
        if ontology_path.exists():
            cmd.extend(["-t", str(ontology_path)])
        else:
            logger.warning(f"Ontology file not found, skipping: {ontology_file}")
    
    # 执行转换
    logger.info(f"Converting OBDA to R2RML: {obda_path} -> {output_ttl}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout
        raise RuntimeError(f"ontop conversion failed: {error_msg}")
    
    if not output_ttl.exists():
        raise RuntimeError(f"ontop conversion succeeded but output file not found: {output_ttl}")
    
    logger.info(f"R2RML conversion successful: {output_ttl}")
    return str(output_ttl)


def extract_ontology_concepts_from_r2rml(r2rml_file: str) -> Tuple[Set[str], Set[str]]:
    """
    从 R2RML 文件提取本体类和属性 URI
    
    Args:
        r2rml_file: R2RML TTL 文件路径
    
    Returns:
        (classes, properties): 类 URI 集合和属性 URI 集合
    
    Raises:
        FileNotFoundError: R2RML 文件不存在
        Exception: RDF 解析失败
    """
    r2rml_path = Path(r2rml_file)
    if not r2rml_path.exists():
        raise FileNotFoundError(f"R2RML file not found: {r2rml_file}")
    
    logger.info(f"Parsing R2RML file: {r2rml_file}")
    
    # 解析 RDF 图
    g = Graph()
    try:
        g.parse(str(r2rml_path), format="turtle")
    except Exception as e:
        raise Exception(f"Failed to parse R2RML file: {e}")
    
    logger.debug(f"Parsed {len(g)} triples from R2RML")
    
    classes = set()
    properties = set()
    
    # 1. 提取所有 rr:class（主语的类型）
    # 注意：使用 RR['class'] 而不是 RR.class_（class 是 Python 保留字）
    for s, p, o in g.triples((None, RR['class'], None)):
        if isinstance(o, URIRef):
            uri = str(o)
            if not uri.startswith(XSD_NAMESPACE):
                classes.add(uri)
    
    # 4. 提取所有 rr:predicate（属性）
    for s, p, o in g.triples((None, RR.predicate, None)):
        if isinstance(o, URIRef):
            uri = str(o)
            if not uri.startswith(XSD_NAMESPACE):
                properties.add(uri)
    
    # 2. 提取 rr:objectMap 中的 rr:class（对象的类型）
    for om in g.subjects(RDF.type, RR.ObjectMap):
        for s, p, o in g.triples((om, RR['class'], None)):
            if isinstance(o, URIRef):
                uri = str(o)
                if not uri.startswith(XSD_NAMESPACE):
                    classes.add(uri)
    
    # 3. 提取 rr:RefObjectMap 中引用的类（通过 join 连接的类）
    for rom in g.subjects(RDF.type, RR.RefObjectMap):
        for s, p, parent_tm in g.triples((rom, RR.parentTriplesMap, None)):
            # 查找 parent TriplesMap 的 subject class
            for sm in g.objects(parent_tm, RR.subjectMap):
                for s2, p2, o2 in g.triples((sm, RR['class'], None)):
                    if isinstance(o2, URIRef):
                        uri = str(o2)
                        if not uri.startswith(XSD_NAMESPACE):
                            classes.add(uri)
    
    # 5. 提取 rr:object（固定对象值，可能是类实例）
    # 例如：rr:predicate up:rank; rr:object up:Species
    for s, p, o in g.triples((None, RR.object, None)):
        if isinstance(o, URIRef):
            uri = str(o)
            if not uri.startswith(XSD_NAMESPACE):
                # 这些可能是类实例，也加入 classes
                classes.add(uri)
    
    logger.info(f"Extracted {len(classes)} classes and {len(properties)} properties")
    return classes, properties


class OntologyConceptWhitelist:
    """基于 OBDA 映射的本体概念白名单"""
    
    def __init__(
        self,
        obda_file: Optional[str] = None,
        ontology_file: Optional[str] = None,
        auto_convert: bool = True,
        include_builtin: bool = True,
        json_cache_file: Optional[str] = None
    ):
        """
        初始化白名单
        
        Args:
            obda_file: OBDA 文件路径
            ontology_file: 本体文件路径（可选，提高转换质量）
            auto_convert: 是否自动转换 OBDA 到 R2RML
            include_builtin: 是否包含 RDF/RDFS/OWL 内置词汇
            json_cache_file: JSON 缓存文件路径（如果存在则直接加载）
        """
        self.classes: Set[str] = set()
        self.properties: Set[str] = set()
        self.all_concepts: Set[str] = set()
        
        # 优先从 JSON 缓存加载
        if json_cache_file and Path(json_cache_file).exists():
            logger.info(f"Loading whitelist from JSON cache: {json_cache_file}")
            self.load_from_json(json_cache_file)
        elif obda_file:
            # 从 OBDA 提取
            if auto_convert:
                r2rml_file = obda_to_r2rml(obda_file, ontology_file=ontology_file)
            else:
                r2rml_file = Path(obda_file).with_suffix('.r2rml.ttl')
            
            self.classes, self.properties = extract_ontology_concepts_from_r2rml(r2rml_file)
            self.all_concepts = self.classes | self.properties
            
            # 添加内置词汇
            if include_builtin:
                self._add_builtin_vocabulary()
        else:
            raise ValueError("Must provide either obda_file or json_cache_file")
    
    def _add_builtin_vocabulary(self):
        """添加 RDF/RDFS/OWL 常用内置词汇（总是允许）"""
        self.all_concepts.update(BUILTIN_VOCABULARY)
        self.properties.update(BUILTIN_VOCABULARY)
        logger.debug(f"Added {len(BUILTIN_VOCABULARY)} builtin vocabulary URIs")
    
    def is_valid_class(self, uri: str) -> bool:
        """检查是否为有效的类 URI"""
        return uri in self.classes
    
    def is_valid_property(self, uri: str) -> bool:
        """检查是否为有效的属性 URI"""
        return uri in self.properties
    
    def is_valid(self, uri: str) -> bool:
        """检查是否为有效的本体概念 URI（类或属性）"""
        return uri in self.all_concepts
    
    def is_allowed(self, uri: str) -> bool:
        """检查 URI 是否在白名单中（is_valid 的别名）"""
        return self.is_valid(uri)
    
    def filter_items(self, items: List[Dict[str, Any]], uri_key: str = 'uri') -> List[Dict[str, Any]]:
        """
        过滤检索结果列表，只保留白名单中的 URI
        
        Args:
            items: 检索结果列表（如 ontology_items）
            uri_key: URI 字段名（默认 'uri'）
        
        Returns:
            过滤后的列表
        """
        filtered = [item for item in items if self.is_valid(item.get(uri_key, ''))]
        
        if len(filtered) < len(items):
            logger.debug(f"Filtered {len(items) - len(filtered)} items not in whitelist")
        
        return filtered
    
    def get_statistics(self) -> Dict[str, int]:
        """返回白名单统计信息"""
        return {
            'total_concepts': len(self.all_concepts),
            'classes': len(self.classes),
            'properties': len(self.properties)
        }
    
    def save_to_json(self, output_path: str) -> None:
        """
        保存白名单到 JSON 文件
        
        Args:
            output_path: 输出 JSON 文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'classes': sorted(list(self.classes)),
            'properties': sorted(list(self.properties)),
            'statistics': self.get_statistics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Whitelist saved to JSON: {output_path}")
    
    def load_from_json(self, json_path: str) -> None:
        """
        从 JSON 文件加载白名单（避免重复转换）
        
        Args:
            json_path: JSON 文件路径
        
        Raises:
            FileNotFoundError: JSON 文件不存在
            ValueError: JSON 格式不正确
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON whitelist file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'classes' not in data or 'properties' not in data:
            raise ValueError(f"Invalid whitelist JSON format: missing 'classes' or 'properties'")
        
        self.classes = set(data['classes'])
        self.properties = set(data['properties'])
        self.all_concepts = self.classes | self.properties
        
        logger.info(f"Whitelist loaded from JSON: {json_path}")
        logger.debug(f"Loaded {len(self.classes)} classes and {len(self.properties)} properties")


def extract_and_save_whitelist(
    obda_file: str,
    output_json: str,
    ontology_file: Optional[str] = None
) -> OntologyConceptWhitelist:
    """
    便捷函数：从 OBDA 提取白名单并保存到 JSON
    
    Args:
        obda_file: OBDA 文件路径
        output_json: 输出 JSON 文件路径
        ontology_file: 本体文件路径（可选）
    
    Returns:
        OntologyConceptWhitelist 实例
    """
    whitelist = OntologyConceptWhitelist(
        obda_file=obda_file,
        ontology_file=ontology_file,
        auto_convert=True,
        include_builtin=True
    )
    
    whitelist.save_to_json(output_json)
    
    return whitelist

