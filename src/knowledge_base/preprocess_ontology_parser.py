from typing import List, Tuple, Union, Optional, Dict, Set
from pathlib import Path
import logging
import json
import argparse
import sys
from rdflib import Graph, URIRef, BNode, Literal, Namespace
from rdflib.term import Node
from rdflib.namespace import RDF, RDFS, OWL
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Restriction:
    """
    OWL限制条件数据结构
    
    Attributes:
        restriction_type: 限制类型
        on_property: 限制作用的属性
        constraints: 限制条件的字典，包含value, cardinality, data_range等
    """
    restriction_type: str
    on_property: Optional[str] = None
    constraints: Dict[str, Union[str, int]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.constraints is None:
            self.constraints = {}


@dataclass
class FlexibleEntityInfo:
    """
    灵活的实体信息聚合数据结构
    
    不预定义固定字段，动态收集所有存在的属性信息
    
    Attributes:
        uri: 实体的URI
        entity_type: 实体类型 (Class, ObjectProperty, DataProperty, NamedIndividual等)
        properties: 所有属性的字典，键为属性URI，值为该属性的所有值的集合
        structural_relations: 结构化关系 (父子类、定义域值域等)
        restrictions: OWL限制条件列表（仅对类有效）
    """
    uri: str
    entity_type: str
    properties: Dict[str, Set[str]] = None
    structural_relations: Dict[str, Set[str]] = None
    restrictions: List[Restriction] = None
    
    def __post_init__(self):
        """初始化后处理，确保字典和列表属性不为None"""
        if self.properties is None:
            self.properties = defaultdict(set)
        if self.structural_relations is None:
            self.structural_relations = defaultdict(set)
        if self.restrictions is None:
            self.restrictions = []
    
    def add_property(self, property_uri: str, value: str):
        """添加属性值"""
        self.properties[property_uri].add(value)
    
    def add_structural_relation(self, relation_type: str, target_uri: str):
        """添加结构化关系"""
        self.structural_relations[relation_type].add(target_uri)
    
    def get_property_values(self, property_uri: str) -> Set[str]:
        """获取指定属性的所有值"""
        return self.properties.get(property_uri, set())
    
    def get_first_property_value(self, property_uri: str) -> Optional[str]:
        """获取指定属性的第一个值"""
        values = self.get_property_values(property_uri)
        return next(iter(values)) if values else None
    
    @property
    def label(self) -> Optional[str]:
        """获取标签（rdfs:label）"""
        return self.get_first_property_value(str(RDFS.label))
    
    @property
    def comment(self) -> Optional[str]:
        """获取注释（rdfs:comment）"""
        return self.get_first_property_value(str(RDFS.comment))
    
    def to_dict(self) -> Dict:
        """转换为字典格式，便于JSON序列化"""
        return {
            'uri': self.uri,
            'entity_type': self.entity_type,
            'properties': {k: list(v) for k, v in self.properties.items()},
            'structural_relations': {k: list(v) for k, v in self.structural_relations.items()},
            'restrictions': [asdict(r) for r in self.restrictions] if self.restrictions else []
        }


class OntologyTripleExtractor:
    """
    本体三元组提取器
    
    用于从TTL或OWL格式的本体文件中提取所有三元组信息。
    """
    
    def __init__(self):
        """
        初始化提取器
        """
        self.graph = Graph()
        self.logger = logging.getLogger(__name__)
        
    def load_ontology_file(self, file_path: Union[str, Path]) -> bool:
        """
        加载本体文件
        
        Args:
            file_path: 本体文件路径，支持TTL和OWL格式
            
        Returns:
            bool: 加载是否成功
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"文件不存在: {file_path}")
                return False
                
            # 根据文件扩展名确定格式
            file_format = self._get_file_format(file_path)
            
            # 解析文件
            self.graph.parse(str(file_path), format=file_format)
            self.logger.info(f"成功加载本体文件 {file_path}，包含 {len(self.graph)} 条三元组")
            return True
            
        except Exception as e:
            self.logger.error(f"加载本体文件失败: {e}")
            return False
    
    def _get_file_format(self, file_path: Path) -> str:
        """
        根据文件扩展名确定RDF格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: RDF格式名称
        """
        suffix = file_path.suffix.lower()
        format_mapping = {
            '.ttl': 'turtle',
            '.owl': 'xml',
            '.rdf': 'xml',
            '.n3': 'n3',
            '.nt': 'nt'
        }
        return format_mapping.get(suffix, 'turtle')  # 默认使用turtle格式
    
    def extract_all_triples(self) -> List[Tuple[str, str, str]]:
        """
        提取所有三元组
        
        Returns:
            List[Tuple[str, str, str]]: 三元组列表，每个元素为(主语, 谓语, 宾语)的字符串形式
        """
        triples = []
        for subject, predicate, obj in self.graph:
            s_str = self._node_to_string(subject)
            p_str = self._node_to_string(predicate)
            o_str = self._node_to_string(obj)
            triples.append((s_str, p_str, o_str))
            
        self.logger.info(f"提取了 {len(triples)} 条三元组")
        return triples
    
    def _node_to_string(self, node: Node) -> str:
        """
        将RDF节点转换为字符串形式
        
        Args:
            node: RDF节点（可能是URIRef, BNode, 或 Literal）
            
        Returns:
            str: 节点的字符串表示
        """
        if isinstance(node, URIRef):
            return str(node)
        elif isinstance(node, BNode):
            return f"_:{node}"  # 匿名节点用 "_:" 前缀表示
        elif isinstance(node, Literal):
            # 字面量可能包含数据类型和语言标签
            if node.datatype:
                return f'"{node}"^^{node.datatype}'
            elif node.language:
                return f'"{node}"@{node.language}'
            else:
                return f'"{node}"'
        else:
            return str(node)
    
    def get_triple_count(self) -> int:
        """
        获取三元组数量
        
        Returns:
            int: 三元组总数
        """
        return len(self.graph)
    
    def _aggregate_entities_by_type(self, entity_uris: List[str], entity_type: str) -> Dict[str, FlexibleEntityInfo]:
        """
        按类型聚合实体信息的通用方法
        
        Args:
            entity_uris: 实体URI列表
            entity_type: 实体类型
            
        Returns:
            Dict[str, FlexibleEntityInfo]: 实体URI到FlexibleEntityInfo对象的映射
        """
        entities = {uri: FlexibleEntityInfo(uri=uri, entity_type=entity_type) for uri in entity_uris}
        
        # 定义结构化关系映射
        structural_mappings = {
            RDFS.subClassOf: ("parent_class", "sub_class"),
            RDFS.subPropertyOf: ("parent_property", "sub_property"),
            RDFS.domain: ("domain", None),
            RDFS.range: ("range", None),
            OWL.equivalentClass: ("equivalent_class", None),
            OWL.equivalentProperty: ("equivalent_property", None),
            OWL.disjointWith: ("disjoint_with", None),
            OWL.inverseOf: ("inverse_of", None),
            RDF.type: ("rdf_type", None)
        }
        
        # 收集每个实体的所有属性信息
        for entity_uri, entity_info in entities.items():
            entity_ref = URIRef(entity_uri)
            
            for predicate, obj in self.graph.predicate_objects(entity_ref):
                predicate_uri = str(predicate)
                
                # 处理结构化关系
                if predicate in structural_mappings:
                    relation_type, reverse_type = structural_mappings[predicate]
                    
                    if isinstance(obj, URIRef):
                        entity_info.add_structural_relation(relation_type, str(obj))
                        # 添加反向关系（如果适用）
                        if reverse_type and str(obj) in entities:
                            entities[str(obj)].add_structural_relation(reverse_type, entity_uri)
                    elif isinstance(obj, BNode) and predicate == RDFS.subClassOf:
                        # 处理限制条件
                        restriction = self._parse_restriction(obj)
                        if restriction:
                            entity_info.restrictions.append(restriction)
                else:
                    # 所有其他属性都作为一般属性处理
                    self._add_property_value(entity_info, predicate_uri, obj)
        
        return entities
    
    def _add_property_value(self, entity_info: FlexibleEntityInfo, predicate_uri: str, obj: Node):
        """
        添加属性值的辅助方法
        
        Args:
            entity_info: 实体信息对象
            predicate_uri: 谓语URI
            obj: 对象节点
        """
        if isinstance(obj, (URIRef, Literal)):
            entity_info.add_property(predicate_uri, str(obj))
        elif isinstance(obj, BNode):
            entity_info.add_property(predicate_uri, f"_:{obj}")
    
    def _discover_entities_by_patterns(self, discovery_patterns: List[Tuple[str, any, any]]) -> Set[str]:
        """
        通用实体发现方法
        
        Args:
            discovery_patterns: 发现模式列表，每个元素为(position, predicate, object)
                              position可为'subject'或'object'
                              
        Returns:
            Set[str]: 发现的实体URI集合
        """
        entity_uris = set()
        
        for position, predicate, obj in discovery_patterns:
            if position == 'subject':
                for entity in self.graph.subjects(predicate, obj):
                    if isinstance(entity, URIRef):
                        entity_uris.add(str(entity))
            elif position == 'object':
                for entity in self.graph.objects(predicate, obj):
                    if isinstance(entity, URIRef):
                        entity_uris.add(str(entity))
        
        return entity_uris
    
    def aggregate_classes(self) -> Dict[str, FlexibleEntityInfo]:
        """
        聚合所有类信息
        
        Returns:
            Dict[str, FlexibleEntityInfo]: 类URI到FlexibleEntityInfo对象的映射
        """
        # 定义类发现模式
        class_patterns = [
            ('subject', RDF.type, OWL.Class),
            ('subject', RDFS.subClassOf, None),
            ('object', RDFS.subClassOf, None),
            ('subject', OWL.equivalentClass, None),
            ('object', OWL.equivalentClass, None)
        ]
        
        class_uris = self._discover_entities_by_patterns(class_patterns)
        classes = self._aggregate_entities_by_type(list(class_uris), "Class")
        
        # 为类添加实例信息
        for subject, obj in self.graph.subject_objects(RDF.type):
            if isinstance(subject, URIRef) and isinstance(obj, URIRef):
                obj_uri = str(obj)
                subject_uri = str(subject)
                if obj_uri in classes:
                    classes[obj_uri].add_structural_relation("instance", subject_uri)
        
        self.logger.info(f"聚合了 {len(classes)} 个类的信息")
        return classes
    
    def aggregate_object_properties(self) -> Dict[str, FlexibleEntityInfo]:
        """
        聚合所有对象属性信息
        
        Returns:
            Dict[str, FlexibleEntityInfo]: 对象属性URI到FlexibleEntityInfo对象的映射
        """
        # 定义对象属性发现模式
        object_property_patterns = [
            ('subject', RDF.type, OWL.ObjectProperty),
            ('subject', OWL.inverseOf, None),
            ('object', OWL.inverseOf, None)
        ]
        
        property_uris = self._discover_entities_by_patterns(object_property_patterns)
        
        # 添加通过subPropertyOf发现的对象属性（需要验证）
        for prop in self.graph.subjects(RDFS.subPropertyOf, None):
            if isinstance(prop, URIRef) and (prop, RDF.type, OWL.ObjectProperty) in self.graph:
                property_uris.add(str(prop))
        
        object_properties = self._aggregate_entities_by_type(list(property_uris), "ObjectProperty")
        
        self.logger.info(f"聚合了 {len(object_properties)} 个对象属性的信息")
        return object_properties
    
    def aggregate_data_properties(self) -> Dict[str, FlexibleEntityInfo]:
        """
        聚合所有数据属性信息
        
        Returns:
            Dict[str, FlexibleEntityInfo]: 数据属性URI到FlexibleEntityInfo对象的映射
        """
        data_property_patterns = [('subject', RDF.type, OWL.DatatypeProperty)]
        property_uris = self._discover_entities_by_patterns(data_property_patterns)
        data_properties = self._aggregate_entities_by_type(list(property_uris), "DataProperty")
        
        self.logger.info(f"聚合了 {len(data_properties)} 个数据属性的信息")
        return data_properties
    
    def aggregate_individuals(self) -> Dict[str, FlexibleEntityInfo]:
        """
        聚合所有个体信息
        
        Returns:
            Dict[str, FlexibleEntityInfo]: 个体URI到FlexibleEntityInfo对象的映射
        """
        individual_patterns = [('subject', RDF.type, OWL.NamedIndividual)]
        individual_uris = self._discover_entities_by_patterns(individual_patterns)
        
        # 添加通过rdf:type指向类的个体（排除本体构造）
        ontology_constructs = {OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, 
                             OWL.AnnotationProperty, OWL.Ontology, RDFS.Class}
        
        for subject, obj in self.graph.subject_objects(RDF.type):
            if isinstance(subject, URIRef) and isinstance(obj, URIRef) and obj not in ontology_constructs:
                individual_uris.add(str(subject))
        
        individuals = self._aggregate_entities_by_type(list(individual_uris), "Individual")
        
        self.logger.info(f"聚合了 {len(individuals)} 个个体的信息")
        return individuals
    
    def aggregate_all_separately(self) -> Tuple[Dict[str, FlexibleEntityInfo], 
                                             Dict[str, FlexibleEntityInfo], 
                                             Dict[str, FlexibleEntityInfo],
                                             Dict[str, FlexibleEntityInfo]]:
        """
        分别聚合所有本体信息
        
        Returns:
            Tuple: (类信息, 对象属性信息, 数据属性信息, 个体信息)
        """
        classes = self.aggregate_classes()
        object_properties = self.aggregate_object_properties()
        data_properties = self.aggregate_data_properties()
        individuals = self.aggregate_individuals()
        
        self.logger.info(f"完成分离聚合：{len(classes)} 个类，{len(object_properties)} 个对象属性，"
                        f"{len(data_properties)} 个数据属性，{len(individuals)} 个个体")
        
        return classes, object_properties, data_properties, individuals
    
    def _parse_restriction(self, restriction_node: BNode) -> Optional[Restriction]:
        """
        解析OWL限制条件
        
        Args:
            restriction_node: 限制条件的匿名节点
            
        Returns:
            Optional[Restriction]: 解析后的限制条件，如果无法解析则返回None
        """
        # 检查是否是限制条件
        if (restriction_node, RDF.type, OWL.Restriction) not in self.graph:
            return None
        
        restriction = Restriction(restriction_type="unknown")
        
        # 获取限制作用的属性
        for on_prop in self.graph.objects(restriction_node, OWL.onProperty):
            if isinstance(on_prop, URIRef):
                restriction.on_property = str(on_prop)
                break
        
        # 定义限制类型映射，简化重复代码
        restriction_mappings = [
            (OWL.allValuesFrom, "allValuesFrom", "value"),
            (OWL.someValuesFrom, "someValuesFrom", "value"),
            (OWL.hasValue, "hasValue", "value"),
            (OWL.onClass, "onClass", "value"),
            (OWL.onDataRange, "onDataRange", "data_range")
        ]
        
        cardinality_mappings = [
            (OWL.cardinality, "cardinality"),
            (OWL.minCardinality, "minCardinality"),
            (OWL.maxCardinality, "maxCardinality"),
            (OWL.qualifiedCardinality, "qualifiedCardinality"),
            (OWL.minQualifiedCardinality, "minQualifiedCardinality"),
            (OWL.maxQualifiedCardinality, "maxQualifiedCardinality")
        ]
        
        # 处理值类型限制
        for owl_property, restriction_type, constraint_key in restriction_mappings:
            for value in self.graph.objects(restriction_node, owl_property):
                restriction.restriction_type = restriction_type
                if isinstance(value, (URIRef, Literal)):
                    restriction.constraints[constraint_key] = str(value)
                break
            if restriction.restriction_type != "unknown":
                break
        
        # 处理基数限制
        for owl_property, restriction_type in cardinality_mappings:
            for cardinality in self.graph.objects(restriction_node, owl_property):
                restriction.restriction_type = restriction_type
                if isinstance(cardinality, Literal):
                    try:
                        restriction.constraints["cardinality"] = int(cardinality)
                    except ValueError:
                        pass
                break
            if restriction.restriction_type != "unknown":
                break
        
        return restriction if restriction.on_property else None
    
    def save_triples_to_file(self, triples: List[Tuple[str, str, str]], 
                           output_path: Union[str, Path]) -> bool:
        """
        将三元组保存到文件
        
        Args:
            triples: 三元组列表
            output_path: 输出文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            output_path = Path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                for s, p, o in triples:
                    f.write(f"{s}\t{p}\t{o}\n")
            
            self.logger.info(f"成功保存 {len(triples)} 条三元组到 {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存三元组文件失败: {e}")
            return False
    
    def save_aggregated_entities_to_json(self, 
                                       classes: Dict[str, FlexibleEntityInfo],
                                       object_properties: Dict[str, FlexibleEntityInfo],
                                       data_properties: Dict[str, FlexibleEntityInfo],
                                       individuals: Dict[str, FlexibleEntityInfo],
                                       storage_name: str) -> bool:
        """
        将聚合的实体信息保存到JSON文件
        
        Args:
            classes: 类信息字典
            object_properties: 对象属性信息字典
            data_properties: 数据属性信息字典
            individuals: 个体信息字典
            storage_name: 存储名称
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 创建输出目录
            output_dir = Path("resources/parsed_ontologies")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建输出文件路径
            output_file = output_dir / f"{storage_name}.json"
            
            # 转换为可序列化的格式
            data = {
                'classes': {uri: entity.to_dict() for uri, entity in classes.items()},
                'object_properties': {uri: entity.to_dict() for uri, entity in object_properties.items()},
                'data_properties': {uri: entity.to_dict() for uri, entity in data_properties.items()},
                'individuals': {uri: entity.to_dict() for uri, entity in individuals.items()}
            }
            
            # 保存到JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"成功保存聚合数据到 {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存聚合数据失败: {e}")
            return False


def extract_triples_from_ontology(ontology_file: Union[str, Path] = "resources/vkg_ontologies/bgee_v14_genex.ttl",
                                output_file: Optional[Union[str, Path]] = None) -> List[Tuple[str, str, str]]:
    """
    从本体文件中提取三元组的便捷函数
    
    Args:
        ontology_file: 本体文件路径，默认为 "resources/vkg_ontologies/bgee_v14_genex.ttl"
        output_file: 可选的输出文件路径，如果提供则保存到文件
        
    Returns:
        List[Tuple[str, str, str]]: 提取的三元组列表
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建提取器
    extractor = OntologyTripleExtractor()
    
    # 加载本体文件
    if not extractor.load_ontology_file(ontology_file):
        return []
    
    # 提取三元组
    triples = extractor.extract_all_triples()
    
    # 如果指定了输出文件，则保存
    if output_file:
        extractor.save_triples_to_file(triples, output_file)
    
    return triples


def parse_ontology_to_entities(ontology_file: Union[str, Path], 
                              storage_name: str) -> Tuple[Dict[str, FlexibleEntityInfo], 
                                                        Dict[str, FlexibleEntityInfo], 
                                                        Dict[str, FlexibleEntityInfo],
                                                        Dict[str, FlexibleEntityInfo]]:
    """
    解析本体文件并将实体信息保存到JSON文件
    
    Args:
        ontology_file: 本体文件路径
        storage_name: 存储名称，用于生成JSON文件名
        
    Returns:
        Tuple: (类信息, 对象属性信息, 数据属性信息, 个体信息)
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建提取器
    extractor = OntologyTripleExtractor()
    
    # 加载本体文件
    if not extractor.load_ontology_file(ontology_file):
        print(f"加载本体文件失败: {ontology_file}")
        return {}, {}, {}, {}
    
    print(f"本体包含 {extractor.get_triple_count()} 条三元组")
    
    # 分别聚合不同类型的信息
    classes, object_properties, data_properties, individuals = extractor.aggregate_all_separately()
    
    # 保存到JSON文件
    success = extractor.save_aggregated_entities_to_json(
        classes, object_properties, data_properties, individuals, storage_name
    )
    
    if success:
        print(f"\n=== 解析完成 ===")
        print(f"类: {len(classes)} 个")
        print(f"对象属性: {len(object_properties)} 个")
        print(f"数据属性: {len(data_properties)} 个")
        print(f"个体: {len(individuals)} 个")
        print(f"数据已保存到: resources/parsed_ontologies/{storage_name}.json")
    else:
        print("保存JSON文件失败")
    
    return classes, object_properties, data_properties, individuals


def main():
    """
    主函数：解析命令行参数并执行本体解析
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="本体文件解析工具 - 将TTL/OWL格式的本体文件转换为结构化实体信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python preprocess_ontology_parser.py -f resources/vkg_ontologies/bgee_v14_genex.ttl -n bgee_v14_genex
  python preprocess_ontology_parser.py --file my_ontology.owl --name my_ontology
  python preprocess_ontology_parser.py  # 使用默认参数
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        default="resources/vkg_ontologies/bgee_v14_genex.ttl",
        help="本体文件路径 (支持TTL/OWL格式，默认: resources/vkg_ontologies/bgee_v14_genex.ttl)"
    )
    
    parser.add_argument(
        '-n', '--name',
        type=str,
        default="bgee_v14_genex",
        help="存储名称，用于生成输出JSON文件名 (默认: bgee_v14_genex)"
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
    
    print("=== 本体解析工具 ===")
    print(f"输入文件: {args.file}")
    print(f"存储名称: {args.name}")
    print()
    
    # 检查输入文件是否存在
    ontology_path = Path(args.file)
    if not ontology_path.exists():
        print(f"错误: 本体文件不存在: {args.file}")
        sys.exit(1)
    
    try:
        # 执行本体解析
        classes, object_properties, data_properties, individuals = parse_ontology_to_entities(
            args.file, args.name
        )
        
        # 检查解析结果
        total_entities = len(classes) + len(object_properties) + len(data_properties) + len(individuals)
        if total_entities == 0:
            print("警告: 未解析到任何实体信息")
            sys.exit(1)
        
    except Exception as e:
        print(f"解析过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
