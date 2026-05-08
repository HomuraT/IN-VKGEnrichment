from __future__ import annotations

from typing import Dict, List, Tuple, Union
from pathlib import Path
import json

from rdflib import Graph
from rdflib.term import Node

from src.config.logging_config import get_logger


logger = get_logger(__name__)


def _node_to_serialized_str(node: Node) -> str:
    """
    将 RDF 节点序列化为字符串（N3 表示，尽量保留数据类型/语言标签）。

    输入:
        node (rdflib.term.Node): RDF 节点 (URIRef | BNode | Literal)

    输出:
        str: 节点的可序列化字符串表示（N3 格式）
    """
    # 使用 n3() 以保留 <URI>、"literal"@lang、"literal"^^<datatype> 等信息
    try:
        return node.n3()  # type: ignore[attr-defined]
    except Exception:
        return str(node)


def aggregate_triples_by_subject_to_jsonl(
    nt_file: Union[str, Path],
    output_jsonl: Union[str, Path],
) -> int:
    """
    读取 N-Triples 文件，按主语聚合为 {"subject": str, "po": [[pred, obj], ...]} 并写入 JSONL。

    输入:
        nt_file (str | Path): N-Triples 源文件路径 (.nt)
        output_jsonl (str | Path): 输出 JSONL 文件路径

    输出:
        int: 写出的主语条目数（JSON 行数）
    """
    nt_path = Path(nt_file)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始解析 RDF 文件: {nt_path}")
    g = Graph()
    # 自动格式回退：优先按后缀推断 .nt → 尝试 nt, 失败则 turtle/n3；否则先 turtle
    suffix = nt_path.suffix.lower()
    candidate_formats = ["nt", "turtle", "n3"] if suffix == ".nt" else ["turtle", "n3", "nt"]

    last_err: Exception | None = None
    for fmt in candidate_formats:
        try:
            logger.info(f"尝试以格式 '{fmt}' 解析……")
            g.parse(str(nt_path), format=fmt)
            logger.info(f"以格式 '{fmt}' 解析成功。")
            break
        except Exception as e:
            last_err = e
            logger.warning(f"以格式 '{fmt}' 解析失败：{e}")
    else:
        # 全部失败
        raise last_err if last_err else RuntimeError("无法解析输入文件，未知错误")

    logger.info(f"解析完成，共 {len(g)} 条三元组。开始按主语聚合……")

    aggregated: Dict[str, List[Tuple[str, str]]] = {}
    for s, p, o in g:
        s_str = _node_to_serialized_str(s)
        p_str = _node_to_serialized_str(p)
        o_str = _node_to_serialized_str(o)
        if s_str not in aggregated:
            aggregated[s_str] = []
        aggregated[s_str].append((p_str, o_str))

    logger.info(f"聚合完成，共 {len(aggregated)} 个主语。写入 JSONL: {out_path}")
    with out_path.open("w", encoding="utf-8") as fout:
        for subject, po_list in aggregated.items():
            record = {"subject": subject, "po": po_list}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"写入完成: {out_path}")
    return len(aggregated)


def main() -> None:
    """
    简单 CLI：按主语聚合 `.nt` 并写 JSONL（不去重）。

    输入:
        通过命令行参数指定：
            - --input / -i: 输入 .nt 文件路径
            - --output / -o: 输出 .jsonl 文件路径（可选，缺省采用默认路径）

    输出:
        None
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate N-Triples by subject and write JSONL (no dedup)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="resources/vkg_triples/bgee_v14_genex-materialized.nt",
        help="Input N-Triples file path (.nt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help=(
            "Output JSONL file path. Default: resources/vkg_triples_aggregated/"
            "bgee_v14_genex-materialized.by_subject.jsonl"
        ),
    )

    args = parser.parse_args()
    input_file = Path(args.input)
    if args.output:
        output_file = Path(args.output)
    else:
        # 默认输出路径
        output_file = Path(
            "resources/vkg_triples_aggregated/bgee_v14_genex-materialized.by_subject.jsonl"
        )

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    num_subjects = aggregate_triples_by_subject_to_jsonl(
        nt_file=input_file, output_jsonl=output_file
    )
    logger.info(f"合计写入 {num_subjects} 个主语的聚合记录到 {output_file}")


if __name__ == "__main__":
    main()


