import argparse
import json
from typing import Dict, Generator, Iterable, List, Optional


def iter_jsonl(path: str) -> Generator[Dict, None, None]:
    """Yield dictionaries parsed from a JSONL file, one per line.

    Cleaning rules:
    - 忽略/移除导入时不需要的元数据字段：
      created_at、updated_at、annotator、annotators、expected_version

    Raises json.JSONDecodeError if a line is not valid JSON.
    """
    ignored_fields = {
        "created_at",
        "updated_at",
        "annotator",
        "annotators",
        "expected_version",
    }
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            rec = json.loads(line_stripped)
            if isinstance(rec, dict):
                for k in list(ignored_fields):
                    if k in rec:
                        rec.pop(k, None)
            yield rec


def load_jsonl(path: str) -> List[Dict]:
    """Load all records from a JSONL file into a list of dictionaries.

    No validation or transformation is performed.
    """
    return list(iter_jsonl(path))


def to_dataframe(records: Iterable[Dict]):
    """Convert an iterable of dicts to a pandas.DataFrame if pandas is available.

    Returns the DataFrame, or None if pandas is not installed.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    return pd.DataFrame(list(records))


def _preview_as_table(records: List[Dict], head: int) -> str:
    df = to_dataframe(records)
    if df is None:
        # Fallback to pretty printing first N dicts
        subset = records[:head]
        return "\n".join(json.dumps(r, ensure_ascii=False) for r in subset)
    return df.head(head).to_string(index=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Minimal JSONL dataset loader")
    parser.add_argument("--path", required=True, help="Path to JSONL file")
    parser.add_argument("--head", type=int, default=10, help="Preview first N rows")
    parser.add_argument("--as", dest="as_format", choices=["df", "list"], default="df",
                        help="Preview format: DataFrame (df) or list of dicts (list)")
    args = parser.parse_args(argv)

    records = load_jsonl(args.path)
    if args.as_format == "list":
        print("\n".join(json.dumps(r, ensure_ascii=False) for r in records[: args.head]))
    else:
        print(_preview_as_table(records, args.head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


