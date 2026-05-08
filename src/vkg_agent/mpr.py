"""
Lightweight MPR module - Backward compatibility wrapper.

The complex multi-perspective rewriting logic has been removed.
This module now only provides compatibility classes and config.
"""

from dataclasses import dataclass
from typing import Any, Optional, Set


@dataclass
class MultiPerspectiveConfig:
    """
    Configuration for retrieval (kept for backward compatibility).
    
    Attributes:
        k1 (int): First round retrieval top-k (default 10)
        k2 (int): Second round retrieval top-k (default 10)
        n_rewrites (int): Number of rewrites (default 10) - no longer used
        topn_demonstrations (int): Max demonstrations per source (default 10)
    """
    k1: int = 10
    k2: int = 10
    n_rewrites: int = 10
    topn_demonstrations: int = 10


@dataclass
class BlockRetrievers:
    """
    Container for retrievers (kept for backward compatibility).
    """
    memory: Optional[Any] = None
    queries: Optional[Any] = None
    vkg_ontology: Optional[Any] = None
    vkg_mappings: Optional[Any] = None
    vkg_aggregated_triples: Optional[Any] = None


class MultiPerspectiveRewriter:
    """
    Lightweight MPR wrapper (backward compatibility only).
    
    The actual rewriting logic has been removed.
    Only provides skip_blocks support for run_experiment.py compatibility.
    """
    
    def __init__(self, llm: Optional[Any] = None, config: Optional[MultiPerspectiveConfig] = None) -> None:
        """Initialize with config."""
        self.config: MultiPerspectiveConfig = config or MultiPerspectiveConfig()
        self.llm = llm
        self.skip_blocks: Set[str] = set()
        self.sources = []
        self.experience_text: str = ""


__all__ = [
    "MultiPerspectiveConfig",
    "BlockRetrievers",
    "MultiPerspectiveRewriter",
]
