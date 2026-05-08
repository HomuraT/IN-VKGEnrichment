"""
Experience Bank - Empty stub for backward compatibility.

The TF-GRPO experience learning has been removed from the simplified pipeline.
This module provides empty implementations to maintain compatibility.
"""

from typing import Any, Dict, List


class ExperienceBank:
    """
    Empty experience bank (backward compatibility only).
    
    All methods return empty results.
    """
    
    def __init__(self) -> None:
        """Initialize empty bank."""
        self._items: List[Dict[str, Any]] = []
    
    def snapshot(self) -> List[Dict[str, Any]]:
        """Return empty snapshot."""
        return []
    
    def export_injection_text(
        self,
        max_items: int = 8,
        max_chars_per_item: int = 160,
        max_total_chars: int = 1200,
    ) -> str:
        """Return empty injection text."""
        return ""
    
    def add(self, text: str, tags: List[str] = None) -> str:
        """No-op add."""
        return ""
    
    def modify(self, item_id: str, text: str = None, tags: List[str] = None) -> bool:
        """No-op modify."""
        return False
    
    def delete(self, item_id: str) -> bool:
        """No-op delete."""
        return False
    
    def merge(self, ids: List[str], text: str, tags: List[str] = None) -> str:
        """No-op merge."""
        return ""
    
    def apply_ops(self, ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """No-op apply ops."""
        return []


__all__ = ["ExperienceBank"]
