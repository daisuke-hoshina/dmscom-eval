"""Metric computations for DMSCOM segmentations."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from utils import TreeNode, convert_labels_to_tree


def compute_depth(root: TreeNode) -> int:
    """Compute maximum depth of the segmentation tree."""

    def _depth(node: TreeNode) -> int:
        if not node.children:
            return 0
        return 1 + max(_depth(child) for child in node.children)

    return _depth(root)


def compute_fragment_imbalance(root: TreeNode) -> float:
    """Average coefficient of variation of child durations across nodes."""

    def _collect(node: TreeNode) -> list[float]:
        values = []
        if len(node.children) >= 2:
            durations = np.array([child.duration() for child in node.children], dtype=float)
            if durations.mean() > 0:
                values.append(durations.std() / durations.mean())
        for child in node.children:
            values.extend(_collect(child))
        return values

    imbalances = _collect(root)
    return float(np.mean(imbalances)) if imbalances else 0.0


def compute_singleton_fragmentation(root: TreeNode) -> float:
    """Fraction of segments that occupy a single frame."""

    def _flatten(node: TreeNode) -> list[TreeNode]:
        nodes = [node]
        for child in node.children:
            nodes.extend(_flatten(child))
        return nodes

    nodes = _flatten(root)
    if not nodes:
        return 0.0
    singletons = sum(1 for n in nodes if n.duration() <= 1)
    return float(singletons) / len(nodes)


def compute_metrics(label_matrix: np.ndarray) -> Dict[str, float]:
    tree = convert_labels_to_tree(label_matrix)
    return {
        "depth": compute_depth(tree),
        "fragment_imbalance": compute_fragment_imbalance(tree),
        "singleton_fragmentation": compute_singleton_fragmentation(tree),
    }


__all__ = [
    "compute_depth",
    "compute_fragment_imbalance",
    "compute_singleton_fragmentation",
    "compute_metrics",
]
