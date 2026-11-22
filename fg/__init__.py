"""Frame-graph compiler package.

The public API is the eight entry points in :mod:`fg.graph`.
"""

from .graph import (
    add_pass,
    alias_memory,
    compile_batch,
    cull_graph,
    derive_state,
    insert_barriers,
    toposort,
    validate_graph,
)

__all__ = [
    "add_pass",
    "toposort",
    "insert_barriers",
    "alias_memory",
    "cull_graph",
    "derive_state",
    "validate_graph",
    "compile_batch",
]

__version__ = "0.1.0"
