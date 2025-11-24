"""NeuroScript v2 Core Components"""

from .block_interface import BlockInterface, BlockCapability
from .capability_parser import CapabilityParser
from .block_registry import BlockRegistry
from .graph_loader import GraphLoader, ArchitectureGraph, GraphNode, GraphEdge
from .hardware_detector import HardwareDetector, HardwareCapabilities
from .constraint_solver import ConstraintSolver, Configuration
from .shape_validator import (
    ShapeValidator,
    ShapePattern,
    UnificationResult,
    parse_shape_pattern,
)
from .compilation_engine import CompilationEngine, CompilationError

__all__ = [
    "BlockInterface",
    "BlockCapability",
    "CapabilityParser",
    "BlockRegistry",
    "GraphLoader",
    "ArchitectureGraph",
    "GraphNode",
    "GraphEdge",
    "HardwareDetector",
    "HardwareCapabilities",
    "ConstraintSolver",
    "Configuration",
    "ShapeValidator",
    "ShapePattern",
    "UnificationResult",
    "parse_shape_pattern",
    "CompilationEngine",
    "CompilationError",
]
