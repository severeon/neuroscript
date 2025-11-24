"""BlockInterface Protocol and BlockCapability definition for NeuroScript blocks."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class DimensionConstraint:
    """Constraint on a dimension value."""

    constraint_str: str
    """The constraint expression (e.g., 'dim % 8 == 0', 'dim == input.x.shape[2]')"""


@dataclass
class ShapePattern:
    """Shape pattern definition for inputs/outputs."""

    pattern: List[str]
    """Shape pattern as list of strings (e.g., ['batch', 'seq', 'dim'] or ['*', 'hidden'])"""

    dtype: Optional[List[str]] = field(default_factory=lambda: None)
    """Optional list of allowed dtypes (e.g., ['float32', 'float16'])"""


@dataclass
class ParameterSpec:
    """Specification for a parameter."""

    name: str
    """Parameter name"""

    type: str
    """Parameter type (e.g., 'int', 'float', 'bool', 'string')"""

    required: bool = True
    """Whether this parameter is required"""

    default: Optional[Any] = None
    """Default value if not required"""

    range: Optional[Tuple[float, float]] = None
    """Valid range [min, max] for numeric types"""

    options: Optional[List[Any]] = None
    """Valid options for enum-like parameters"""


@dataclass
class BlockCapability:
    """Complete capability specification of a block."""

    name: str
    """Block name"""

    version: str
    """Version string"""

    inputs: Dict[str, ShapePattern] = field(default_factory=dict)
    """Input specifications mapping input name to shape pattern"""

    outputs: Dict[str, ShapePattern] = field(default_factory=dict)
    """Output specifications mapping output name to shape pattern"""

    params: Dict[str, ParameterSpec] = field(default_factory=dict)
    """Parameter specifications mapping param name to param spec"""

    constraints: List[DimensionConstraint] = field(default_factory=list)
    """Cross-dimensional constraints"""

    hardware_requirements: Optional[Dict[str, Any]] = None
    """Hardware requirements (e.g., {'cuda_capability': (7, 0)})"""


@runtime_checkable
class BlockInterface(Protocol):
    """Protocol defining the interface all neural network blocks must implement."""

    def get_capabilities(self) -> BlockCapability:
        """
        Return the capability specification of this block.

        Returns:
            BlockCapability: Block's input/output shapes, parameters, and constraints
        """
        ...

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass implementation.

        Args:
            *args: Positional arguments matching block inputs
            **kwargs: Keyword arguments for block execution

        Returns:
            Block output tensor or tuple of tensors
        """
        ...
