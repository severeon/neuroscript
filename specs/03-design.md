# Design Document

## Overview

This document provides detailed specifications for each component in the NeuroScript v2 system. Components are designed for loose coupling, enabling parallel development of blocks and core validation engine.

## Design Principles

1. **Protocol-Based Contracts**: Use Python Protocols (PEP 544) for interfaces to avoid tight coupling
2. **Immutable Capabilities**: Block capabilities are parsed once and cached for performance
3. **Early Validation**: Catch errors at graph construction time, not runtime
4. **Hardware-Aware**: Filter incompatible blocks before validation to provide better UX
5. **Actionable Errors**: Every error includes suggested fixes

---

## Component Specifications

### Component: BlockInterface

**Purpose**: Defines the protocol contract that all blocks must implement

**Location**: `src/core/block_interface.py`

**Interface**:

```python
from typing import Protocol, Dict, Any
from dataclasses import dataclass

@dataclass
class BlockCapability:
    """Parsed capability specification"""
    inputs: Dict[str, Dict[str, Any]]
    outputs: Dict[str, Dict[str, Any]]
    params: Dict[str, Dict[str, Any]]
    constraints: List[str]
    compute: Dict[str, Any]  # params, flops, memory estimates

class BlockInterface(Protocol):
    """All blocks must implement this protocol

    Implements Req 10.1, 10.2, 10.3
    """

    @staticmethod
    def get_capabilities() -> BlockCapability:
        """Return this block's capability specification

        Implements Req 10.1
        """
        ...

    def __init__(self, **params):
        """Initialize block with validated parameters

        Implements Req 10.2
        """
        ...

    def forward(self, **inputs) -> Dict[str, Any]:
        """Execute block's forward pass

        Implements Req 10.3
        """
        ...
```

**Dependencies**: None (protocol definition only)

---

### Component: CapabilityParser

**Purpose**: Parses block.yaml capability specifications into structured constraint objects

**Location**: `src/core/capability_parser.py`

**Interface**:

```python
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class DimensionConstraint:
    """Represents a constraint on a dimension"""
    dimension: str
    constraint_type: str  # 'range', 'enum', 'expression'
    value: Any

@dataclass
class ParsedCapability:
    """Structured capability with extracted constraints"""
    name: str
    version: str
    inputs: Dict[str, ShapePattern]
    outputs: Dict[str, ShapePattern]
    params: Dict[str, ParamSpec]
    constraints: List[DimensionConstraint]
    compute: ComputeSpec

class CapabilityParser:
    """Parses block.yaml files into structured capabilities

    Implements Req 1.1, 1.2, 1.3, 1.4, 1.5
    """

    def parse_file(self, path: Path) -> ParsedCapability:
        """Parse a block.yaml file

        Implements Req 1.1, 1.5

        Raises:
            ValidationError: If YAML is malformed or missing fields
        """
        ...

    def extract_dimensions(self, shape_pattern: str) -> List[str]:
        """Extract dimension variables from shape pattern

        Implements Req 1.2

        Example:
            "[batch, seq, dim]" -> ["batch", "seq", "dim"]
        """
        ...

    def parse_constraint(self, constraint_str: str) -> DimensionConstraint:
        """Parse constraint expression into structured form

        Implements Req 1.3

        Example:
            "dim == input.x.shape[2]" -> DimensionConstraint(...)
        """
        ...

    def validate_param_range(self, param_spec: Dict, value: Any) -> bool:
        """Validate parameter value against range constraint

        Implements Req 1.4
        """
        ...
```

**Dependencies**:
- `yamale` for YAML schema validation
- `re` for pattern parsing

---

### Component: BlockRegistry

**Purpose**: Discovers, loads, and indexes available blocks from the filesystem

**Location**: `src/core/block_registry.py`

**Interface**:

```python
from pathlib import Path
from typing import Dict, Optional
from .capability_parser import ParsedCapability, CapabilityParser

class BlockRegistry:
    """Manages block discovery and registration

    Implements Req 2.1, 2.2, 2.3, 2.4, 2.5, 10.4, 10.5
    """

    def __init__(self, blocks_dir: Path, parser: CapabilityParser):
        """Initialize registry and discover blocks

        Implements Req 2.1
        """
        self.blocks_dir = blocks_dir
        self.parser = parser
        self._registry: Dict[str, ParsedCapability] = {}
        self._discover_blocks()

    def _discover_blocks(self) -> None:
        """Scan blocks/ directory for block.yaml files

        Implements Req 2.1, 2.2, 2.4, 2.5
        """
        ...

    def get_block(self, name: str) -> Optional[ParsedCapability]:
        """Retrieve block capability by name

        Implements Req 2.3
        """
        ...

    def list_blocks(self) -> List[str]:
        """Return list of all registered block names"""
        ...

    def validate_interface(self, block_impl: Any) -> bool:
        """Verify block implements BlockInterface protocol

        Implements Req 10.4, 10.5
        """
        ...
```

**Dependencies**:
- `CapabilityParser` for parsing block.yaml
- `BlockInterface` for interface validation

---

### Component: GraphLoader

**Purpose**: Loads architecture YAML files and constructs internal graph representation

**Location**: `src/core/graph_loader.py`

**Interface**:

```python
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class GraphNode:
    """Represents a block instance in the graph"""
    id: str
    block_type: str
    params: Dict[str, Any]

@dataclass
class GraphEdge:
    """Represents a connection between blocks"""
    source: str
    target: str
    source_output: str = "default"
    target_input: str = "default"

@dataclass
class ArchitectureGraph:
    """Internal graph representation"""
    name: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any]

class GraphLoader:
    """Loads and parses architecture YAML files

    Implements Req 3.1, 3.2, 3.3, 3.4, 3.5
    """

    def __init__(self, registry: BlockRegistry):
        self.registry = registry

    def load(self, path: Path) -> ArchitectureGraph:
        """Load architecture from YAML file

        Implements Req 3.1, 3.2

        Raises:
            ValidationError: If YAML is invalid
        """
        ...

    def parse_topology(self, topology_str: str) -> List[GraphEdge]:
        """Parse topology string into edges

        Implements Req 3.2

        Example:
            "input -> encoder -> decoder" -> [Edge(...), Edge(...)]
        """
        ...

    def validate_references(self, graph: ArchitectureGraph) -> None:
        """Ensure all block references exist

        Implements Req 3.3

        Raises:
            BlockNotFoundError: With list of available blocks
        """
        ...

    def validate_parameters(self, node: GraphNode) -> None:
        """Validate component parameters against block schema

        Implements Req 3.4
        """
        ...

    def detect_cycles(self, graph: ArchitectureGraph) -> Optional[List[str]]:
        """Detect cycles in graph topology

        Implements Req 3.5

        Returns:
            Cycle path if found, None otherwise
        """
        ...
```

**Dependencies**:
- `BlockRegistry` for block lookups
- `pyyaml` for YAML parsing

---

### Component: ShapeValidator

**Purpose**: Validates tensor shape compatibility between connected blocks using unification

**Location**: `src/core/shape_validator.py`

**Interface**:

```python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ShapePattern:
    """Represents a shape pattern with dimensions"""
    dimensions: List[str]  # e.g., ["batch", "seq", "dim"]
    is_wildcard: List[bool]  # Which dimensions are wildcards

@dataclass
class UnificationResult:
    """Result of shape unification"""
    success: bool
    bindings: Dict[str, int]  # dimension -> value
    errors: List[str]

class ShapeValidator:
    """Validates shape compatibility via unification

    Implements Req 4.1, 4.2, 4.3, 4.4, 4.5
    """

    def __init__(self):
        self.bindings: Dict[str, int] = {}  # Global dimension bindings

    def unify(
        self,
        output_pattern: ShapePattern,
        input_pattern: ShapePattern
    ) -> UnificationResult:
        """Unify two shape patterns

        Implements Req 4.1, 4.2, 4.3

        Returns:
            UnificationResult with success status and bindings
        """
        ...

    def match_dimension(
        self,
        dim1: str,
        dim2: str,
        is_wildcard1: bool,
        is_wildcard2: bool
    ) -> Tuple[bool, Optional[int]]:
        """Match two dimension specifications

        Implements Req 4.4

        Returns:
            (success, bound_value)
        """
        ...

    def validate_consistency(self, dim: str, value: int) -> bool:
        """Check if dimension binding is consistent with existing bindings

        Implements Req 4.5
        """
        ...
```

**Dependencies**:
- `ParsedCapability` for shape patterns

---

### Component: HardwareDetector

**Purpose**: Detects available hardware (CUDA capability, CPU cores, memory) and filters compatible blocks

**Location**: `src/core/hardware_detector.py`

**Interface**:

```python
from typing import Dict, Optional
from dataclasses import dataclass
import torch

@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities"""
    cuda_available: bool
    cuda_compute_capability: Optional[Tuple[int, int]]
    gpu_memory_gb: Optional[float]
    cpu_cores: int
    system_memory_gb: float

class HardwareDetector:
    """Detects hardware capabilities

    Implements Req 5.1, 5.2, 5.3, 5.4, 5.5
    """

    def __init__(self):
        self.capabilities = self._detect()

    def _detect(self) -> HardwareCapabilities:
        """Detect all hardware capabilities

        Implements Req 5.1, 5.2, 5.4
        """
        ...

    def check_cuda_available(self) -> bool:
        """Check if CUDA is available

        Implements Req 5.1
        """
        return torch.cuda.is_available()

    def get_compute_capability(self) -> Optional[Tuple[int, int]]:
        """Get CUDA compute capability

        Implements Req 5.2

        Returns:
            (major, minor) or None if no CUDA
        """
        if not self.check_cuda_available():
            return None
        return torch.cuda.get_device_capability(0)

    def is_block_compatible(
        self,
        block_requirements: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Check if block is compatible with hardware

        Implements Req 5.3, 5.5

        Returns:
            (compatible, warning_messages)
        """
        ...
```

**Dependencies**:
- `torch` for CUDA detection
- `psutil` for CPU/memory detection

---

### Component: ConstraintSolver

**Purpose**: Enumerates valid shape configurations satisfying all block constraints

**Location**: `src/core/constraint_solver.py`

**Interface**:

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Configuration:
    """A valid dimension configuration"""
    bindings: Dict[str, int]  # dimension -> value
    estimated_params: int
    estimated_memory_gb: float

class ConstraintSolver:
    """Solves constraint satisfaction problems for shapes

    Implements Req 6.1, 6.2, 6.3, 6.4, 6.5
    """

    def __init__(self, graph: ArchitectureGraph, registry: BlockRegistry):
        self.graph = graph
        self.registry = registry
        self.constraints: List[Constraint] = []

    def solve(self) -> List[Configuration]:
        """Find all valid configurations

        Implements Req 6.1, 6.2

        Returns:
            List of valid configurations with estimates
        """
        ...

    def check_satisfiable(self) -> Tuple[bool, List[str]]:
        """Check if constraints are satisfiable

        Implements Req 6.3

        Returns:
            (satisfiable, unsatisfiable_constraints)
        """
        ...

    def apply_configuration(self, config: Configuration) -> None:
        """Bind all dimensions to concrete values

        Implements Req 6.4
        """
        ...

    def detect_conflicts(self) -> List[Tuple[str, str]]:
        """Detect conflicting constraints

        Implements Req 6.5

        Returns:
            List of (constraint1, constraint2) conflicts
        """
        ...
```

**Dependencies**:
- `ArchitectureGraph` for constraint collection
- `BlockRegistry` for block capabilities

---

### Component: GraphValidator

**Purpose**: Orchestrates validation of entire architecture graph (topology, shapes, constraints)

**Location**: `src/core/graph_validator.py`

**Interface**:

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ValidationError:
    """Validation error with suggestions"""
    error_type: str
    message: str
    source_block: Optional[str]
    target_block: Optional[str]
    suggestions: List[str]

@dataclass
class ValidationResult:
    """Complete validation result"""
    valid: bool
    errors: List[ValidationError]
    configurations: List[Configuration]

class GraphValidator:
    """Orchestrates complete graph validation

    Implements Req 7.1, 7.2, 7.3, 7.4, 7.5
    """

    def __init__(
        self,
        graph: ArchitectureGraph,
        registry: BlockRegistry,
        hardware: HardwareDetector,
        shape_validator: ShapeValidator,
        constraint_solver: ConstraintSolver
    ):
        self.graph = graph
        self.registry = registry
        self.hardware = hardware
        self.shape_validator = shape_validator
        self.constraint_solver = constraint_solver

    def validate(self) -> ValidationResult:
        """Perform complete graph validation

        Implements Req 7.1, 7.2, 7.3
        """
        ...

    def create_shape_error(
        self,
        source: str,
        target: str,
        expected_shape: ShapePattern,
        actual_shape: ShapePattern
    ) -> ValidationError:
        """Create shape mismatch error with suggestions

        Implements Req 7.1
        """
        ...

    def create_constraint_error(
        self,
        constraint: str,
        violating_value: Any,
        valid_alternatives: List[Any]
    ) -> ValidationError:
        """Create constraint violation error

        Implements Req 7.2
        """
        ...

    def suggest_similar_blocks(self, missing_block: str) -> List[str]:
        """Find blocks with similar names

        Implements Req 7.4
        """
        ...
```

**Dependencies**:
- All other validation components

---

### Component: CompilationEngine

**Purpose**: Generates executable PyTorch code from validated architecture graphs

**Location**: `src/core/compilation_engine.py`

**Interface**:

```python
from pathlib import Path
from typing import Dict, Any

class CompilationEngine:
    """Generates PyTorch code from validated graphs

    Implements Req 8.1, 8.2, 8.3, 8.4, 8.5
    """

    def __init__(
        self,
        graph: ArchitectureGraph,
        config: Configuration,
        registry: BlockRegistry
    ):
        self.graph = graph
        self.config = config
        self.registry = registry

    def compile(self, output_path: Path) -> None:
        """Generate PyTorch module file

        Implements Req 8.1, 8.2
        """
        ...

    def generate_sequential(self, nodes: List[GraphNode]) -> str:
        """Generate nn.Sequential for linear chains

        Implements Req 8.3
        """
        ...

    def generate_parallel(
        self,
        branches: List[List[GraphNode]],
        merge_op: str
    ) -> str:
        """Generate code for parallel execution

        Implements Req 8.4
        """
        ...

    def generate_shape_assertions(self, node: GraphNode) -> str:
        """Generate runtime shape assertions

        Implements Req 8.5
        """
        ...
```

**Dependencies**:
- `jinja2` for code templating
- `black` for code formatting

---

### Component: ContainerRuntime

**Purpose**: Executes generated models in Docker containers with resource limits

**Location**: `src/core/container_runtime.py`

**Interface**:

```python
from pathlib import Path
from typing import Dict, Any, Tuple
import docker

@dataclass
class ExecutionResult:
    """Result of container execution"""
    exit_code: int
    stdout: str
    stderr: str
    metrics: Dict[str, Any]

class ContainerRuntime:
    """Manages Docker container execution

    Implements Req 9.1, 9.2, 9.3, 9.4, 9.5
    """

    def __init__(
        self,
        hardware: HardwareCapabilities,
        image_name: str = "neuroscript-runtime:latest"
    ):
        self.hardware = hardware
        self.image_name = image_name
        self.client = docker.from_env()

    def execute(
        self,
        code_path: Path,
        args: List[str]
    ) -> ExecutionResult:
        """Execute model in container

        Implements Req 9.1, 9.5
        """
        ...

    def create_container(self, code_path: Path) -> docker.Container:
        """Create container with resource limits

        Implements Req 9.2, 9.3
        """
        ...

    def get_resource_limits(self) -> Dict[str, Any]:
        """Calculate resource limits from hardware

        Implements Req 9.2
        """
        ...

    def handle_oom(self, container: docker.Container) -> ExecutionResult:
        """Handle out-of-memory termination

        Implements Req 9.4
        """
        ...
```

**Dependencies**:
- `docker` Python SDK
- `HardwareCapabilities` for resource limits

---

## Example Block Implementation

### Block: Linear

**Location**: `blocks/linear/`

**Files**:
- `block.yaml` - Capability specification
- `module.py` - PyTorch implementation
- `test.py` - Unit tests

**block.yaml**:
```yaml
name: Linear
version: 1.0.0

capabilities:
  inputs:
    x:
      type: Tensor
      dtype: [float32, float16, bfloat16]
      shape: ["*", "in_features"]

  outputs:
    y:
      type: Tensor
      dtype: input.x.dtype
      shape: ["*", "out_features"]

  params:
    in_features:
      type: int
      range: [1, 100000]
      required: true
    out_features:
      type: int
      range: [1, 100000]
      required: true
    bias:
      type: bool
      default: true

  compute:
    params: in_features * out_features + (out_features if bias else 0)
    flops: 2 * in_features * out_features
    memory: in_features * out_features * 4  # float32 bytes
```

**module.py**:
```python
import torch
import torch.nn as nn
from ..core.block_interface import BlockInterface, BlockCapability

class Linear(nn.Module):
    """Linear transformation block"""

    @staticmethod
    def get_capabilities() -> BlockCapability:
        # Load from block.yaml
        ...

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
```

---

## Design Summary

- **10 Core Components** implementing the validation pipeline
- **Protocol-Based Contracts** for loose coupling
- **Hardware-Aware Validation** for better UX
- **Comprehensive Error Reporting** with actionable suggestions
- **Container-Based Execution** for resource safety
