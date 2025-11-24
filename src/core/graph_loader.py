"""GraphLoader for loading and validating architecture YAML files."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from .block_registry import BlockRegistry, BlockRegistryError
from .block_interface import BlockCapability

logger = logging.getLogger(__name__)


class GraphLoaderError(Exception):
    """Raised when graph loading or validation fails."""

    pass


@dataclass
class GraphNode:
    """Represents a node in the computation graph."""

    id: str
    """Unique node identifier"""

    block_id: str
    """Reference to block identifier in registry"""

    params: Dict[str, Any] = field(default_factory=dict)
    """Instantiation parameters for this node"""

    label: Optional[str] = None
    """Optional human-readable label"""


@dataclass
class GraphEdge:
    """Represents a directed edge in the computation graph."""

    source: str
    """Source node ID"""

    target: str
    """Target node ID"""

    source_output: str = "y"
    """Source output name (default: 'y')"""

    target_input: str = "x"
    """Target input name (default: 'x')"""


@dataclass
class ArchitectureGraph:
    """Complete representation of a neural architecture."""

    nodes: List[GraphNode] = field(default_factory=list)
    """List of computation nodes"""

    edges: List[GraphEdge] = field(default_factory=list)
    """List of directed edges between nodes"""

    inputs: List[str] = field(default_factory=list)
    """Graph input node IDs"""

    outputs: List[str] = field(default_factory=list)
    """Graph output node IDs"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""


class GraphLoader:
    """
    Loader for architecture YAML files.

    Parses architecture definitions and validates them against registered blocks.
    """

    def __init__(self, registry: BlockRegistry):
        """
        Initialize GraphLoader with a block registry.

        Args:
            registry: BlockRegistry instance for block validation
        """
        self.registry = registry

    def load(self, file_path: str) -> ArchitectureGraph:
        """
        Load and parse an architecture YAML file.

        Args:
            file_path: Path to architecture.yaml file

        Returns:
            ArchitectureGraph: Parsed and validated architecture

        Raises:
            GraphLoaderError: If parsing or validation fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise GraphLoaderError(f"Architecture file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise GraphLoaderError(f"Malformed YAML in {file_path}: {e}")
        except Exception as e:
            raise GraphLoaderError(f"Error reading {file_path}: {e}")

        if not isinstance(data, dict):
            raise GraphLoaderError(
                f"Expected YAML dict in {file_path}, got {type(data)}"
            )

        return self._parse_architecture(data, file_path)

    def _parse_architecture(self, data: Dict[str, Any], file_path: Path) -> ArchitectureGraph:
        """
        Parse architecture data into ArchitectureGraph.

        Args:
            data: Parsed YAML data
            file_path: Path for error reporting

        Returns:
            ArchitectureGraph: Parsed architecture

        Raises:
            GraphLoaderError: If parsing fails
        """
        # Get components section
        components = data.get('components', {})
        if not isinstance(components, dict):
            raise GraphLoaderError(
                f"'components' must be a dict in {file_path}"
            )

        # Get topology section
        topology = data.get('topology', [])
        if isinstance(topology, str):
            topology = [topology]
        if not isinstance(topology, list):
            raise GraphLoaderError(
                f"'topology' must be a string or list in {file_path}"
            )

        # Parse nodes
        nodes = []
        for comp_id, comp_spec in components.items():
            if isinstance(comp_spec, str):
                # Shorthand: component_id: block_name
                block_id = comp_spec
                params = {}
            elif isinstance(comp_spec, dict):
                block_id = comp_spec.get('block')
                params = comp_spec.get('params', {})
                if not block_id:
                    raise GraphLoaderError(
                        f"Component '{comp_id}' missing 'block' field in {file_path}"
                    )
            else:
                raise GraphLoaderError(
                    f"Component '{comp_id}' must be string or dict in {file_path}"
                )

            nodes.append(GraphNode(id=comp_id, block_id=str(block_id), params=params))

        # Parse topology
        edges = []
        for topo_item in topology:
            parsed_edges = self.parse_topology(str(topo_item))
            edges.extend(parsed_edges)

        # Get input/output specifications
        inputs = data.get('inputs', [])
        outputs = data.get('outputs', [])

        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

        graph = ArchitectureGraph(
            nodes=nodes,
            edges=edges,
            inputs=inputs,
            outputs=outputs,
            metadata=data.get('metadata', {}),
        )

        # Validate
        self.validate_references(graph, file_path)
        self.validate_parameters(graph, file_path)
        self.detect_cycles(graph)

        return graph

    def parse_topology(self, topology_str: str) -> List[GraphEdge]:
        """
        Parse topology string into GraphEdge objects.

        Supports formats:
        - "A -> B" creates edge A (output y) -> B (input x)
        - "A.output -> B.input" specifies custom port names

        Args:
            topology_str: Topology specification string

        Returns:
            List of GraphEdge objects

        Raises:
            GraphLoaderError: If topology format is invalid
        """
        edges = []

        # Split by arrow separator
        if '->' not in topology_str:
            raise GraphLoaderError(
                f"Invalid topology format: '{topology_str}'. "
                f"Expected 'A -> B' or 'A.out -> B.in'"
            )

        # Handle chains like "A -> B -> C"
        parts = topology_str.split('->')
        parts = [p.strip() for p in parts]

        for i in range(len(parts) - 1):
            source_spec = parts[i]
            target_spec = parts[i + 1]

            # Parse source: "node_id" or "node_id.output"
            source_parts = source_spec.split('.')
            source_node = source_parts[0].strip()
            source_output = source_parts[1].strip() if len(source_parts) > 1 else "y"

            # Parse target: "node_id" or "node_id.input"
            target_parts = target_spec.split('.')
            target_node = target_parts[0].strip()
            target_input = target_parts[1].strip() if len(target_parts) > 1 else "x"

            if not source_node or not target_node:
                raise GraphLoaderError(
                    f"Invalid topology format: '{topology_str}'. "
                    f"Node names cannot be empty."
                )

            edges.append(
                GraphEdge(
                    source=source_node,
                    target=target_node,
                    source_output=source_output,
                    target_input=target_input,
                )
            )

        return edges

    def validate_references(self, graph: ArchitectureGraph, file_path: Optional[Path] = None) -> None:
        """
        Validate that all block references exist in registry.

        Args:
            graph: ArchitectureGraph to validate
            file_path: Path for error reporting

        Raises:
            GraphLoaderError: If block not found
        """
        for node in graph.nodes:
            if node.block_id not in self.registry:
                available = self.registry.list_blocks()
                msg = (
                    f"Block '{node.block_id}' referenced in component '{node.id}' "
                    f"not found in registry. "
                    f"Available blocks: {', '.join(available) if available else 'none'}"
                )
                if file_path:
                    msg += f" in {file_path}"
                raise GraphLoaderError(msg)

    def validate_parameters(self, graph: ArchitectureGraph, file_path: Optional[Path] = None) -> None:
        """
        Validate that component parameters match block specifications.

        Args:
            graph: ArchitectureGraph to validate
            file_path: Path for error reporting

        Raises:
            GraphLoaderError: If parameter validation fails
        """
        for node in graph.nodes:
            try:
                block_cap = self.registry.get_block(node.block_id)
            except BlockRegistryError:
                continue

            # Check for unknown parameters
            for param_name in node.params:
                if param_name not in block_cap.params:
                    msg = (
                        f"Unknown parameter '{param_name}' for block '{node.block_id}' "
                        f"in component '{node.id}'. "
                        f"Valid parameters: {', '.join(block_cap.params.keys())}"
                    )
                    if file_path:
                        msg += f" in {file_path}"
                    raise GraphLoaderError(msg)

            # Check required parameters are provided
            for param_name, param_spec in block_cap.params.items():
                if param_spec.required and param_name not in node.params:
                    msg = (
                        f"Missing required parameter '{param_name}' for block "
                        f"'{node.block_id}' in component '{node.id}'"
                    )
                    if file_path:
                        msg += f" in {file_path}"
                    raise GraphLoaderError(msg)

    def detect_cycles(self, graph: ArchitectureGraph) -> None:
        """
        Detect cycles in the computation graph using DFS.

        Args:
            graph: ArchitectureGraph to check

        Raises:
            GraphLoaderError: If cycle detected
        """
        # Build adjacency list
        adj: Dict[str, List[str]] = {}
        node_ids = {node.id for node in graph.nodes}

        for node in graph.nodes:
            adj[node.id] = []

        for edge in graph.edges:
            if edge.source in node_ids and edge.target in node_ids:
                adj[edge.source].append(edge.target)

        # DFS to detect cycles
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def has_cycle(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle_path = path[cycle_start:] + [neighbor]
                    raise GraphLoaderError(
                        f"Cycle detected in architecture: {' -> '.join(cycle_path)}"
                    )

            path.pop()
            rec_stack.discard(node)
            return False

        for node_id in adj:
            if node_id not in visited:
                try:
                    has_cycle(node_id, [])
                except GraphLoaderError:
                    raise

    def validate_graph_io(self, graph: ArchitectureGraph) -> None:
        """
        Validate that input/output nodes exist.

        Args:
            graph: ArchitectureGraph to validate

        Raises:
            GraphLoaderError: If inputs or outputs are invalid
        """
        node_ids = {node.id for node in graph.nodes}

        for inp in graph.inputs:
            if inp not in node_ids:
                raise GraphLoaderError(
                    f"Input node '{inp}' not found in components. "
                    f"Available: {', '.join(sorted(node_ids))}"
                )

        for out in graph.outputs:
            if out not in node_ids:
                raise GraphLoaderError(
                    f"Output node '{out}' not found in components. "
                    f"Available: {', '.join(sorted(node_ids))}"
                )
