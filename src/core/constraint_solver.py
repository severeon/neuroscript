"""ConstraintSolver for enumerating valid shape configurations."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .block_interface import BlockCapability, DimensionConstraint
from .block_registry import BlockRegistry
from .graph_loader import ArchitectureGraph, GraphNode

logger = logging.getLogger(__name__)


class ConstraintSolverError(Exception):
    """Raised when constraint solving fails."""

    pass


@dataclass
class Configuration:
    """A valid dimension configuration with resource estimates."""

    bindings: Dict[str, int] = field(default_factory=dict)
    """Mapping of dimension names to concrete values"""

    estimated_params: int = 0
    """Estimated total parameter count"""

    estimated_memory_gb: float = 0.0
    """Estimated memory usage in GB"""

    def __repr__(self) -> str:
        """String representation of configuration."""
        bindings_str = ", ".join(f"{k}={v}" for k, v in sorted(self.bindings.items()))
        return (
            f"Configuration({bindings_str}, "
            f"params={self.estimated_params:,}, "
            f"memory={self.estimated_memory_gb:.2f}GB)"
        )


@dataclass
class Constraint:
    """Internal representation of a parsed constraint."""

    expression: str
    """Original constraint expression"""

    dimension: Optional[str] = None
    """Primary dimension involved (if single variable)"""

    constraint_type: str = "expression"
    """Type: 'enum', 'range', 'modulo', 'equality', 'expression'"""

    values: Optional[List[int]] = None
    """Valid values for 'enum' constraints"""

    min_value: Optional[int] = None
    """Minimum value for 'range' constraints"""

    max_value: Optional[int] = None
    """Maximum value for 'range' constraints"""

    modulo: Optional[Tuple[int, int]] = None
    """(divisor, remainder) for modulo constraints"""

    related_dimensions: Set[str] = field(default_factory=set)
    """All dimensions referenced in the constraint"""


class ConstraintSolver:
    """
    Solves constraint satisfaction problems for dimension configurations.

    Implements Req 6.1, 6.2, 6.3, 6.4, 6.5
    """

    # Regex patterns for constraint parsing
    ENUM_PATTERN = re.compile(r'(\w+)\s+in\s+\[([^\]]+)\]')
    RANGE_PATTERN = re.compile(r'(\w+)\s*(>=|>|<=|<)\s*(\d+)')
    MODULO_PATTERN = re.compile(r'(\w+)\s*%\s*(\d+)\s*==\s*(\d+)')
    EQUALITY_PATTERN = re.compile(r'(\w+)\s*==\s*(\d+)')
    DIMENSION_PATTERN = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')

    def __init__(self, graph: ArchitectureGraph, registry: BlockRegistry):
        """
        Initialize ConstraintSolver.

        Args:
            graph: Architecture graph to solve constraints for
            registry: Block registry for accessing block capabilities
        """
        self.graph = graph
        self.registry = registry
        self.constraints: List[Constraint] = []
        self.dimensions: Set[str] = set()
        self.dimension_domains: Dict[str, Set[int]] = {}
        self._collected = False

    def solve(self, max_configs: int = 100) -> List[Configuration]:
        """
        Find all valid dimension configurations.

        Implements Req 6.1, 6.2

        Args:
            max_configs: Maximum number of configurations to return

        Returns:
            List of valid configurations with resource estimates

        Raises:
            ConstraintSolverError: If constraint collection or solving fails
        """
        # Collect constraints if not already done
        if not self._collected:
            self._collect_constraints()
            self._collected = True

        # Check if satisfiable
        satisfiable, unsatisfiable = self.check_satisfiable()
        if not satisfiable:
            raise ConstraintSolverError(
                f"Unsatisfiable constraints: {', '.join(unsatisfiable)}"
            )

        # Initialize dimension domains
        self._initialize_domains()

        # Apply constraint propagation
        self._propagate_constraints()

        # Generate configurations
        configs = self._enumerate_configurations(max_configs)

        # Estimate resources for each configuration
        for config in configs:
            self._estimate_resources(config)

        logger.info(f"Found {len(configs)} valid configuration(s)")
        return configs

    def check_satisfiable(self) -> Tuple[bool, List[str]]:
        """
        Check if constraints are satisfiable.

        Implements Req 6.3

        Returns:
            Tuple of (satisfiable, list of unsatisfiable constraint expressions)
        """
        if not self._collected:
            self._collect_constraints()
            self._collected = True

        unsatisfiable = []

        # Check for obvious contradictions
        for constraint in self.constraints:
            if constraint.constraint_type == "enum":
                if constraint.values is not None and len(constraint.values) == 0:
                    unsatisfiable.append(constraint.expression)
            elif constraint.constraint_type == "range":
                if (
                    constraint.min_value is not None
                    and constraint.max_value is not None
                    and constraint.min_value > constraint.max_value
                ):
                    unsatisfiable.append(constraint.expression)

        # Check for conflicting constraints on same dimension
        conflicts = self.detect_conflicts()
        for c1_expr, c2_expr in conflicts:
            if c1_expr not in unsatisfiable:
                unsatisfiable.append(f"{c1_expr} conflicts with {c2_expr}")

        return (len(unsatisfiable) == 0, unsatisfiable)

    def apply_configuration(self, config: Configuration) -> None:
        """
        Bind all dimensions to concrete values from configuration.

        Implements Req 6.4

        Args:
            config: Configuration with dimension bindings
        """
        # Apply bindings to dimension domains
        for dim, value in config.bindings.items():
            if dim in self.dimension_domains:
                self.dimension_domains[dim] = {value}
            else:
                self.dimension_domains[dim] = {value}

        logger.info(f"Applied configuration: {config}")

    def detect_conflicts(self) -> List[Tuple[str, str]]:
        """
        Detect conflicting constraints.

        Implements Req 6.5

        Returns:
            List of (constraint1_expr, constraint2_expr) tuples representing conflicts
        """
        conflicts = []

        # Group constraints by dimension
        dim_constraints: Dict[str, List[Constraint]] = {}
        for constraint in self.constraints:
            if constraint.dimension:
                if constraint.dimension not in dim_constraints:
                    dim_constraints[constraint.dimension] = []
                dim_constraints[constraint.dimension].append(constraint)

        # Check for conflicts within each dimension
        for dim, constraints_list in dim_constraints.items():
            for i, c1 in enumerate(constraints_list):
                for c2 in constraints_list[i + 1 :]:
                    if self._constraints_conflict(c1, c2):
                        conflicts.append((c1.expression, c2.expression))

        return conflicts

    def _collect_constraints(self) -> None:
        """
        Collect all constraints from graph nodes and blocks.

        Populates self.constraints and self.dimensions
        """
        # Collect dimensions and constraints from each node
        for node in self.graph.nodes:
            try:
                block_cap = self.registry.get_block(node.block_id)
                self._collect_from_block(block_cap, node)
            except Exception as e:
                logger.warning(
                    f"Failed to get block capability for {node.block_id}: {e}"
                )
                continue

    def _collect_from_block(self, block_cap: BlockCapability, node: GraphNode) -> None:
        """
        Collect dimensions and constraints from a block capability.

        Args:
            block_cap: Block capability specification
            node: Graph node instance
        """
        # Collect dimensions from inputs
        for input_name, shape_pattern in block_cap.inputs.items():
            for dim in shape_pattern.pattern:
                if dim != "*" and not dim.isdigit():
                    self.dimensions.add(dim)

        # Collect dimensions from outputs
        for output_name, shape_pattern in block_cap.outputs.items():
            for dim in shape_pattern.pattern:
                if dim != "*" and not dim.isdigit():
                    self.dimensions.add(dim)

        # Collect constraints from block
        for constraint_obj in block_cap.constraints:
            constraint_str = constraint_obj.constraint_str
            parsed = self._parse_constraint(constraint_str)
            self.constraints.append(parsed)

        # Collect constraints from parameter ranges
        for param_name, param_spec in block_cap.params.items():
            if param_name in node.params:
                # Parameter is bound to a specific value
                continue

            if param_spec.range is not None:
                # Create range constraint for parameter
                min_val, max_val = param_spec.range
                constraint_str = f"{param_name} >= {int(min_val)} and {param_name} <= {int(max_val)}"
                parsed = self._parse_constraint(constraint_str)
                self.constraints.append(parsed)

            if param_spec.options is not None:
                # Create enum constraint for parameter
                options_str = ", ".join(str(opt) for opt in param_spec.options)
                constraint_str = f"{param_name} in [{options_str}]"
                parsed = self._parse_constraint(constraint_str)
                self.constraints.append(parsed)

    def _parse_constraint(self, constraint_str: str) -> Constraint:
        """
        Parse a constraint string into a Constraint object.

        Args:
            constraint_str: Constraint expression string

        Returns:
            Parsed Constraint object
        """
        constraint_str = constraint_str.strip()

        # Extract all dimensions referenced
        dimensions = set(self.DIMENSION_PATTERN.findall(constraint_str))
        # Filter out keywords
        keywords = {"in", "and", "or", "not"}
        dimensions = {d for d in dimensions if d not in keywords}

        # Try to parse as enum constraint: "dim in [128, 256, 512]"
        match = self.ENUM_PATTERN.search(constraint_str)
        if match:
            dim = match.group(1)
            values_str = match.group(2)
            values = [int(v.strip()) for v in values_str.split(",")]
            return Constraint(
                expression=constraint_str,
                dimension=dim,
                constraint_type="enum",
                values=values,
                related_dimensions=dimensions,
            )

        # Try to parse as modulo constraint: "dim % 8 == 0"
        match = self.MODULO_PATTERN.search(constraint_str)
        if match:
            dim = match.group(1)
            divisor = int(match.group(2))
            remainder = int(match.group(3))
            return Constraint(
                expression=constraint_str,
                dimension=dim,
                constraint_type="modulo",
                modulo=(divisor, remainder),
                related_dimensions=dimensions,
            )

        # Try to parse as equality constraint: "dim == 512"
        match = self.EQUALITY_PATTERN.search(constraint_str)
        if match:
            dim = match.group(1)
            value = int(match.group(2))
            return Constraint(
                expression=constraint_str,
                dimension=dim,
                constraint_type="equality",
                values=[value],
                related_dimensions=dimensions,
            )

        # Try to parse as range constraint: "dim >= 1"
        match = self.RANGE_PATTERN.search(constraint_str)
        if match:
            dim = match.group(1)
            operator = match.group(2)
            value = int(match.group(3))

            min_val = None
            max_val = None

            if operator in (">=", ">"):
                min_val = value if operator == ">=" else value + 1
            elif operator in ("<=", "<"):
                max_val = value if operator == "<=" else value - 1

            return Constraint(
                expression=constraint_str,
                dimension=dim,
                constraint_type="range",
                min_value=min_val,
                max_value=max_val,
                related_dimensions=dimensions,
            )

        # Complex expression - store as-is
        return Constraint(
            expression=constraint_str,
            constraint_type="expression",
            related_dimensions=dimensions,
        )

    def _initialize_domains(self) -> None:
        """
        Initialize dimension domains with reasonable default values.

        Creates initial domain sets for each dimension.
        """
        # Common dimension values for neural networks
        default_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 2048, 4096]

        for dim in self.dimensions:
            # Start with default values
            self.dimension_domains[dim] = set(default_values)

    def _propagate_constraints(self) -> None:
        """
        Apply constraint propagation to reduce dimension domains.

        Uses arc consistency to narrow down possible values.
        """
        changed = True
        max_iterations = 100
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for constraint in self.constraints:
                if constraint.constraint_type == "enum" and constraint.dimension:
                    # Reduce domain to enumerated values
                    if constraint.values is not None:
                        old_domain = self.dimension_domains.get(constraint.dimension, set())
                        new_domain = old_domain & set(constraint.values)
                        if new_domain != old_domain:
                            self.dimension_domains[constraint.dimension] = new_domain
                            changed = True

                elif constraint.constraint_type == "equality" and constraint.dimension:
                    # Reduce domain to single value
                    if constraint.values is not None:
                        old_domain = self.dimension_domains.get(constraint.dimension, set())
                        new_domain = old_domain & set(constraint.values)
                        if new_domain != old_domain:
                            self.dimension_domains[constraint.dimension] = new_domain
                            changed = True

                elif constraint.constraint_type == "range" and constraint.dimension:
                    # Reduce domain to range
                    old_domain = self.dimension_domains.get(constraint.dimension, set())
                    new_domain = old_domain.copy()

                    if constraint.min_value is not None:
                        new_domain = {v for v in new_domain if v >= constraint.min_value}

                    if constraint.max_value is not None:
                        new_domain = {v for v in new_domain if v <= constraint.max_value}

                    if new_domain != old_domain:
                        self.dimension_domains[constraint.dimension] = new_domain
                        changed = True

                elif constraint.constraint_type == "modulo" and constraint.dimension:
                    # Filter by modulo constraint
                    if constraint.modulo is not None:
                        divisor, remainder = constraint.modulo
                        old_domain = self.dimension_domains.get(constraint.dimension, set())
                        new_domain = {v for v in old_domain if v % divisor == remainder}
                        if new_domain != old_domain:
                            self.dimension_domains[constraint.dimension] = new_domain
                            changed = True

        logger.debug(f"Constraint propagation completed in {iteration} iterations")

    def _enumerate_configurations(self, max_configs: int) -> List[Configuration]:
        """
        Enumerate valid configurations from dimension domains.

        Args:
            max_configs: Maximum number of configurations to generate

        Returns:
            List of valid configurations
        """
        if not self.dimensions:
            # No dimensions to configure
            return [Configuration()]

        # Sort dimensions for consistent ordering
        sorted_dims = sorted(self.dimensions)

        # Generate configurations using backtracking
        configs = []

        def backtrack(index: int, current_bindings: Dict[str, int]) -> None:
            if index == len(sorted_dims):
                # Check if configuration satisfies all constraints
                if self._verify_configuration(current_bindings):
                    configs.append(Configuration(bindings=current_bindings.copy()))
                return

            if len(configs) >= max_configs:
                return

            dim = sorted_dims[index]
            domain = self.dimension_domains.get(dim, set())

            for value in sorted(domain):
                current_bindings[dim] = value
                backtrack(index + 1, current_bindings)

                if len(configs) >= max_configs:
                    return

        backtrack(0, {})
        return configs

    def _verify_configuration(self, bindings: Dict[str, int]) -> bool:
        """
        Verify that a configuration satisfies all constraints.

        Args:
            bindings: Dimension to value mapping

        Returns:
            True if configuration is valid
        """
        for constraint in self.constraints:
            if not self._check_constraint(constraint, bindings):
                return False
        return True

    def _check_constraint(self, constraint: Constraint, bindings: Dict[str, int]) -> bool:
        """
        Check if a constraint is satisfied by given bindings.

        Args:
            constraint: Constraint to check
            bindings: Dimension to value mapping

        Returns:
            True if constraint is satisfied
        """
        if constraint.constraint_type == "enum":
            if constraint.dimension and constraint.dimension in bindings:
                value = bindings[constraint.dimension]
                return constraint.values is not None and value in constraint.values
            return True

        elif constraint.constraint_type == "equality":
            if constraint.dimension and constraint.dimension in bindings:
                value = bindings[constraint.dimension]
                return constraint.values is not None and value in constraint.values
            return True

        elif constraint.constraint_type == "range":
            if constraint.dimension and constraint.dimension in bindings:
                value = bindings[constraint.dimension]
                if constraint.min_value is not None and value < constraint.min_value:
                    return False
                if constraint.max_value is not None and value > constraint.max_value:
                    return False
            return True

        elif constraint.constraint_type == "modulo":
            if constraint.dimension and constraint.dimension in bindings:
                value = bindings[constraint.dimension]
                if constraint.modulo is not None:
                    divisor, remainder = constraint.modulo
                    return value % divisor == remainder
            return True

        elif constraint.constraint_type == "expression":
            # For complex expressions, try to evaluate with bindings
            try:
                # Create a safe evaluation environment
                env = bindings.copy()
                # Simple evaluation - this could be enhanced with a proper expression evaluator
                result = eval(constraint.expression, {"__builtins__": {}}, env)
                return bool(result)
            except Exception as e:
                logger.debug(f"Could not evaluate constraint '{constraint.expression}': {e}")
                # If we can't evaluate, assume it's satisfied
                return True

        return True

    def _estimate_resources(self, config: Configuration) -> None:
        """
        Estimate parameter count and memory usage for a configuration.

        Implements Req 6.2 (part of returning configurations with estimates)

        Args:
            config: Configuration to estimate resources for
        """
        total_params = 0
        total_memory = 0.0

        for node in self.graph.nodes:
            try:
                block_cap = self.registry.get_block(node.block_id)

                # Get parameter count if specified
                # Note: This is a simplified estimation
                # Real implementation would need to evaluate compute expressions

                # Estimate based on dimensions
                # Common patterns: Linear layers = in_features * out_features
                # Embeddings = vocab_size * embed_dim

                # For now, use a heuristic
                node_params = 0
                for dim_name, dim_value in config.bindings.items():
                    # Simple heuristic: accumulate dimension products
                    if dim_name in block_cap.params:
                        param_spec = block_cap.params[dim_name]
                        if param_spec.type in ("int", "float"):
                            node_params += dim_value

                total_params += node_params

                # Estimate memory (4 bytes per float32 parameter)
                total_memory += node_params * 4 / (1024**3)  # Convert to GB

            except Exception as e:
                logger.debug(f"Could not estimate resources for {node.block_id}: {e}")
                continue

        config.estimated_params = total_params
        config.estimated_memory_gb = total_memory

    def _constraints_conflict(self, c1: Constraint, c2: Constraint) -> bool:
        """
        Check if two constraints conflict.

        Args:
            c1: First constraint
            c2: Second constraint

        Returns:
            True if constraints conflict
        """
        # Only check conflicts for same dimension
        if c1.dimension != c2.dimension or not c1.dimension:
            return False

        # Check enum constraints
        if c1.constraint_type == "enum" and c2.constraint_type == "enum":
            if c1.values is not None and c2.values is not None:
                intersection = set(c1.values) & set(c2.values)
                return len(intersection) == 0

        # Check equality constraints
        if c1.constraint_type == "equality" and c2.constraint_type == "equality":
            if c1.values is not None and c2.values is not None:
                return c1.values != c2.values

        # Check range constraints
        if c1.constraint_type == "range" and c2.constraint_type == "range":
            min1 = c1.min_value if c1.min_value is not None else float("-inf")
            max1 = c1.max_value if c1.max_value is not None else float("inf")
            min2 = c2.min_value if c2.min_value is not None else float("-inf")
            max2 = c2.max_value if c2.max_value is not None else float("inf")

            # Ranges conflict if they don't overlap
            return min1 > max2 or min2 > max1

        return False
