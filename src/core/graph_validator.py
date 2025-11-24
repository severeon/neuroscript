"""GraphValidator for orchestrating complete architecture graph validation.

This module coordinates validation of neural architecture graphs including:
- Shape compatibility checking
- Constraint satisfaction
- Hardware compatibility
- Actionable error reporting

Implements Requirements 7.1, 7.2, 7.3, 7.4, 7.5 from specs/02-requirements.md
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import difflib

from .block_registry import BlockRegistry, BlockRegistryError
from .constraint_solver import ConstraintSolver, Configuration, ConstraintSolverError
from .graph_loader import ArchitectureGraph, GraphEdge, GraphNode
from .hardware_detector import HardwareDetector
from .shape_validator import (
    ShapeValidator,
    ShapePattern as ValidatorShapePattern,
    parse_shape_pattern
)
from .block_interface import ShapePattern as BlockShapePattern

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Validation error with actionable suggestions.

    Implements Req 7.1, 7.2, 7.3, 7.4

    Attributes:
        error_type: Type of error (shape, constraint, hardware, missing_block)
        message: Human-readable error description
        source_block: Source block identifier (for connection errors)
        target_block: Target block identifier (for connection errors)
        suggestions: List of actionable suggestions to fix the error
        technical_details: Optional technical details for debugging
    """

    error_type: str
    message: str
    source_block: Optional[str] = None
    target_block: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    technical_details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Format error for display."""
        lines = [f"✗ {self.error_type.upper()}: {self.message}"]

        if self.source_block or self.target_block:
            location = []
            if self.source_block:
                location.append(f"source: {self.source_block}")
            if self.target_block:
                location.append(f"target: {self.target_block}")
            lines.append(f"  Location: {', '.join(location)}")

        if self.suggestions:
            lines.append("\n  Suggested fixes:")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"    {i}. {suggestion}")

        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Complete validation result.

    Implements Req 7.1, 7.2, 7.3

    Attributes:
        valid: Whether the architecture is valid
        errors: List of validation errors (if any)
        warnings: List of validation warnings
        configurations: Valid dimension configurations (if any)
        hardware_compatible: Whether architecture is compatible with hardware
    """

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    configurations: List[Configuration] = field(default_factory=list)
    hardware_compatible: bool = True

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def __str__(self) -> str:
        """Format result for display."""
        lines = []

        if self.valid:
            lines.append("✓ Validation passed")
            if self.configurations:
                lines.append(f"\nValid configurations ({len(self.configurations)} found):")
                for i, config in enumerate(self.configurations[:5], 1):
                    lines.append(f"  {i}. {config}")
                if len(self.configurations) > 5:
                    lines.append(f"  ... and {len(self.configurations) - 5} more")
        else:
            lines.append("✗ Validation failed")
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(str(error))
                lines.append("")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)


class GraphValidator:
    """Orchestrates complete graph validation.

    Implements Requirements 7.1, 7.2, 7.3, 7.4, 7.5

    Coordinates validation across multiple subsystems:
    - ShapeValidator for tensor shape compatibility
    - ConstraintSolver for dimension configuration
    - HardwareDetector for resource compatibility

    Generates actionable error messages with clear suggestions.
    """

    def __init__(
        self,
        graph: ArchitectureGraph,
        registry: BlockRegistry,
        hardware: HardwareDetector,
        shape_validator: Optional[ShapeValidator] = None,
        constraint_solver: Optional[ConstraintSolver] = None
    ):
        """Initialize validator.

        Args:
            graph: Architecture graph to validate
            registry: Block registry for capability lookup
            hardware: Hardware detector for compatibility checking
            shape_validator: Optional shape validator (created if not provided)
            constraint_solver: Optional constraint solver (created if not provided)
        """
        self.graph = graph
        self.registry = registry
        self.hardware = hardware
        self.shape_validator = shape_validator or ShapeValidator()
        self.constraint_solver = constraint_solver or ConstraintSolver(graph, registry)

    def validate(self) -> ValidationResult:
        """Perform complete graph validation.

        Implements Req 7.1, 7.2, 7.3

        Validation steps:
        1. Validate block references exist
        2. Validate shape compatibility between connections
        3. Validate constraint satisfaction
        4. Check hardware compatibility
        5. Enumerate valid configurations

        Returns:
            ValidationResult with errors, warnings, and configurations
        """
        result = ValidationResult(valid=True)

        logger.info(f"Validating architecture with {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        # Step 1: Validate block references
        logger.debug("Step 1: Validating block references")
        self._validate_block_references(result)
        if not result.valid:
            return result

        # Step 2: Validate shape compatibility
        logger.debug("Step 2: Validating shape compatibility")
        self._validate_shapes(result)

        # Step 3: Validate constraints
        logger.debug("Step 3: Validating constraints")
        self._validate_constraints(result)

        # Step 4: Check hardware compatibility
        logger.debug("Step 4: Checking hardware compatibility")
        self._validate_hardware(result)

        # Step 5: Generate configurations (if validation passed so far)
        if result.valid:
            logger.debug("Step 5: Generating configurations")
            self._generate_configurations(result)

        logger.info(f"Validation {'passed' if result.valid else 'failed'} with {len(result.errors)} errors, {len(result.warnings)} warnings")

        return result

    def _validate_block_references(self, result: ValidationResult) -> None:
        """Validate that all block references exist in registry.

        Implements Req 7.4

        Args:
            result: ValidationResult to update with errors
        """
        for node in self.graph.nodes:
            if not self.registry.block_exists(node.block_id):
                error = ValidationError(
                    error_type="missing_block",
                    message=f"Block '{node.block_id}' not found in registry",
                    source_block=node.id,
                    suggestions=self._create_missing_block_suggestions(node.block_id)
                )
                result.add_error(error)

    def _validate_shapes(self, result: ValidationResult) -> None:
        """Validate shape compatibility across all edges.

        Implements Req 7.1

        Args:
            result: ValidationResult to update with errors
        """
        # Build node lookup
        node_map = {node.id: node for node in self.graph.nodes}

        for edge in self.graph.edges:
            # Skip if nodes don't exist (caught in reference validation)
            if edge.source not in node_map or edge.target not in node_map:
                continue

            source_node = node_map[edge.source]
            target_node = node_map[edge.target]

            # Get block capabilities
            try:
                source_cap = self.registry.get_block(source_node.block_id)
                target_cap = self.registry.get_block(target_node.block_id)
            except BlockRegistryError as e:
                logger.warning(f"Could not get block capabilities: {e}")
                continue

            # Get shape patterns
            source_output = edge.source_output
            target_input = edge.target_input

            if source_output not in source_cap.outputs:
                error = ValidationError(
                    error_type="missing_output",
                    message=f"Block '{source_node.block_id}' has no output named '{source_output}'",
                    source_block=source_node.id,
                    suggestions=[
                        f"Available outputs: {', '.join(source_cap.outputs.keys())}",
                        f"Check the edge definition: {edge.source}.{source_output} -> {edge.target}.{target_input}"
                    ]
                )
                result.add_error(error)
                continue

            if target_input not in target_cap.inputs:
                error = ValidationError(
                    error_type="missing_input",
                    message=f"Block '{target_node.block_id}' has no input named '{target_input}'",
                    target_block=target_node.id,
                    suggestions=[
                        f"Available inputs: {', '.join(target_cap.inputs.keys())}",
                        f"Check the edge definition: {edge.source}.{source_output} -> {edge.target}.{target_input}"
                    ]
                )
                result.add_error(error)
                continue

            # Unify shapes - convert BlockShapePattern to ValidatorShapePattern
            output_block_pattern = source_cap.outputs[source_output]
            input_block_pattern = target_cap.inputs[target_input]

            output_pattern = self._convert_shape_pattern(output_block_pattern)
            input_pattern = self._convert_shape_pattern(input_block_pattern)

            unify_result = self.shape_validator.unify(output_pattern, input_pattern)

            if not unify_result.success:
                error = self.create_shape_error(
                    source_node.id,
                    target_node.id,
                    input_pattern,
                    output_pattern,
                    unify_result.errors
                )
                result.add_error(error)

    def _validate_constraints(self, result: ValidationResult) -> None:
        """Validate constraint satisfaction.

        Implements Req 7.2

        Args:
            result: ValidationResult to update with errors
        """
        try:
            # Check if constraints are satisfiable
            satisfiable, unsatisfiable = self.constraint_solver.check_satisfiable()

            if not satisfiable:
                for constraint_expr in unsatisfiable:
                    error = self.create_constraint_error(
                        constraint_expr,
                        None,
                        []
                    )
                    result.add_error(error)

            # Check for conflicts
            conflicts = self.constraint_solver.detect_conflicts()
            for c1, c2 in conflicts:
                error = ValidationError(
                    error_type="constraint_conflict",
                    message=f"Conflicting constraints detected",
                    suggestions=[
                        f"Constraint 1: {c1}",
                        f"Constraint 2: {c2}",
                        "These constraints cannot both be satisfied. Consider:",
                        "  - Removing one of the conflicting constraints",
                        "  - Using different blocks with compatible requirements",
                        "  - Adjusting parameter ranges to allow overlap"
                    ]
                )
                result.add_error(error)
        except ConstraintSolverError as e:
            logger.warning(f"Error during constraint validation: {e}")
            error = ValidationError(
                error_type="constraint_error",
                message=f"Failed to validate constraints: {str(e)}",
                suggestions=[
                    "Check that all dimension constraints are valid",
                    "Verify parameter ranges are reasonable"
                ]
            )
            result.add_error(error)

    def _validate_hardware(self, result: ValidationResult) -> None:
        """Validate hardware compatibility.

        Implements Req 7.3

        Args:
            result: ValidationResult to update with errors/warnings
        """
        for node in self.graph.nodes:
            try:
                block_cap = self.registry.get_block(node.block_id)

                # Build hardware requirements from block
                requirements = {}

                # Check if block requires CUDA
                # This would come from block metadata in a real implementation
                # For now, we'll skip detailed hardware checks

                # Check compatibility
                compatible, warnings = self.hardware.is_block_compatible(requirements)

                if not compatible:
                    error = self._create_hardware_error(node, requirements)
                    result.add_error(error)
                    result.hardware_compatible = False

                for warning in warnings:
                    result.add_warning(f"Block '{node.id}': {warning}")

            except BlockRegistryError as e:
                logger.debug(f"Could not check hardware compatibility for {node.id}: {e}")

    def _generate_configurations(self, result: ValidationResult) -> None:
        """Generate valid configurations.

        Implements Req 7.1 (part of complete validation)

        Args:
            result: ValidationResult to populate with configurations
        """
        try:
            configs = self.constraint_solver.solve(max_configs=10)
            result.configurations = configs

            if not configs:
                error = ValidationError(
                    error_type="no_configurations",
                    message="No valid dimension configurations found",
                    suggestions=[
                        "Check that dimension constraints are not too restrictive",
                        "Verify that connected blocks have compatible dimension requirements",
                        "Consider relaxing parameter ranges or using different blocks"
                    ]
                )
                result.add_error(error)
        except ConstraintSolverError as e:
            logger.warning(f"Error generating configurations: {e}")
            error = ValidationError(
                error_type="configuration_error",
                message=f"Failed to generate configurations: {str(e)}",
                suggestions=[
                    "Check constraint syntax and validity",
                    "Verify all dimension references are defined"
                ]
            )
            result.add_error(error)

    def _convert_shape_pattern(self, block_pattern: BlockShapePattern) -> ValidatorShapePattern:
        """Convert BlockShapePattern to ValidatorShapePattern.

        Args:
            block_pattern: Shape pattern from block capability

        Returns:
            ValidatorShapePattern suitable for shape validation
        """
        dimensions = []
        is_wildcard = []

        for dim in block_pattern.pattern:
            dimensions.append(str(dim))
            is_wildcard.append(dim == "*")

        return ValidatorShapePattern(dimensions=dimensions, is_wildcard=is_wildcard)

    def create_shape_error(
        self,
        source: str,
        target: str,
        expected_shape: ValidatorShapePattern,
        actual_shape: ValidatorShapePattern,
        unify_errors: List[str]
    ) -> ValidationError:
        """Create shape mismatch error with suggestions.

        Implements Req 7.1, 7.5

        Args:
            source: Source node ID
            target: Target node ID
            expected_shape: Expected input shape pattern
            actual_shape: Actual output shape pattern
            unify_errors: Detailed unification errors

        Returns:
            ValidationError with actionable suggestions
        """
        # Build human-readable message
        message = (
            f"Shape mismatch between '{source}' output and '{target}' input\n"
            f"  Output shape: {actual_shape}\n"
            f"  Input requires: {expected_shape}"
        )

        # Build suggestions
        suggestions = []

        # Analyze the mismatch
        if len(actual_shape) != len(expected_shape):
            suggestions.append(
                f"Shapes have different ranks (dimensions): "
                f"{len(actual_shape)} vs {len(expected_shape)}. "
                f"Consider adding a reshape or projection block."
            )
        else:
            # Check each dimension
            for i, (out_dim, in_dim) in enumerate(zip(actual_shape.dimensions, expected_shape.dimensions)):
                if out_dim != in_dim and not (actual_shape.is_wildcard[i] or expected_shape.is_wildcard[i]):
                    # Try to parse as integers
                    try:
                        out_val = int(out_dim)
                        in_val = int(in_dim)
                        suggestions.append(
                            f"Dimension {i} mismatch: output is {out_val}, input requires {in_val}. "
                            f"Add a Linear({out_val}, {in_val}) adapter block between them."
                        )
                    except ValueError:
                        # Variable names
                        suggestions.append(
                            f"Dimension {i} uses different variable names: '{out_dim}' vs '{in_dim}'. "
                            f"Ensure these dimensions are bound to the same value or add a transformation block."
                        )

        # Generic suggestions
        suggestions.extend([
            f"Verify the block specifications for '{source}' and '{target}' are correct",
            "Consider using a different block with compatible shapes",
            "Add an intermediate adapter block to transform the shape"
        ])

        return ValidationError(
            error_type="shape_mismatch",
            message=message,
            source_block=source,
            target_block=target,
            suggestions=suggestions,
            technical_details={"unify_errors": unify_errors}
        )

    def create_constraint_error(
        self,
        constraint: str,
        violating_value: Optional[Any],
        valid_alternatives: List[Any]
    ) -> ValidationError:
        """Create constraint violation error.

        Implements Req 7.2, 7.5

        Args:
            constraint: Constraint expression that failed
            violating_value: Value that violated the constraint (if known)
            valid_alternatives: List of valid alternative values

        Returns:
            ValidationError with actionable suggestions
        """
        if violating_value is not None:
            message = f"Constraint violated: {constraint}\n  Attempted value: {violating_value}"
        else:
            message = f"Unsatisfiable constraint: {constraint}"

        suggestions = []

        if valid_alternatives:
            alternatives_str = ", ".join(str(v) for v in valid_alternatives[:5])
            suggestions.append(f"Try one of these valid values: {alternatives_str}")
            if len(valid_alternatives) > 5:
                suggestions.append(f"  ... and {len(valid_alternatives) - 5} more options")

        suggestions.extend([
            "Check if the constraint is too restrictive",
            "Verify that connected blocks have compatible constraints",
            "Consider using blocks with more flexible dimension requirements"
        ])

        return ValidationError(
            error_type="constraint_violation",
            message=message,
            suggestions=suggestions,
            technical_details={
                "constraint": constraint,
                "violating_value": violating_value,
                "valid_alternatives": valid_alternatives
            }
        )

    def suggest_similar_blocks(self, missing_block: str, max_suggestions: int = 5) -> List[str]:
        """Find blocks with similar names using fuzzy matching.

        Implements Req 7.4

        Uses Python's difflib for fuzzy string matching to find blocks
        with similar names to the missing block.

        Args:
            missing_block: Name of missing block
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of similar block names, sorted by similarity
        """
        available_blocks = self.registry.list_blocks()

        if not available_blocks:
            return []

        # Use difflib to find close matches
        # cutoff=0.3 means 30% similarity threshold (fairly lenient)
        close_matches = difflib.get_close_matches(
            missing_block,
            available_blocks,
            n=max_suggestions,
            cutoff=0.3
        )

        return close_matches

    def _create_missing_block_suggestions(self, missing_block: str) -> List[str]:
        """Create suggestions for a missing block.

        Implements Req 7.4, 7.5

        Args:
            missing_block: Name of missing block

        Returns:
            List of actionable suggestions
        """
        suggestions = []

        # Try to find similar blocks
        similar = self.suggest_similar_blocks(missing_block)

        if similar:
            suggestions.append(f"Did you mean one of these? {', '.join(similar)}")

        # List all available blocks
        available = self.registry.list_blocks()
        if available:
            if len(available) <= 10:
                suggestions.append(f"Available blocks: {', '.join(available)}")
            else:
                suggestions.append(f"Available blocks ({len(available)} total): {', '.join(available[:10])}, ...")
        else:
            suggestions.append("No blocks are currently registered. Check your blocks directory.")

        suggestions.extend([
            f"Check the spelling of '{missing_block}'",
            "Verify the block.yaml file exists in blocks/{name}/ directory",
            "Run block discovery to register new blocks"
        ])

        return suggestions

    def _create_hardware_error(
        self,
        node: GraphNode,
        requirements: Dict[str, Any]
    ) -> ValidationError:
        """Create hardware insufficiency error.

        Implements Req 7.3, 7.5

        Args:
            node: Graph node with incompatible hardware requirements
            requirements: Hardware requirements that failed

        Returns:
            ValidationError with actionable suggestions
        """
        message = f"Block '{node.id}' ({node.block_id}) is not compatible with available hardware"

        suggestions = []

        # Analyze what's missing
        if requirements.get('requires_cuda'):
            suggestions.append(
                "This block requires CUDA (GPU) but CUDA is not available. "
                "Run on a machine with GPU or use a CPU-compatible block."
            )

        if requirements.get('min_cuda_capability'):
            cap = requirements['min_cuda_capability']
            current = self.hardware.capabilities.cuda_compute_capability
            suggestions.append(
                f"Block requires CUDA compute capability {cap} "
                f"but only {current} is available. "
                "Upgrade GPU or use a block with lower compute requirements."
            )

        if requirements.get('estimated_memory_gb'):
            mem = requirements['estimated_memory_gb']
            available = (
                self.hardware.capabilities.gpu_memory_gb
                if self.hardware.capabilities.cuda_available
                else self.hardware.capabilities.system_memory_gb
            )
            suggestions.append(
                f"Block requires ~{mem:.2f}GB memory but only {available:.2f}GB available. "
                "Try reducing batch size, using smaller dimensions, or upgrading hardware."
            )

        # Generic suggestions
        suggestions.extend([
            "Use smaller model dimensions to reduce memory requirements",
            "Try a different block with lower hardware requirements",
            "Run on hardware with more resources"
        ])

        return ValidationError(
            error_type="hardware_incompatible",
            message=message,
            source_block=node.id,
            suggestions=suggestions,
            technical_details={"requirements": requirements}
        )
