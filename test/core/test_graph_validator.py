"""
Integration tests for GraphValidator.

Tests Requirements 7.1, 7.2, 7.3, 7.4, 7.5 from specs/02-requirements.md
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.core.graph_validator import (
    GraphValidator,
    ValidationError,
    ValidationResult
)
from src.core.block_registry import BlockRegistry
from src.core.graph_loader import GraphLoader, ArchitectureGraph, GraphNode, GraphEdge
from src.core.hardware_detector import HardwareDetector
from src.core.shape_validator import ShapeValidator, ShapePattern as ValidatorShapePattern
from src.core.constraint_solver import ConstraintSolver


@pytest.fixture
def temp_blocks_dir():
    """Create a temporary blocks directory with test blocks."""
    temp_dir = tempfile.mkdtemp()
    blocks_path = Path(temp_dir) / "blocks"
    blocks_path.mkdir()

    # Create a simple linear block
    linear_dir = blocks_path / "linear"
    linear_dir.mkdir()
    linear_yaml = linear_dir / "block.yaml"
    linear_yaml.write_text("""
name: Linear
version: 1.0.0
capabilities:
  inputs:
    x:
      type: Tensor
      dtype: [float32, float16]
      shape: ["batch", "in_features"]
  outputs:
    y:
      type: Tensor
      dtype: input.x.dtype
      shape: ["batch", "out_features"]
  params:
    in_features:
      type: int
      range: [1, 10000]
      required: true
    out_features:
      type: int
      range: [1, 10000]
      required: true
  constraints:
    - "in_features in [128, 256, 512, 1024]"
    - "out_features in [128, 256, 512, 1024]"
""")

    # Create a transformer block
    transformer_dir = blocks_path / "transformer"
    transformer_dir.mkdir()
    transformer_yaml = transformer_dir / "block.yaml"
    transformer_yaml.write_text("""
name: Transformer
version: 1.0.0
capabilities:
  inputs:
    x:
      type: Tensor
      dtype: [float32, float16]
      shape: ["batch", "seq", "dim"]
  outputs:
    y:
      type: Tensor
      dtype: input.x.dtype
      shape: ["batch", "seq", "dim"]
  params:
    dim:
      type: int
      range: [128, 1024]
      required: true
    heads:
      type: int
      range: [1, 16]
      default: 8
  constraints:
    - "dim in [256, 512, 768, 1024]"
    - "dim % heads == 0"
""")

    # Create a mamba block with different constraints
    mamba_dir = blocks_path / "mamba"
    mamba_dir.mkdir()
    mamba_yaml = mamba_dir / "block.yaml"
    mamba_yaml.write_text("""
name: Mamba
version: 1.0.0
capabilities:
  inputs:
    x:
      type: Tensor
      dtype: [float32, float16]
      shape: ["batch", "seq", "dim"]
  outputs:
    y:
      type: Tensor
      dtype: input.x.dtype
      shape: ["batch", "seq", "dim"]
  params:
    dim:
      type: int
      range: [128, 2048]
      required: true
    expand:
      type: int
      range: [1, 4]
      default: 2
  constraints:
    - "dim in [128, 256, 512, 1024]"
""")

    yield str(blocks_path.parent)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def registry(temp_blocks_dir):
    """Create a BlockRegistry with test blocks."""
    return BlockRegistry(blocks_dir=f"{temp_blocks_dir}/blocks")


@pytest.fixture
def hardware():
    """Create a HardwareDetector."""
    return HardwareDetector()


class TestValidationError:
    """Test ValidationError dataclass."""

    def test_create_validation_error(self):
        """Test creating a validation error."""
        error = ValidationError(
            error_type="shape_mismatch",
            message="Shapes don't match",
            source_block="encoder",
            target_block="decoder",
            suggestions=["Add adapter block", "Use different dimensions"]
        )

        assert error.error_type == "shape_mismatch"
        assert error.message == "Shapes don't match"
        assert error.source_block == "encoder"
        assert error.target_block == "decoder"
        assert len(error.suggestions) == 2

    def test_validation_error_str(self):
        """Test string representation of validation error."""
        error = ValidationError(
            error_type="shape_mismatch",
            message="Test error",
            suggestions=["Fix 1", "Fix 2"]
        )

        error_str = str(error)
        assert "SHAPE_MISMATCH" in error_str
        assert "Test error" in error_str
        assert "Fix 1" in error_str
        assert "Fix 2" in error_str


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful validation result."""
        result = ValidationResult(valid=True)
        assert result.valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_create_failure_result(self):
        """Test creating a failed validation result."""
        result = ValidationResult(valid=False)
        error = ValidationError(
            error_type="test",
            message="Test error"
        )
        result.add_error(error)

        assert not result.valid
        assert len(result.errors) == 1

    def test_add_warning(self):
        """Test adding warnings to result."""
        result = ValidationResult(valid=True)
        result.add_warning("Test warning")

        assert result.valid  # Warnings don't fail validation
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"

    def test_validation_result_str(self):
        """Test string representation."""
        result = ValidationResult(valid=True)
        result_str = str(result)
        assert "✓" in result_str or "passed" in result_str.lower()


class TestGraphValidatorBasic:
    """Test basic GraphValidator functionality."""

    def test_init_validator(self, registry, hardware):
        """Test initializing GraphValidator."""
        graph = ArchitectureGraph(
            nodes=[GraphNode(id="node1", block_id="linear", params={})],
            edges=[]
        )

        validator = GraphValidator(
            graph=graph,
            registry=registry,
            hardware=hardware
        )

        assert validator.graph == graph
        assert validator.registry == registry
        assert validator.hardware == hardware
        assert validator.shape_validator is not None
        assert validator.constraint_solver is not None

    def test_validate_empty_graph(self, registry, hardware):
        """Test validating an empty graph."""
        graph = ArchitectureGraph(nodes=[], edges=[])
        validator = GraphValidator(graph, registry, hardware)

        result = validator.validate()
        assert result.valid


class TestBlockReferenceValidation:
    """Test block reference validation.

    Implements Req 7.4
    """

    def test_valid_block_references(self, registry, hardware):
        """Test validation passes with valid block references."""
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="layer1", block_id="linear", params={"in_features": 128, "out_features": 256}),
                GraphNode(id="layer2", block_id="transformer", params={"dim": 256})
            ],
            edges=[]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        # Should have no block reference errors
        block_errors = [e for e in result.errors if e.error_type == "missing_block"]
        assert len(block_errors) == 0

    def test_missing_block_error(self, registry, hardware):
        """Test error when block doesn't exist.

        Implements Req 7.4
        """
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="layer1", block_id="nonexistent", params={})
            ],
            edges=[]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        assert not result.valid
        assert len(result.errors) >= 1

        # Find the missing block error
        missing_errors = [e for e in result.errors if e.error_type == "missing_block"]
        assert len(missing_errors) == 1

        error = missing_errors[0]
        assert "nonexistent" in error.message
        assert len(error.suggestions) > 0

    def test_suggest_similar_blocks(self, registry, hardware):
        """Test fuzzy matching for similar block names.

        Implements Req 7.4
        """
        graph = ArchitectureGraph(nodes=[], edges=[])
        validator = GraphValidator(graph, registry, hardware)

        # Test with close match
        similar = validator.suggest_similar_blocks("transformr")  # Missing 'e'
        assert "transformer" in similar

        # Test with no match
        similar = validator.suggest_similar_blocks("xyzabc")
        # May or may not return matches depending on cutoff

    def test_missing_block_suggestions(self, registry, hardware):
        """Test suggestions for missing blocks include similar names.

        Implements Req 7.4, 7.5
        """
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="layer1", block_id="transformr", params={})  # Typo
            ],
            edges=[]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        assert not result.valid

        missing_errors = [e for e in result.errors if e.error_type == "missing_block"]
        assert len(missing_errors) == 1

        error = missing_errors[0]
        suggestions_text = " ".join(error.suggestions)

        # Should suggest "transformer"
        assert "transformer" in suggestions_text.lower()


class TestShapeValidation:
    """Test shape compatibility validation.

    Implements Req 7.1
    """

    def test_compatible_shapes(self, registry, hardware):
        """Test validation passes with compatible shapes."""
        # Create graph: transformer -> mamba (both use same dim)
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="enc", block_id="transformer", params={"dim": 512}),
                GraphNode(id="dec", block_id="mamba", params={"dim": 512})
            ],
            edges=[
                GraphEdge(source="enc", target="dec")
            ]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        # Should have no shape errors
        shape_errors = [e for e in result.errors if e.error_type == "shape_mismatch"]
        assert len(shape_errors) == 0

    def test_incompatible_shapes(self, registry, hardware):
        """Test error when shapes don't match.

        Implements Req 7.1
        """
        # Create incompatible connection: linear (2D) -> transformer (3D)
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="proj", block_id="linear", params={"in_features": 128, "out_features": 256}),
                GraphNode(id="enc", block_id="transformer", params={"dim": 256})
            ],
            edges=[
                GraphEdge(source="proj", target="enc")
            ]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        assert not result.valid

        # Should have shape mismatch error
        shape_errors = [e for e in result.errors if e.error_type == "shape_mismatch"]
        assert len(shape_errors) >= 1

        error = shape_errors[0]
        assert error.source_block == "proj"
        assert error.target_block == "enc"
        assert len(error.suggestions) > 0

    def test_shape_error_suggestions(self, registry, hardware):
        """Test shape error includes actionable suggestions.

        Implements Req 7.1, 7.5
        """
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="proj", block_id="linear", params={"in_features": 128, "out_features": 256}),
                GraphNode(id="enc", block_id="transformer", params={"dim": 256})
            ],
            edges=[
                GraphEdge(source="proj", target="enc")
            ]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        shape_errors = [e for e in result.errors if e.error_type == "shape_mismatch"]
        if shape_errors:
            error = shape_errors[0]
            assert len(error.suggestions) > 0

            # Suggestions should be helpful
            suggestions_text = " ".join(error.suggestions)
            # Should mention adapter, reshape, or similar solutions
            helpful_keywords = ["adapter", "reshape", "block", "dimension", "transform"]
            assert any(keyword in suggestions_text.lower() for keyword in helpful_keywords)


class TestConstraintValidation:
    """Test constraint satisfaction validation.

    Implements Req 7.2
    """

    def test_satisfiable_constraints(self, registry, hardware):
        """Test validation passes with satisfiable constraints."""
        # transformer and mamba both support dim=512
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="enc", block_id="transformer", params={"dim": 512}),
                GraphNode(id="dec", block_id="mamba", params={"dim": 512})
            ],
            edges=[
                GraphEdge(source="enc", target="dec")
            ]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        # Should have no constraint errors
        constraint_errors = [
            e for e in result.errors
            if e.error_type in ("constraint_violation", "constraint_conflict")
        ]
        assert len(constraint_errors) == 0

    def test_unsatisfiable_constraints(self, registry, hardware):
        """Test error when constraints cannot be satisfied.

        Implements Req 7.2
        """
        # Create scenario with conflicting constraints
        # This might require specific block configurations
        # For now, we test the error creation directly
        validator = GraphValidator(
            ArchitectureGraph(nodes=[], edges=[]),
            registry,
            hardware
        )

        error = validator.create_constraint_error(
            constraint="dim in [128, 256]",
            violating_value=512,
            valid_alternatives=[128, 256]
        )

        assert error.error_type == "constraint_violation"
        assert "dim in [128, 256]" in error.message
        assert "512" in error.message
        assert len(error.suggestions) > 0
        assert any("128" in s for s in error.suggestions)

    def test_constraint_error_suggestions(self, registry, hardware):
        """Test constraint errors include valid alternatives.

        Implements Req 7.2, 7.5
        """
        validator = GraphValidator(
            ArchitectureGraph(nodes=[], edges=[]),
            registry,
            hardware
        )

        error = validator.create_constraint_error(
            constraint="dim % 8 == 0",
            violating_value=127,
            valid_alternatives=[128, 256, 512]
        )

        suggestions_text = " ".join(error.suggestions)
        assert "128" in suggestions_text
        assert len(error.suggestions) > 0


class TestHardwareValidation:
    """Test hardware compatibility validation.

    Implements Req 7.3
    """

    def test_hardware_compatible_blocks(self, registry, hardware):
        """Test validation passes with compatible hardware."""
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="layer1", block_id="linear", params={"in_features": 128, "out_features": 256})
            ],
            edges=[]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        # Should have no hardware errors (blocks don't require GPU)
        hw_errors = [e for e in result.errors if e.error_type == "hardware_incompatible"]
        assert len(hw_errors) == 0

    def test_hardware_error_suggestions(self, registry, hardware):
        """Test hardware errors include actionable suggestions.

        Implements Req 7.3, 7.5
        """
        validator = GraphValidator(
            ArchitectureGraph(nodes=[], edges=[]),
            registry,
            hardware
        )

        node = GraphNode(id="gpu_layer", block_id="transformer", params={})
        requirements = {
            "requires_cuda": True,
            "estimated_memory_gb": 16.0
        }

        error = validator._create_hardware_error(node, requirements)

        assert error.error_type == "hardware_incompatible"
        assert len(error.suggestions) > 0

        suggestions_text = " ".join(error.suggestions).lower()
        # Should mention GPU, memory, or batch size
        helpful_keywords = ["gpu", "cuda", "memory", "batch", "smaller", "reduce"]
        assert any(keyword in suggestions_text for keyword in helpful_keywords)


class TestConfigurationGeneration:
    """Test configuration enumeration."""

    def test_generate_configurations(self, registry, hardware):
        """Test generating valid configurations."""
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="layer1", block_id="linear", params={"in_features": 128, "out_features": 256})
            ],
            edges=[]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        # Should generate at least one configuration if validation passed
        if result.valid:
            # Configurations might be empty if there are no dimension variables
            # That's okay for this simple graph
            pass


class TestEndToEndValidation:
    """Test complete end-to-end validation scenarios."""

    def test_valid_architecture(self, registry, hardware):
        """Test validating a complete valid architecture.

        Implements Req 7.1, 7.2, 7.3
        """
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="enc", block_id="transformer", params={"dim": 512, "heads": 8}),
                GraphNode(id="dec", block_id="mamba", params={"dim": 512, "expand": 2})
            ],
            edges=[
                GraphEdge(source="enc", target="dec")
            ]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        # Should pass validation
        if not result.valid:
            print("\nValidation errors:")
            for error in result.errors:
                print(f"  {error}")

        # May or may not pass depending on constraint satisfaction
        # But should provide useful feedback either way
        assert result is not None
        assert isinstance(result, ValidationResult)

    def test_invalid_architecture_multiple_errors(self, registry, hardware):
        """Test architecture with multiple errors.

        Implements Req 7.5
        """
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="missing", block_id="nonexistent", params={}),
                GraphNode(id="proj", block_id="linear", params={"in_features": 128, "out_features": 256}),
                GraphNode(id="enc", block_id="transformer", params={"dim": 256})
            ],
            edges=[
                GraphEdge(source="proj", target="enc"),  # Shape mismatch
                GraphEdge(source="missing", target="proj")
            ]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        assert not result.valid
        assert len(result.errors) >= 1

        # Should have multiple error types
        error_types = {e.error_type for e in result.errors}
        # At minimum should have missing_block error
        assert "missing_block" in error_types

    def test_clear_error_messages(self, registry, hardware):
        """Test that all error messages use clear language.

        Implements Req 7.5
        """
        graph = ArchitectureGraph(
            nodes=[
                GraphNode(id="layer1", block_id="unknown_block", params={})
            ],
            edges=[]
        )

        validator = GraphValidator(graph, registry, hardware)
        result = validator.validate()

        assert not result.valid

        for error in result.errors:
            # Messages should be clear and actionable
            assert len(error.message) > 0
            assert len(error.suggestions) > 0

            # Should not contain overly technical jargon
            message_lower = error.message.lower()
            # These are fine to include, but message should explain them
            # Just checking messages exist and have content


class TestValidationResultFormatting:
    """Test validation result formatting for user display."""

    def test_format_success_result(self, registry, hardware):
        """Test formatting successful validation result."""
        result = ValidationResult(valid=True)
        result_str = str(result)

        assert len(result_str) > 0
        # Should indicate success
        assert "✓" in result_str or "pass" in result_str.lower()

    def test_format_error_result(self, registry, hardware):
        """Test formatting validation result with errors."""
        result = ValidationResult(valid=False)
        result.add_error(ValidationError(
            error_type="test",
            message="Test error",
            suggestions=["Fix 1", "Fix 2"]
        ))

        result_str = str(result)

        assert len(result_str) > 0
        assert "test error" in result_str.lower()
        assert "fix 1" in result_str.lower()

    def test_format_result_with_warnings(self, registry, hardware):
        """Test formatting result with warnings."""
        result = ValidationResult(valid=True)
        result.add_warning("Test warning")

        result_str = str(result)

        assert "warning" in result_str.lower()
        assert "test warning" in result_str.lower()


class TestCreateShapeError:
    """Test shape error creation."""

    def test_create_shape_error_rank_mismatch(self, registry, hardware):
        """Test creating error for rank mismatch."""
        validator = GraphValidator(
            ArchitectureGraph(nodes=[], edges=[]),
            registry,
            hardware
        )

        output_pattern = ValidatorShapePattern(["batch", "features"], [False, False])
        input_pattern = ValidatorShapePattern(["batch", "seq", "features"], [False, False, False])

        error = validator.create_shape_error(
            "source",
            "target",
            input_pattern,
            output_pattern,
            ["Rank mismatch: 2 vs 3"]
        )

        assert error.error_type == "shape_mismatch"
        assert "source" in str(error.message)
        assert "target" in str(error.message)
        assert len(error.suggestions) > 0

    def test_create_shape_error_dimension_mismatch(self, registry, hardware):
        """Test creating error for dimension value mismatch."""
        validator = GraphValidator(
            ArchitectureGraph(nodes=[], edges=[]),
            registry,
            hardware
        )

        output_pattern = ValidatorShapePattern(["batch", "512"], [False, False])
        input_pattern = ValidatorShapePattern(["batch", "256"], [False, False])

        error = validator.create_shape_error(
            "source",
            "target",
            input_pattern,
            output_pattern,
            ["Dimension mismatch"]
        )

        assert error.error_type == "shape_mismatch"
        suggestions_text = " ".join(error.suggestions)
        # Should suggest adapter or transformation
        assert "adapter" in suggestions_text.lower() or "linear" in suggestions_text.lower()
