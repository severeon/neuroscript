"""
Unit tests for ShapeValidator.

Tests Requirements 4.1, 4.2, 4.3, 4.4, 4.5 from specs/02-requirements.md
"""

import pytest
from src.core.shape_validator import (
    ShapeValidator,
    ShapePattern,
    UnificationResult,
    parse_shape_pattern
)


class TestShapePattern:
    """Test ShapePattern dataclass."""

    def test_create_shape_pattern(self):
        """Test creating a shape pattern."""
        pattern = ShapePattern(
            dimensions=["batch", "seq", "dim"],
            is_wildcard=[False, False, False]
        )
        assert len(pattern) == 3
        assert pattern.dimensions == ["batch", "seq", "dim"]

    def test_shape_pattern_validation(self):
        """Test that dimensions and is_wildcard must have same length."""
        with pytest.raises(ValueError, match="must have same length"):
            ShapePattern(
                dimensions=["batch", "seq"],
                is_wildcard=[False, False, False]
            )

    def test_shape_pattern_str(self):
        """Test string representation."""
        pattern = ShapePattern(
            dimensions=["batch", "seq", "dim"],
            is_wildcard=[False, False, False]
        )
        assert str(pattern) == "[batch, seq, dim]"


class TestUnificationResult:
    """Test UnificationResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = UnificationResult(success=True, bindings={"dim": "512"})
        assert result.success
        assert result.bindings == {"dim": "512"}
        assert result.errors == []

    def test_create_failure_result(self):
        """Test creating a failed result."""
        result = UnificationResult(success=False)
        result.add_error("Shape mismatch")
        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0] == "Shape mismatch"


class TestShapeValidatorBasicUnification:
    """Test basic unification scenarios."""

    def test_exact_match(self):
        """Test unification of identical patterns.

        Implements Req 4.1, 4.2
        """
        validator = ShapeValidator()
        pattern1 = ShapePattern(
            dimensions=["batch", "seq", "dim"],
            is_wildcard=[False, False, False]
        )
        pattern2 = ShapePattern(
            dimensions=["batch", "seq", "dim"],
            is_wildcard=[False, False, False]
        )

        result = validator.unify(pattern1, pattern2)
        assert result.success
        assert len(result.errors) == 0

    def test_concrete_to_variable_binding(self):
        """Test binding concrete value to variable.

        Implements Req 4.1, 4.2
        """
        validator = ShapeValidator()
        output_pattern = ShapePattern(
            dimensions=["batch", "seq", "512"],
            is_wildcard=[False, False, False]
        )
        input_pattern = ShapePattern(
            dimensions=["batch", "seq", "dim"],
            is_wildcard=[False, False, False]
        )

        result = validator.unify(output_pattern, input_pattern)
        assert result.success
        assert result.bindings.get("dim") == "512"
        assert len(result.errors) == 0

    def test_rank_mismatch(self):
        """Test that different ranks produce error.

        Implements Req 4.3
        """
        validator = ShapeValidator()
        output_pattern = ShapePattern(
            dimensions=["batch", "seq"],
            is_wildcard=[False, False]
        )
        input_pattern = ShapePattern(
            dimensions=["batch", "seq", "dim"],
            is_wildcard=[False, False, False]
        )

        result = validator.unify(output_pattern, input_pattern)
        assert not result.success
        assert len(result.errors) > 0
        assert "rank mismatch" in result.errors[0].lower()

    def test_concrete_value_mismatch(self):
        """Test that incompatible concrete values produce error.

        Implements Req 4.3
        """
        validator = ShapeValidator()
        output_pattern = ShapePattern(
            dimensions=["batch", "seq", "512"],
            is_wildcard=[False, False, False]
        )
        input_pattern = ShapePattern(
            dimensions=["batch", "seq", "256"],
            is_wildcard=[False, False, False]
        )

        result = validator.unify(output_pattern, input_pattern)
        assert not result.success
        assert len(result.errors) > 0


class TestShapeValidatorWildcards:
    """Test wildcard handling.

    Implements Req 4.4
    """

    def test_wildcard_matches_anything(self):
        """Test that wildcard matches any dimension."""
        validator = ShapeValidator()
        output_pattern = ShapePattern(
            dimensions=["*", "seq", "512"],
            is_wildcard=[True, False, False]
        )
        input_pattern = ShapePattern(
            dimensions=["batch", "seq", "512"],
            is_wildcard=[False, False, False]
        )

        result = validator.unify(output_pattern, input_pattern)
        assert result.success

    def test_wildcard_on_input_side(self):
        """Test wildcard on input side."""
        validator = ShapeValidator()
        output_pattern = ShapePattern(
            dimensions=["batch", "seq", "512"],
            is_wildcard=[False, False, False]
        )
        input_pattern = ShapePattern(
            dimensions=["batch", "*", "512"],
            is_wildcard=[False, True, False]
        )

        result = validator.unify(output_pattern, input_pattern)
        assert result.success

    def test_both_wildcards(self):
        """Test matching two wildcards."""
        validator = ShapeValidator()
        output_pattern = ShapePattern(
            dimensions=["*", "seq", "dim"],
            is_wildcard=[True, False, False]
        )
        input_pattern = ShapePattern(
            dimensions=["*", "seq", "dim"],
            is_wildcard=[True, False, False]
        )

        result = validator.unify(output_pattern, input_pattern)
        assert result.success


class TestShapeValidatorConsistency:
    """Test consistency validation across multiple unifications.

    Implements Req 4.5
    """

    def test_consistent_bindings_across_edges(self):
        """Test that same dimension name gets consistent values."""
        validator = ShapeValidator()

        # First edge: output dim=512, input dim=dim -> dim=512
        result1 = validator.unify(
            ShapePattern(["batch", "seq", "512"], [False, False, False]),
            ShapePattern(["batch", "seq", "dim"], [False, False, False])
        )
        assert result1.success
        assert validator.bindings.get("dim") == "512"

        # Second edge: output dim=dim, input dim=512 -> should succeed
        result2 = validator.unify(
            ShapePattern(["batch", "seq", "dim"], [False, False, False]),
            ShapePattern(["batch", "seq", "512"], [False, False, False])
        )
        assert result2.success

    def test_inconsistent_bindings_rejected(self):
        """Test that inconsistent bindings are rejected."""
        validator = ShapeValidator()

        # First edge: bind dim to 512
        result1 = validator.unify(
            ShapePattern(["batch", "seq", "512"], [False, False, False]),
            ShapePattern(["batch", "seq", "dim"], [False, False, False])
        )
        assert result1.success

        # Second edge: try to bind dim to 256 -> should fail
        result2 = validator.unify(
            ShapePattern(["batch", "seq", "256"], [False, False, False]),
            ShapePattern(["batch", "seq", "dim"], [False, False, False])
        )
        assert not result2.success
        assert "inconsistent" in result2.errors[0].lower()

    def test_variable_to_variable_binding(self):
        """Test binding one variable to another."""
        validator = ShapeValidator()

        # First edge: dim1 binds to dim2
        result = validator.unify(
            ShapePattern(["batch", "seq", "dim1"], [False, False, False]),
            ShapePattern(["batch", "seq", "dim2"], [False, False, False])
        )
        assert result.success
        # One should be bound to the other
        assert "dim2" in result.bindings or "dim1" in result.bindings


class TestMatchDimension:
    """Test the match_dimension method."""

    def test_wildcard_matches_concrete(self):
        """Test wildcard matching concrete value."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("*", "512", True, False)
        assert success
        assert binding is None

    def test_concrete_matches_wildcard(self):
        """Test concrete value matching wildcard."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("512", "*", False, True)
        assert success
        assert binding is None

    def test_concrete_matches_concrete_same(self):
        """Test matching identical concrete values."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("512", "512", False, False)
        assert success
        assert binding is None

    def test_concrete_matches_concrete_different(self):
        """Test matching different concrete values."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("512", "256", False, False)
        assert not success

    def test_concrete_binds_to_variable(self):
        """Test binding concrete value to variable."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("512", "dim", False, False)
        assert success
        assert binding == ("dim", "512")

    def test_variable_binds_to_concrete(self):
        """Test binding variable to concrete value."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("dim", "512", False, False)
        assert success
        assert binding == ("dim", "512")

    def test_same_variable_names(self):
        """Test matching same variable names."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("dim", "dim", False, False)
        assert success
        assert binding is None

    def test_different_variable_names(self):
        """Test matching different variable names."""
        validator = ShapeValidator()
        success, binding = validator.match_dimension("dim1", "dim2", False, False)
        assert success
        assert binding is not None
        # Should create equivalence binding


class TestValidateConsistency:
    """Test the validate_consistency method."""

    def test_no_existing_binding(self):
        """Test consistency when no existing binding."""
        validator = ShapeValidator()
        assert validator.validate_consistency("dim", "512")

    def test_consistent_binding(self):
        """Test consistency with matching existing binding."""
        validator = ShapeValidator()
        validator.bindings["dim"] = "512"
        assert validator.validate_consistency("dim", "512")

    def test_inconsistent_binding(self):
        """Test inconsistency with conflicting existing binding."""
        validator = ShapeValidator()
        validator.bindings["dim"] = "512"
        assert not validator.validate_consistency("dim", "256")


class TestParseShapePattern:
    """Test the parse_shape_pattern helper function."""

    def test_parse_named_dimensions(self):
        """Test parsing named dimensions."""
        pattern = parse_shape_pattern(["batch", "seq", "dim"])
        assert pattern.dimensions == ["batch", "seq", "dim"]
        assert pattern.is_wildcard == [False, False, False]

    def test_parse_wildcards(self):
        """Test parsing wildcards."""
        pattern = parse_shape_pattern(["*", "*", "dim"])
        assert pattern.dimensions == ["*", "*", "dim"]
        assert pattern.is_wildcard == [True, True, False]

    def test_parse_concrete_values(self):
        """Test parsing concrete integer values."""
        pattern = parse_shape_pattern([32, 128, 512])
        assert pattern.dimensions == ["32", "128", "512"]
        assert pattern.is_wildcard == [False, False, False]

    def test_parse_mixed(self):
        """Test parsing mixed pattern."""
        pattern = parse_shape_pattern(["*", "seq", 512])
        assert pattern.dimensions == ["*", "seq", "512"]
        assert pattern.is_wildcard == [True, False, False]


class TestShapeValidatorHelpers:
    """Test helper methods."""

    def test_reset_bindings(self):
        """Test resetting bindings."""
        validator = ShapeValidator()
        validator.bindings = {"dim": "512", "batch": "32"}
        validator.reset_bindings()
        assert len(validator.bindings) == 0

    def test_get_bindings(self):
        """Test getting bindings copy."""
        validator = ShapeValidator()
        validator.bindings = {"dim": "512"}
        bindings = validator.get_bindings()
        assert bindings == {"dim": "512"}
        # Verify it's a copy
        bindings["other"] = "value"
        assert "other" not in validator.bindings


class TestComplexScenarios:
    """Test complex multi-edge validation scenarios."""

    def test_chain_of_three_blocks(self):
        """Test validating a chain: A -> B -> C."""
        validator = ShapeValidator()

        # A -> B: [batch, 128, 512] -> [batch, seq, dim]
        result1 = validator.unify(
            ShapePattern(["batch", "128", "512"], [False, False, False]),
            ShapePattern(["batch", "seq", "dim"], [False, False, False])
        )
        assert result1.success
        assert validator.bindings["seq"] == "128"
        assert validator.bindings["dim"] == "512"

        # B -> C: [batch, seq, dim] -> [batch, 128, 512]
        result2 = validator.unify(
            ShapePattern(["batch", "seq", "dim"], [False, False, False]),
            ShapePattern(["batch", "128", "512"], [False, False, False])
        )
        assert result2.success

    def test_shape_transformation(self):
        """Test shape transformation (dimension changes)."""
        validator = ShapeValidator()

        # Encoder: [batch, seq, 256] -> [batch, seq, 512]
        result = validator.unify(
            ShapePattern(["batch", "seq", "256"], [False, False, False]),
            ShapePattern(["batch", "seq", "512"], [False, False, False])
        )
        # This should fail because 256 != 512
        assert not result.success

    def test_shape_preservation(self):
        """Test shape preservation through block."""
        validator = ShapeValidator()

        # LayerNorm: input shape = output shape
        result1 = validator.unify(
            ShapePattern(["batch", "seq", "dim"], [False, False, False]),
            ShapePattern(["batch", "seq", "dim"], [False, False, False])
        )
        assert result1.success

    def test_error_message_quality(self):
        """Test that error messages are descriptive.

        Implements Req 4.3
        """
        validator = ShapeValidator()
        output_pattern = ShapePattern(
            dimensions=["batch", "seq", "512"],
            is_wildcard=[False, False, False]
        )
        input_pattern = ShapePattern(
            dimensions=["batch", "seq", "dim", "extra"],
            is_wildcard=[False, False, False, False]
        )

        result = validator.unify(output_pattern, input_pattern)
        assert not result.success
        assert len(result.errors) > 0
        # Error should mention dimensions and patterns
        error_msg = result.errors[0]
        assert "3" in error_msg  # output dims
        assert "4" in error_msg  # input dims
