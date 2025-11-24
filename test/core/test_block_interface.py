"""Unit tests for BlockInterface protocol."""

import pytest
from src.core.block_interface import (
    BlockCapability,
    BlockInterface,
    DimensionConstraint,
    ParameterSpec,
    ShapePattern,
)


class ValidBlock:
    """A valid block implementation that implements BlockInterface."""

    def get_capabilities(self) -> BlockCapability:
        return BlockCapability(
            name="TestBlock",
            version="1.0.0",
            inputs={"x": ShapePattern(["batch", "seq", "dim"])},
            outputs={"y": ShapePattern(["batch", "seq", "dim"])},
            params={"hidden": ParameterSpec("hidden", "int", required=True)},
        )

    def forward(self, x):
        return x


class MissingForward:
    """Block missing forward method."""

    def get_capabilities(self) -> BlockCapability:
        return BlockCapability(
            name="MissingForward",
            version="1.0.0",
        )


class MissingCapabilities:
    """Block missing get_capabilities method."""

    def forward(self, x):
        return x


def test_valid_block_implements_interface():
    """Test that a properly implemented block satisfies BlockInterface."""
    block = ValidBlock()
    assert isinstance(block, BlockInterface)


def test_missing_forward_fails_protocol():
    """Test that missing forward() method fails protocol check."""
    block = MissingForward()
    assert not isinstance(block, BlockInterface)


def test_missing_capabilities_fails_protocol():
    """Test that missing get_capabilities() method fails protocol check."""
    block = MissingCapabilities()
    assert not isinstance(block, BlockInterface)


def test_shape_pattern_creation():
    """Test ShapePattern dataclass creation."""
    pattern = ShapePattern(["batch", "seq", "dim"], dtype=["float32", "float16"])
    assert pattern.pattern == ["batch", "seq", "dim"]
    assert pattern.dtype == ["float32", "float16"]


def test_shape_pattern_optional_dtype():
    """Test ShapePattern without dtype specification."""
    pattern = ShapePattern(["*", "hidden"])
    assert pattern.pattern == ["*", "hidden"]
    assert pattern.dtype is None


def test_parameter_spec_required():
    """Test ParameterSpec with required parameter."""
    spec = ParameterSpec("hidden_dim", "int", required=True, range=(1, 10000))
    assert spec.name == "hidden_dim"
    assert spec.type == "int"
    assert spec.required is True
    assert spec.range == (1, 10000)


def test_parameter_spec_optional_with_default():
    """Test ParameterSpec with optional parameter and default."""
    spec = ParameterSpec("use_bias", "bool", required=False, default=True)
    assert spec.name == "use_bias"
    assert spec.type == "bool"
    assert spec.required is False
    assert spec.default is True


def test_parameter_spec_with_options():
    """Test ParameterSpec with enum options."""
    spec = ParameterSpec(
        "activation", "string", required=True, options=["relu", "gelu", "tanh"]
    )
    assert spec.options == ["relu", "gelu", "tanh"]


def test_dimension_constraint_creation():
    """Test DimensionConstraint creation."""
    constraint = DimensionConstraint("dim % 8 == 0")
    assert constraint.constraint_str == "dim % 8 == 0"


def test_block_capability_complete():
    """Test BlockCapability with all fields."""
    capability = BlockCapability(
        name="Linear",
        version="1.0.0",
        inputs={"x": ShapePattern(["*", "in_features"])},
        outputs={"y": ShapePattern(["*", "out_features"])},
        params={
            "in_features": ParameterSpec("in_features", "int", required=True),
            "out_features": ParameterSpec("out_features", "int", required=True),
        },
        constraints=[DimensionConstraint("in_features > 0"), DimensionConstraint("out_features > 0")],
        hardware_requirements={"min_memory_gb": 1},
    )
    assert capability.name == "Linear"
    assert capability.version == "1.0.0"
    assert len(capability.inputs) == 1
    assert len(capability.outputs) == 1
    assert len(capability.params) == 2
    assert len(capability.constraints) == 2
    assert capability.hardware_requirements["min_memory_gb"] == 1


def test_block_capability_minimal():
    """Test BlockCapability with minimal fields."""
    capability = BlockCapability(
        name="MinimalBlock",
        version="0.1.0",
    )
    assert capability.name == "MinimalBlock"
    assert capability.version == "0.1.0"
    assert len(capability.inputs) == 0
    assert len(capability.outputs) == 0
    assert len(capability.params) == 0
    assert len(capability.constraints) == 0
    assert capability.hardware_requirements is None


def test_valid_block_get_capabilities():
    """Test calling get_capabilities on valid block."""
    block = ValidBlock()
    cap = block.get_capabilities()
    assert cap.name == "TestBlock"
    assert "x" in cap.inputs
    assert "y" in cap.outputs
    assert "hidden" in cap.params
