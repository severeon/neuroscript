"""Unit tests for CapabilityParser."""

import tempfile
from pathlib import Path

import pytest

from src.core.block_interface import ParameterSpec
from src.core.capability_parser import CapabilityParseError, CapabilityParser


@pytest.fixture
def parser():
    """Create a CapabilityParser instance."""
    return CapabilityParser()


@pytest.fixture
def temp_yaml(tmp_path):
    """Create a temporary YAML file."""
    def _create_yaml(content: str) -> Path:
        yaml_file = tmp_path / "block.yaml"
        yaml_file.write_text(content)
        return yaml_file
    return _create_yaml


def test_extract_dimensions_list(parser):
    """Test extracting dimensions from list format."""
    dims = parser.extract_dimensions(["batch", "seq", "dim"])
    assert dims == ["batch", "seq", "dim"]


def test_extract_dimensions_bracket_string(parser):
    """Test extracting dimensions from [dim1, dim2, ...] format."""
    dims = parser.extract_dimensions("[batch, seq, dim]")
    assert dims == ["batch", "seq", "dim"]


def test_extract_dimensions_space_separated(parser):
    """Test extracting dimensions from space-separated format."""
    dims = parser.extract_dimensions("batch seq dim")
    assert dims == ["batch", "seq", "dim"]


def test_extract_dimensions_with_wildcards(parser):
    """Test extracting dimensions with wildcards."""
    dims = parser.extract_dimensions(["*", "hidden", "*"])
    assert dims == ["*", "hidden", "*"]


def test_extract_dimensions_bracket_with_wildcard(parser):
    """Test extracting wildcard dimensions from bracket format."""
    dims = parser.extract_dimensions("[*, seq, dim]")
    assert dims == ["*", "seq", "dim"]


def test_extract_dimensions_invalid_type(parser, tmp_path):
    """Test that invalid dimension type raises error."""
    with pytest.raises(CapabilityParseError):
        parser.extract_dimensions(123, tmp_path / "test.yaml")


def test_validate_param_range_valid(parser):
    """Test validating parameter within range."""
    spec = ParameterSpec("hidden", "int", required=True, range=(1, 10000))
    # Should not raise
    parser.validate_param_range("hidden", 512, spec)


def test_validate_param_range_min(parser):
    """Test parameter at minimum bound."""
    spec = ParameterSpec("hidden", "int", required=True, range=(1, 10000))
    parser.validate_param_range("hidden", 1, spec)


def test_validate_param_range_max(parser):
    """Test parameter at maximum bound."""
    spec = ParameterSpec("hidden", "int", required=True, range=(1, 10000))
    parser.validate_param_range("hidden", 10000, spec)


def test_validate_param_range_below_min(parser):
    """Test parameter below minimum raises error."""
    spec = ParameterSpec("hidden", "int", required=True, range=(1, 10000))
    with pytest.raises(CapabilityParseError, match="out of range"):
        parser.validate_param_range("hidden", 0, spec)


def test_validate_param_range_above_max(parser):
    """Test parameter above maximum raises error."""
    spec = ParameterSpec("hidden", "int", required=True, range=(1, 10000))
    with pytest.raises(CapabilityParseError, match="out of range"):
        parser.validate_param_range("hidden", 10001, spec)


def test_validate_param_range_no_range(parser):
    """Test that params without range don't raise errors."""
    spec = ParameterSpec("activation", "string")
    # Should not raise
    parser.validate_param_range("activation", "relu", spec)


def test_parse_file_missing_file(parser):
    """Test parsing non-existent file."""
    with pytest.raises(CapabilityParseError, match="File not found"):
        parser.parse_file("/nonexistent/path/block.yaml")


def test_parse_file_malformed_yaml(parser, temp_yaml):
    """Test parsing malformed YAML."""
    yaml_file = temp_yaml("invalid: yaml: content:")
    with pytest.raises(CapabilityParseError, match="Malformed YAML"):
        parser.parse_file(str(yaml_file))


def test_parse_file_not_dict(parser, temp_yaml):
    """Test parsing YAML that's not a dict."""
    yaml_file = temp_yaml("- item1\n- item2")
    with pytest.raises(CapabilityParseError, match="Expected YAML dict"):
        parser.parse_file(str(yaml_file))


def test_parse_file_missing_name(parser, temp_yaml):
    """Test parsing file without name."""
    yaml_file = temp_yaml("version: 1.0.0\ncapabilities: {}")
    with pytest.raises(CapabilityParseError, match="Missing required field 'name'"):
        parser.parse_file(str(yaml_file))


def test_parse_file_missing_version(parser, temp_yaml):
    """Test parsing file without version."""
    yaml_file = temp_yaml("name: TestBlock\ncapabilities: {}")
    with pytest.raises(CapabilityParseError, match="Missing required field 'version'"):
        parser.parse_file(str(yaml_file))


def test_parse_file_minimal(parser, temp_yaml):
    """Test parsing minimal valid block.yaml."""
    yaml_file = temp_yaml("name: MinimalBlock\nversion: 1.0.0")
    cap = parser.parse_file(str(yaml_file))
    assert cap.name == "MinimalBlock"
    assert cap.version == "1.0.0"
    assert len(cap.inputs) == 0
    assert len(cap.outputs) == 0


def test_parse_file_with_inputs(parser, temp_yaml):
    """Test parsing block with inputs."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: [batch, seq, dim]
      dtype: [float32, float16]
"""
    yaml_file = temp_yaml(content)
    cap = parser.parse_file(str(yaml_file))
    assert "x" in cap.inputs
    assert cap.inputs["x"].pattern == ["batch", "seq", "dim"]
    assert cap.inputs["x"].dtype == ["float32", "float16"]


def test_parse_file_with_outputs(parser, temp_yaml):
    """Test parsing block with outputs."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  outputs:
    y:
      shape: [batch, seq, hidden]
"""
    yaml_file = temp_yaml(content)
    cap = parser.parse_file(str(yaml_file))
    assert "y" in cap.outputs
    assert cap.outputs["y"].pattern == ["batch", "seq", "hidden"]


def test_parse_file_with_params(parser, temp_yaml):
    """Test parsing block with parameters."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  params:
    hidden_dim:
      type: int
      required: true
      range: [1, 10000]
    use_bias:
      type: bool
      required: false
      default: true
"""
    yaml_file = temp_yaml(content)
    cap = parser.parse_file(str(yaml_file))
    assert "hidden_dim" in cap.params
    assert cap.params["hidden_dim"].type == "int"
    assert cap.params["hidden_dim"].required is True
    assert cap.params["hidden_dim"].range == (1.0, 10000.0)
    assert "use_bias" in cap.params
    assert cap.params["use_bias"].default is True


def test_parse_file_with_constraints(parser, temp_yaml):
    """Test parsing block with constraints."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  constraints:
    - hidden_dim % 8 == 0
    - hidden_dim > 0
"""
    yaml_file = temp_yaml(content)
    cap = parser.parse_file(str(yaml_file))
    assert len(cap.constraints) == 2
    assert cap.constraints[0].constraint_str == "hidden_dim % 8 == 0"


def test_parse_file_invalid_range_not_numeric(parser, temp_yaml):
    """Test parsing with non-numeric range values."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  params:
    param1:
      type: int
      range: [one, two]
"""
    yaml_file = temp_yaml(content)
    with pytest.raises(CapabilityParseError, match="Range values must be numeric"):
        parser.parse_file(str(yaml_file))


def test_parse_file_invalid_range_wrong_length(parser, temp_yaml):
    """Test parsing with wrong range length."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  params:
    param1:
      type: int
      range: [1, 2, 3]
"""
    yaml_file = temp_yaml(content)
    with pytest.raises(CapabilityParseError, match="must be \\[min, max\\]"):
        parser.parse_file(str(yaml_file))


def test_parse_file_missing_input_shape(parser, temp_yaml):
    """Test parsing input without shape."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  inputs:
    x:
      dtype: float32
"""
    yaml_file = temp_yaml(content)
    with pytest.raises(CapabilityParseError, match="Missing 'shape'"):
        parser.parse_file(str(yaml_file))


def test_parse_file_missing_param_type(parser, temp_yaml):
    """Test parsing parameter without type."""
    content = """
name: TestBlock
version: 1.0.0
capabilities:
  params:
    param1:
      required: true
"""
    yaml_file = temp_yaml(content)
    with pytest.raises(CapabilityParseError, match="Missing 'type'"):
        parser.parse_file(str(yaml_file))


def test_parse_file_complete_linear_block(parser, temp_yaml):
    """Test parsing a complete Linear block specification."""
    content = """
name: Linear
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: [*, in_features]
      dtype: [float32, float16]
  outputs:
    y:
      shape: [*, out_features]
      dtype: [float32, float16]
  params:
    in_features:
      type: int
      required: true
      range: [1, 100000]
    out_features:
      type: int
      required: true
      range: [1, 100000]
    bias:
      type: bool
      required: false
      default: true
  constraints:
    - in_features > 0
    - out_features > 0
hardware_requirements:
  min_memory_gb: 0.5
"""
    yaml_file = temp_yaml(content)
    cap = parser.parse_file(str(yaml_file))
    assert cap.name == "Linear"
    assert "x" in cap.inputs
    assert "y" in cap.outputs
    assert "in_features" in cap.params
    assert "out_features" in cap.params
    assert "bias" in cap.params
    assert len(cap.constraints) == 2
    assert cap.hardware_requirements["min_memory_gb"] == 0.5
