"""Tests for Linear block implementation."""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from blocks.linear.module import Linear
from src.core.block_interface import BlockInterface
from src.core.capability_parser import CapabilityParser
from src.core.block_registry import BlockRegistry


class TestLinearBlock:
    """Test suite for Linear block."""

    def test_linear_initialization(self):
        """Test that Linear block can be initialized with valid parameters."""
        block = Linear(in_features=128, out_features=256, bias=True)
        assert block.in_features == 128
        assert block.out_features == 256
        assert block.has_bias is True

    def test_linear_initialization_no_bias(self):
        """Test Linear block initialization without bias."""
        block = Linear(in_features=128, out_features=256, bias=False)
        assert block.has_bias is False

    def test_linear_default_bias(self):
        """Test that bias defaults to True."""
        block = Linear(in_features=128, out_features=256)
        assert block.has_bias is True

    def test_linear_parameter_validation_in_features(self):
        """Test that invalid in_features raises ValueError."""
        with pytest.raises(ValueError, match="in_features must be a positive integer"):
            Linear(in_features=0, out_features=256)

        with pytest.raises(ValueError, match="in_features must be a positive integer"):
            Linear(in_features=-1, out_features=256)

    def test_linear_parameter_validation_out_features(self):
        """Test that invalid out_features raises ValueError."""
        with pytest.raises(ValueError, match="out_features must be a positive integer"):
            Linear(in_features=128, out_features=0)

        with pytest.raises(ValueError, match="out_features must be a positive integer"):
            Linear(in_features=128, out_features=-1)

    def test_linear_parameter_range_upper_bound(self):
        """Test that parameters exceeding upper bound raise ValueError."""
        with pytest.raises(ValueError, match="in_features must be <= 100000"):
            Linear(in_features=100001, out_features=256)

        with pytest.raises(ValueError, match="out_features must be <= 100000"):
            Linear(in_features=128, out_features=100001)

    def test_linear_forward_2d_tensor(self):
        """Test forward pass with 2D tensor [batch, in_features]."""
        block = Linear(in_features=128, out_features=256)
        x = torch.randn(32, 128)  # [batch=32, in_features=128]
        y = block(x)

        assert y.shape == (32, 256)  # [batch=32, out_features=256]
        assert y.dtype == x.dtype

    def test_linear_forward_3d_tensor(self):
        """Test forward pass with 3D tensor [batch, seq, in_features]."""
        block = Linear(in_features=128, out_features=256)
        x = torch.randn(16, 50, 128)  # [batch=16, seq=50, in_features=128]
        y = block(x)

        assert y.shape == (16, 50, 256)  # [batch=16, seq=50, out_features=256]
        assert y.dtype == x.dtype

    def test_linear_forward_1d_tensor(self):
        """Test forward pass with 1D tensor [in_features]."""
        block = Linear(in_features=128, out_features=256)
        x = torch.randn(128)  # [in_features=128]
        y = block(x)

        assert y.shape == (256,)  # [out_features=256]
        assert y.dtype == x.dtype

    def test_linear_dtype_preservation_float32(self):
        """Test that float32 dtype is preserved."""
        block = Linear(in_features=128, out_features=256)
        x = torch.randn(32, 128, dtype=torch.float32)
        y = block(x)

        assert y.dtype == torch.float32

    def test_linear_dtype_preservation_float16(self):
        """Test that float16 dtype is preserved."""
        block = Linear(in_features=128, out_features=256)
        block = block.half()  # Convert to half precision
        x = torch.randn(32, 128, dtype=torch.float16)
        y = block(x)

        assert y.dtype == torch.float16

    def test_linear_shape_mismatch_error(self):
        """Test that shape mismatch raises ValueError."""
        block = Linear(in_features=128, out_features=256)
        x = torch.randn(32, 64)  # Wrong in_features (64 instead of 128)

        with pytest.raises(ValueError, match="Input feature dimension mismatch"):
            block(x)

    def test_linear_implements_block_interface(self):
        """Test that Linear block implements BlockInterface protocol."""
        block = Linear(in_features=128, out_features=256)
        assert isinstance(block, BlockInterface)

    def test_get_capabilities_returns_block_capability(self):
        """Test that get_capabilities() returns a BlockCapability object."""
        block = Linear(in_features=128, out_features=256)
        capabilities = block.get_capabilities()

        assert capabilities.name == "Linear"
        assert capabilities.version == "1.0.0"
        assert "x" in capabilities.inputs
        assert "y" in capabilities.outputs

    def test_get_capabilities_input_shape(self):
        """Test that input shape pattern is correctly specified."""
        block = Linear(in_features=128, out_features=256)
        capabilities = block.get_capabilities()

        x_input = capabilities.inputs["x"]
        assert x_input.pattern == ["*", "in_features"]
        assert set(x_input.dtype) == {"float32", "float16", "bfloat16"}

    def test_get_capabilities_output_shape(self):
        """Test that output shape pattern is correctly specified."""
        block = Linear(in_features=128, out_features=256)
        capabilities = block.get_capabilities()

        y_output = capabilities.outputs["y"]
        assert y_output.pattern == ["*", "out_features"]
        assert y_output.dtype == ["input.x.dtype"]  # Should reference input dtype (as list)

    def test_get_capabilities_parameters(self):
        """Test that parameters are correctly specified."""
        block = Linear(in_features=128, out_features=256)
        capabilities = block.get_capabilities()

        assert "in_features" in capabilities.params
        assert "out_features" in capabilities.params
        assert "bias" in capabilities.params

        in_features_param = capabilities.params["in_features"]
        assert in_features_param.type == "int"
        assert in_features_param.required is True
        assert in_features_param.range == (1, 100000)

        out_features_param = capabilities.params["out_features"]
        assert out_features_param.type == "int"
        assert out_features_param.required is True
        assert out_features_param.range == (1, 100000)

        bias_param = capabilities.params["bias"]
        assert bias_param.type == "bool"
        assert bias_param.required is False
        assert bias_param.default is True

    def test_get_capabilities_constraints(self):
        """Test that constraints are correctly specified."""
        block = Linear(in_features=128, out_features=256)
        capabilities = block.get_capabilities()

        assert len(capabilities.constraints) == 2
        constraint_strs = [c.constraint_str for c in capabilities.constraints]
        assert "in_features >= 1" in constraint_strs
        assert "out_features >= 1" in constraint_strs

    def test_capability_parser_integration(self):
        """Test that CapabilityParser can parse Linear block.yaml."""
        block_dir = Path(__file__).parent
        block_yaml = block_dir / "block.yaml"

        parser = CapabilityParser()
        capabilities = parser.parse_file(str(block_yaml))

        assert capabilities.name == "Linear"
        assert capabilities.version == "1.0.0"
        assert "x" in capabilities.inputs
        assert "y" in capabilities.outputs
        assert "in_features" in capabilities.params
        assert "out_features" in capabilities.params
        assert "bias" in capabilities.params

    def test_block_registry_discovery(self):
        """Test that BlockRegistry can discover Linear block."""
        # BlockRegistry looks for blocks in blocks/ directory
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Check that linear block was discovered
        assert "linear" in registry.list_blocks()

    def test_block_registry_get_block(self):
        """Test that BlockRegistry can retrieve Linear block capabilities."""
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Get linear block capabilities
        capabilities = registry.get_block("linear")

        assert capabilities is not None
        assert capabilities.name == "Linear"
        assert capabilities.version == "1.0.0"

    def test_repr(self):
        """Test string representation of Linear block."""
        block = Linear(in_features=128, out_features=256, bias=True)
        repr_str = repr(block)

        assert "Linear" in repr_str
        assert "in_features=128" in repr_str
        assert "out_features=256" in repr_str
        assert "bias=True" in repr_str

    def test_small_dimensions(self):
        """Test with minimum allowed dimensions."""
        block = Linear(in_features=1, out_features=1)
        x = torch.randn(1, 1)
        y = block(x)

        assert y.shape == (1, 1)

    def test_large_dimensions(self):
        """Test with large dimensions."""
        block = Linear(in_features=10000, out_features=5000)
        x = torch.randn(2, 10000)
        y = block(x)

        assert y.shape == (2, 5000)

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = Linear(in_features=128, out_features=256)
        x = torch.randn(32, 128, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert block.linear.weight.grad is not None
        if block.has_bias:
            assert block.linear.bias.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
