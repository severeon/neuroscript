"""Tests for LayerNorm block implementation."""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from blocks.layernorm.module import LayerNorm
from src.core.block_interface import BlockInterface
from src.core.capability_parser import CapabilityParser
from src.core.block_registry import BlockRegistry


class TestLayerNormBlock:
    """Test suite for LayerNorm block."""

    def test_layernorm_initialization(self):
        """Test that LayerNorm block can be initialized with valid parameters."""
        block = LayerNorm(normalized_dim=128)
        assert block.normalized_dim == 128
        assert block.eps == 1e-5  # default value
        assert block.elementwise_affine is True  # default value

    def test_layernorm_initialization_custom_params(self):
        """Test LayerNorm block initialization with custom parameters."""
        block = LayerNorm(normalized_dim=256, eps=1e-6, elementwise_affine=False)
        assert block.normalized_dim == 256
        assert block.eps == 1e-6
        assert block.elementwise_affine is False

    def test_layernorm_parameter_validation_normalized_dim(self):
        """Test that invalid normalized_dim raises ValueError."""
        with pytest.raises(ValueError, match="normalized_dim must be a positive integer"):
            LayerNorm(normalized_dim=0)

        with pytest.raises(ValueError, match="normalized_dim must be a positive integer"):
            LayerNorm(normalized_dim=-1)

    def test_layernorm_parameter_range_upper_bound(self):
        """Test that normalized_dim exceeding upper bound raises ValueError."""
        with pytest.raises(ValueError, match="normalized_dim must be <= 10000"):
            LayerNorm(normalized_dim=10001)

    def test_layernorm_eps_validation(self):
        """Test that invalid eps raises ValueError."""
        with pytest.raises(ValueError, match="eps must be a non-negative number"):
            LayerNorm(normalized_dim=128, eps=-1e-5)

    def test_layernorm_elementwise_affine_validation(self):
        """Test that invalid elementwise_affine raises ValueError."""
        with pytest.raises(ValueError, match="elementwise_affine must be a boolean"):
            LayerNorm(normalized_dim=128, elementwise_affine="true")

    def test_layernorm_forward_2d_tensor(self):
        """Test forward pass with 2D tensor [batch, normalized_dim]."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(32, 128)  # [batch=32, normalized_dim=128]
        y = block(x)

        assert y.shape == (32, 128)  # Shape preserved
        assert y.dtype == torch.float32

    def test_layernorm_forward_3d_tensor(self):
        """Test forward pass with 3D tensor [batch, seq, normalized_dim]."""
        block = LayerNorm(normalized_dim=256)
        x = torch.randn(16, 50, 256)  # [batch=16, seq=50, normalized_dim=256]
        y = block(x)

        assert y.shape == (16, 50, 256)  # Shape preserved
        assert y.dtype == torch.float32

    def test_layernorm_forward_4d_tensor(self):
        """Test forward pass with 4D tensor [batch, heads, seq, normalized_dim]."""
        block = LayerNorm(normalized_dim=64)
        x = torch.randn(8, 12, 100, 64)  # [batch=8, heads=12, seq=100, normalized_dim=64]
        y = block(x)

        assert y.shape == (8, 12, 100, 64)  # Shape preserved
        assert y.dtype == torch.float32

    def test_layernorm_shape_preservation(self):
        """Test that output shape exactly matches input shape."""
        block = LayerNorm(normalized_dim=512)
        
        # Test various input shapes
        shapes = [
            (32, 512),           # 2D
            (16, 50, 512),       # 3D
            (8, 12, 100, 512),   # 4D
            (4, 2, 10, 20, 512), # 5D
        ]
        
        for shape in shapes:
            x = torch.randn(*shape)
            y = block(x)
            assert y.shape == x.shape, f"Shape not preserved for input shape {shape}"

    def test_layernorm_dtype_validation(self):
        """Test that invalid input dtype raises ValueError."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randint(0, 100, (32, 128))  # int64 instead of float

        with pytest.raises(ValueError, match="Input must have dtype"):
            block(x)

    def test_layernorm_dtype_preservation_float32(self):
        """Test that float32 dtype is preserved."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(32, 128, dtype=torch.float32)
        y = block(x)
        assert y.dtype == torch.float32

    def test_layernorm_dtype_preservation_float16(self):
        """Test that float16 dtype is preserved."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(32, 128, dtype=torch.float16)
        y = block(x)
        assert y.dtype == torch.float16

    def test_layernorm_dtype_preservation_bfloat16(self):
        """Test that bfloat16 dtype is preserved."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(32, 128, dtype=torch.bfloat16)
        y = block(x)
        assert y.dtype == torch.bfloat16

    def test_layernorm_last_dimension_validation(self):
        """Test that mismatched last dimension raises ValueError."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(32, 256)  # Last dimension is 256, not 128

        with pytest.raises(ValueError, match="Input last dimension must be 128"):
            block(x)

    def test_layernorm_normalization_mean(self):
        """Test that normalized output has mean ≈ 0 over normalized dimension."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(32, 128) * 10 + 5  # Mean around 5, std around 10
        y = block(x)

        # Compute mean over the normalized dimension (last dimension)
        mean = y.mean(dim=-1)
        
        # Mean should be close to 0 (within tolerance for numerical precision)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_layernorm_normalization_std(self):
        """Test that normalized output has std ≈ 1 over normalized dimension."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(32, 128) * 10 + 5  # Mean around 5, std around 10
        y = block(x)

        # Compute std over the normalized dimension (last dimension)
        std = y.std(dim=-1, unbiased=False)
        
        # Std should be close to 1 (within tolerance for numerical precision)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-5)

    def test_layernorm_elementwise_affine_true(self):
        """Test that elementwise_affine=True includes learnable parameters."""
        block = LayerNorm(normalized_dim=128, elementwise_affine=True)
        
        # Should have weight and bias parameters
        assert hasattr(block.layer_norm, 'weight')
        assert hasattr(block.layer_norm, 'bias')
        assert block.layer_norm.weight is not None
        assert block.layer_norm.bias is not None

    def test_layernorm_elementwise_affine_false(self):
        """Test that elementwise_affine=False excludes learnable parameters."""
        block = LayerNorm(normalized_dim=128, elementwise_affine=False)
        
        # Should not have weight and bias parameters
        assert block.layer_norm.weight is None
        assert block.layer_norm.bias is None

    def test_layernorm_eps_prevents_division_by_zero(self):
        """Test that eps prevents division by zero for constant inputs."""
        block = LayerNorm(normalized_dim=128, eps=1e-5)
        x = torch.ones(32, 128) * 5.0  # Constant input (std = 0)
        
        # Should not raise error due to division by zero
        y = block(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_layernorm_implements_block_interface(self):
        """Test that LayerNorm block implements BlockInterface protocol."""
        block = LayerNorm(normalized_dim=128)
        assert isinstance(block, BlockInterface)

    def test_get_capabilities_returns_block_capability(self):
        """Test that get_capabilities() returns a BlockCapability object."""
        block = LayerNorm(normalized_dim=128)
        capabilities = block.get_capabilities()

        assert capabilities.name == "LayerNorm"
        assert capabilities.version == "1.0.0"
        assert "x" in capabilities.inputs
        assert "y" in capabilities.outputs

    def test_get_capabilities_input_shape(self):
        """Test that input shape pattern is correctly specified."""
        block = LayerNorm(normalized_dim=128)
        capabilities = block.get_capabilities()

        x_input = capabilities.inputs["x"]
        assert x_input.pattern == ["*", "normalized_dim"]
        assert set(x_input.dtype) == {"float32", "float16", "bfloat16"}

    def test_get_capabilities_output_shape(self):
        """Test that output shape pattern is correctly specified."""
        block = LayerNorm(normalized_dim=128)
        capabilities = block.get_capabilities()

        y_output = capabilities.outputs["y"]
        assert y_output.pattern == ["*", "normalized_dim"]
        # Output dtype should reference input dtype
        assert "input.x.dtype" in str(y_output.dtype) or y_output.dtype == ["input.x.dtype"]

    def test_get_capabilities_parameters(self):
        """Test that parameters are correctly specified."""
        block = LayerNorm(normalized_dim=128)
        capabilities = block.get_capabilities()

        assert "normalized_dim" in capabilities.params
        assert "eps" in capabilities.params
        assert "elementwise_affine" in capabilities.params

        normalized_dim_param = capabilities.params["normalized_dim"]
        assert normalized_dim_param.type == "int"
        assert normalized_dim_param.required is True
        assert normalized_dim_param.range == (1, 10000)

        eps_param = capabilities.params["eps"]
        assert eps_param.type == "float"
        assert eps_param.required is False
        # YAML parser may return as string or float
        assert eps_param.default == 1e-5 or eps_param.default == "1e-5"

        affine_param = capabilities.params["elementwise_affine"]
        assert affine_param.type == "bool"
        assert affine_param.required is False
        assert affine_param.default is True

    def test_get_capabilities_constraints(self):
        """Test that constraints are correctly specified."""
        block = LayerNorm(normalized_dim=128)
        capabilities = block.get_capabilities()

        assert len(capabilities.constraints) == 1
        constraint_strs = [c.constraint_str for c in capabilities.constraints]
        assert "normalized_dim >= 1" in constraint_strs

    def test_capability_parser_integration(self):
        """Test that CapabilityParser can parse LayerNorm block.yaml."""
        block_dir = Path(__file__).parent
        block_yaml = block_dir / "block.yaml"

        parser = CapabilityParser()
        capabilities = parser.parse_file(str(block_yaml))

        assert capabilities.name == "LayerNorm"
        assert capabilities.version == "1.0.0"
        assert "x" in capabilities.inputs
        assert "y" in capabilities.outputs
        assert "normalized_dim" in capabilities.params
        assert "eps" in capabilities.params
        assert "elementwise_affine" in capabilities.params

    def test_block_registry_discovery(self):
        """Test that BlockRegistry can discover LayerNorm block."""
        # BlockRegistry looks for blocks in blocks/ directory
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Check that layernorm block was discovered
        assert "layernorm" in registry.list_blocks()

    def test_block_registry_get_block(self):
        """Test that BlockRegistry can retrieve LayerNorm block capabilities."""
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Get layernorm block capabilities
        capabilities = registry.get_block("layernorm")

        assert capabilities is not None
        assert capabilities.name == "LayerNorm"
        assert capabilities.version == "1.0.0"

    def test_repr(self):
        """Test string representation of LayerNorm block."""
        block = LayerNorm(normalized_dim=128)
        repr_str = repr(block)

        assert "LayerNorm" in repr_str
        assert "normalized_dim=128" in repr_str
        assert "eps=1e-05" in repr_str or "eps=0.00001" in repr_str
        assert "elementwise_affine=True" in repr_str

    def test_repr_custom_params(self):
        """Test string representation with custom parameters."""
        block = LayerNorm(normalized_dim=256, eps=1e-6, elementwise_affine=False)
        repr_str = repr(block)

        assert "normalized_dim=256" in repr_str
        assert "eps=1e-06" in repr_str or "eps=0.000001" in repr_str
        assert "elementwise_affine=False" in repr_str

    def test_small_dimensions(self):
        """Test with minimum allowed dimensions."""
        block = LayerNorm(normalized_dim=1)
        x = torch.randn(32, 1)
        y = block(x)

        assert y.shape == (32, 1)

    def test_large_dimensions(self):
        """Test with large dimensions."""
        block = LayerNorm(normalized_dim=4096)
        x = torch.randn(8, 4096)
        y = block(x)

        assert y.shape == (8, 4096)

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = LayerNorm(normalized_dim=128, elementwise_affine=True)
        x = torch.randn(32, 128, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert block.layer_norm.weight.grad is not None
        assert block.layer_norm.bias.grad is not None

    def test_gradient_flow_no_affine(self):
        """Test gradient flow when elementwise_affine=False."""
        block = LayerNorm(normalized_dim=128, elementwise_affine=False)
        x = torch.randn(32, 128, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None

    def test_empty_batch(self):
        """Test with empty batch dimension."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(0, 128)
        y = block(x)

        assert y.shape == (0, 128)

    def test_single_sample(self):
        """Test with single sample (batch size = 1)."""
        block = LayerNorm(normalized_dim=128)
        x = torch.randn(1, 128)
        y = block(x)

        assert y.shape == (1, 128)
        
        # Check normalization
        mean = y.mean(dim=-1)
        std = y.std(dim=-1, unbiased=False)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
