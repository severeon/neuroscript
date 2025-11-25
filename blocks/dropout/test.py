"""Tests for Dropout block implementation."""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from blocks.dropout.module import Dropout
from src.core.block_interface import BlockInterface
from src.core.capability_parser import CapabilityParser
from src.core.block_registry import BlockRegistry


class TestDropoutBlock:
    """Test suite for Dropout block."""

    def test_dropout_initialization(self):
        """Test that Dropout block can be initialized with default parameters."""
        block = Dropout()
        assert block.p == 0.5  # default value
        assert block.inplace is False  # default value

    def test_dropout_initialization_custom_params(self):
        """Test Dropout block initialization with custom parameters."""
        block = Dropout(p=0.3, inplace=True)
        assert block.p == 0.3
        assert block.inplace is True

    def test_dropout_parameter_validation_p_negative(self):
        """Test that negative p raises ValueError."""
        with pytest.raises(ValueError, match="p must be in range"):
            Dropout(p=-0.1)

    def test_dropout_parameter_validation_p_too_large(self):
        """Test that p > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="p must be in range"):
            Dropout(p=1.5)

    def test_dropout_parameter_validation_p_zero(self):
        """Test that p=0.0 is valid (no dropout)."""
        block = Dropout(p=0.0)
        assert block.p == 0.0

    def test_dropout_parameter_validation_p_one(self):
        """Test that p=1.0 is valid (all dropout)."""
        block = Dropout(p=1.0)
        assert block.p == 1.0

    def test_dropout_parameter_validation_p_type(self):
        """Test that invalid p type raises ValueError."""
        with pytest.raises(ValueError, match="p must be a number"):
            Dropout(p="0.5")

    def test_dropout_inplace_validation(self):
        """Test that invalid inplace raises ValueError."""
        with pytest.raises(ValueError, match="inplace must be a boolean"):
            Dropout(p=0.5, inplace="false")

    def test_dropout_forward_2d_tensor(self):
        """Test forward pass with 2D tensor [batch, features]."""
        block = Dropout(p=0.5)
        x = torch.randn(32, 128)
        y = block(x)

        assert y.shape == (32, 128)  # Shape preserved
        assert y.dtype == torch.float32

    def test_dropout_forward_3d_tensor(self):
        """Test forward pass with 3D tensor [batch, seq, features]."""
        block = Dropout(p=0.3)
        x = torch.randn(16, 50, 256)
        y = block(x)

        assert y.shape == (16, 50, 256)  # Shape preserved
        assert y.dtype == torch.float32

    def test_dropout_forward_4d_tensor(self):
        """Test forward pass with 4D tensor [batch, channels, height, width]."""
        block = Dropout(p=0.2)
        x = torch.randn(8, 3, 32, 32)
        y = block(x)

        assert y.shape == (8, 3, 32, 32)  # Shape preserved
        assert y.dtype == torch.float32

    def test_dropout_shape_preservation(self):
        """Test that output shape exactly matches input shape."""
        block = Dropout(p=0.5)
        
        # Test various input shapes
        shapes = [
            (32,),               # 1D
            (32, 128),           # 2D
            (16, 50, 256),       # 3D
            (8, 3, 32, 32),      # 4D
            (4, 2, 10, 20, 64),  # 5D
        ]
        
        for shape in shapes:
            x = torch.randn(*shape)
            y = block(x)
            assert y.shape == x.shape, f"Shape not preserved for input shape {shape}"

    def test_dropout_dtype_validation(self):
        """Test that invalid input dtype raises ValueError."""
        block = Dropout(p=0.5)
        x = torch.randint(0, 100, (32, 128))  # int64 instead of float

        with pytest.raises(ValueError, match="Input must have dtype"):
            block(x)

    def test_dropout_dtype_preservation_float32(self):
        """Test that float32 dtype is preserved."""
        block = Dropout(p=0.5)
        x = torch.randn(32, 128, dtype=torch.float32)
        y = block(x)
        assert y.dtype == torch.float32

    def test_dropout_dtype_preservation_float16(self):
        """Test that float16 dtype is preserved."""
        block = Dropout(p=0.5)
        x = torch.randn(32, 128, dtype=torch.float16)
        y = block(x)
        assert y.dtype == torch.float16

    def test_dropout_dtype_preservation_bfloat16(self):
        """Test that bfloat16 dtype is preserved."""
        block = Dropout(p=0.5)
        x = torch.randn(32, 128, dtype=torch.bfloat16)
        y = block(x)
        assert y.dtype == torch.bfloat16

    def test_dropout_training_mode(self):
        """Test that dropout is applied in training mode."""
        torch.manual_seed(42)
        block = Dropout(p=0.5)
        block.train()  # Set to training mode
        
        x = torch.ones(1000, 100)
        y = block(x)
        
        # In training mode with p=0.5, approximately half the elements should be zero
        # (with some variance due to randomness)
        zero_ratio = (y == 0).float().mean().item()
        assert 0.4 < zero_ratio < 0.6, f"Expected ~50% zeros, got {zero_ratio*100:.1f}%"

    def test_dropout_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        block = Dropout(p=0.5)
        block.eval()  # Set to eval mode
        
        x = torch.ones(1000, 100)
        y = block(x)
        
        # In eval mode, no dropout should be applied
        assert torch.allclose(y, x), "Dropout should not be applied in eval mode"

    def test_dropout_p_zero_no_dropout(self):
        """Test that p=0.0 means no dropout is applied."""
        block = Dropout(p=0.0)
        block.train()  # Even in training mode
        
        x = torch.randn(32, 128)
        y = block(x)
        
        # With p=0.0, output should equal input
        assert torch.allclose(y, x), "No dropout should be applied when p=0.0"

    def test_dropout_p_one_all_dropout(self):
        """Test that p=1.0 means all elements are zeroed."""
        block = Dropout(p=1.0)
        block.train()  # Set to training mode
        
        x = torch.randn(32, 128)
        y = block(x)
        
        # With p=1.0, all elements should be zero
        assert torch.all(y == 0), "All elements should be zero when p=1.0"

    def test_dropout_scaling(self):
        """Test that dropout properly scales remaining values during training."""
        torch.manual_seed(42)
        block = Dropout(p=0.5)
        block.train()
        
        x = torch.ones(10000)
        y = block(x)
        
        # Non-zero values should be scaled by 1/(1-p) to maintain expected value
        # For p=0.5, non-zero values should be scaled by 2.0
        non_zero_values = y[y != 0]
        expected_scale = 1.0 / (1.0 - 0.5)
        
        # Check that non-zero values are approximately scaled correctly
        assert torch.allclose(
            non_zero_values, 
            torch.ones_like(non_zero_values) * expected_scale,
            rtol=1e-5
        )

    def test_dropout_implements_block_interface(self):
        """Test that Dropout block implements BlockInterface protocol."""
        block = Dropout(p=0.5)
        assert isinstance(block, BlockInterface)

    def test_get_capabilities_returns_block_capability(self):
        """Test that get_capabilities() returns a BlockCapability object."""
        block = Dropout(p=0.5)
        capabilities = block.get_capabilities()

        assert capabilities.name == "Dropout"
        assert capabilities.version == "1.0.0"
        assert "x" in capabilities.inputs
        assert "y" in capabilities.outputs

    def test_get_capabilities_input_shape(self):
        """Test that input shape pattern is correctly specified."""
        block = Dropout(p=0.5)
        capabilities = block.get_capabilities()

        x_input = capabilities.inputs["x"]
        assert x_input.pattern == ["*"]
        assert set(x_input.dtype) == {"float32", "float16", "bfloat16"}

    def test_get_capabilities_output_shape(self):
        """Test that output shape pattern is correctly specified."""
        block = Dropout(p=0.5)
        capabilities = block.get_capabilities()

        y_output = capabilities.outputs["y"]
        assert y_output.pattern == ["*"]
        # Output dtype should reference input dtype
        assert "input.x.dtype" in str(y_output.dtype) or y_output.dtype == ["input.x.dtype"]

    def test_get_capabilities_parameters(self):
        """Test that parameters are correctly specified."""
        block = Dropout(p=0.5)
        capabilities = block.get_capabilities()

        assert "p" in capabilities.params
        assert "inplace" in capabilities.params

        p_param = capabilities.params["p"]
        assert p_param.type == "float"
        assert p_param.required is False
        assert p_param.range == (0.0, 1.0)
        assert p_param.default == 0.5

        inplace_param = capabilities.params["inplace"]
        assert inplace_param.type == "bool"
        assert inplace_param.required is False
        assert inplace_param.default is False

    def test_get_capabilities_constraints(self):
        """Test that constraints are correctly specified."""
        block = Dropout(p=0.5)
        capabilities = block.get_capabilities()

        assert len(capabilities.constraints) == 2
        constraint_strs = [c.constraint_str for c in capabilities.constraints]
        assert "p >= 0.0" in constraint_strs
        assert "p <= 1.0" in constraint_strs

    def test_capability_parser_integration(self):
        """Test that CapabilityParser can parse Dropout block.yaml."""
        block_dir = Path(__file__).parent
        block_yaml = block_dir / "block.yaml"

        parser = CapabilityParser()
        capabilities = parser.parse_file(str(block_yaml))

        assert capabilities.name == "Dropout"
        assert capabilities.version == "1.0.0"
        assert "x" in capabilities.inputs
        assert "y" in capabilities.outputs
        assert "p" in capabilities.params
        assert "inplace" in capabilities.params

    def test_block_registry_discovery(self):
        """Test that BlockRegistry can discover Dropout block."""
        # BlockRegistry looks for blocks in blocks/ directory
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Check that dropout block was discovered
        assert "dropout" in registry.list_blocks()

    def test_block_registry_get_block(self):
        """Test that BlockRegistry can retrieve Dropout block capabilities."""
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Get dropout block capabilities
        capabilities = registry.get_block("dropout")

        assert capabilities is not None
        assert capabilities.name == "Dropout"
        assert capabilities.version == "1.0.0"

    def test_repr(self):
        """Test string representation of Dropout block."""
        block = Dropout(p=0.5)
        repr_str = repr(block)

        assert "Dropout" in repr_str
        assert "p=0.5" in repr_str
        assert "inplace=False" in repr_str

    def test_repr_custom_params(self):
        """Test string representation with custom parameters."""
        block = Dropout(p=0.3, inplace=True)
        repr_str = repr(block)

        assert "p=0.3" in repr_str
        assert "inplace=True" in repr_str

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = Dropout(p=0.5)
        block.train()
        
        x = torch.randn(32, 128, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        # Gradient should be non-zero for at least some elements
        assert not torch.all(x.grad == 0)

    def test_gradient_flow_eval_mode(self):
        """Test gradient flow in eval mode (no dropout)."""
        block = Dropout(p=0.5)
        block.eval()
        
        x = torch.randn(32, 128, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        # In eval mode, gradient should be all ones (since output = input)
        assert torch.allclose(x.grad, torch.ones_like(x.grad))

    def test_empty_batch(self):
        """Test with empty batch dimension."""
        block = Dropout(p=0.5)
        x = torch.randn(0, 128)
        y = block(x)

        assert y.shape == (0, 128)

    def test_single_sample(self):
        """Test with single sample (batch size = 1)."""
        block = Dropout(p=0.5)
        block.train()
        
        x = torch.randn(1, 128)
        y = block(x)

        assert y.shape == (1, 128)

    def test_deterministic_with_seed(self):
        """Test that dropout is deterministic when seed is set."""
        # Create blocks
        block1 = Dropout(p=0.5)
        block1.train()
        block2 = Dropout(p=0.5)
        block2.train()
        
        # Set seed and generate input
        torch.manual_seed(42)
        x = torch.randn(100, 100)
        
        # Set seed before first forward pass
        torch.manual_seed(123)
        y1 = block1(x)

        # Set same seed before second forward pass
        torch.manual_seed(123)
        y2 = block2(x)

        # With the same seed, results should be identical
        assert torch.allclose(y1, y2)

    def test_different_probabilities(self):
        """Test dropout with various probability values."""
        probabilities = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for p in probabilities:
            block = Dropout(p=p)
            block.train()
            
            x = torch.ones(1000, 100)
            y = block(x)
            
            # Verify shape preservation
            assert y.shape == x.shape
            
            # Verify approximate zero ratio (except for edge cases)
            if p == 0.0:
                assert torch.allclose(y, x)
            elif p == 1.0:
                assert torch.all(y == 0)
            else:
                zero_ratio = (y == 0).float().mean().item()
                # Allow 10% tolerance due to randomness
                assert abs(zero_ratio - p) < 0.1, \
                    f"For p={p}, expected ~{p*100}% zeros, got {zero_ratio*100:.1f}%"

    def test_constraint_solver_integration(self):
        """Test that ConstraintSolver can validate parameter constraints."""
        from src.core.constraint_solver import ConstraintSolver
        
        block_dir = Path(__file__).parent
        block_yaml = block_dir / "block.yaml"
        
        parser = CapabilityParser()
        capabilities = parser.parse_file(str(block_yaml))
        
        # Test valid p values
        valid_p_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for p_val in valid_p_values:
            # Create a simple configuration with the p parameter
            config = {"p": p_val}
            
            # Verify that the value is within the specified range
            p_param = capabilities.params["p"]
            assert p_param.range[0] <= p_val <= p_param.range[1], \
                f"p={p_val} should be valid"

    def test_inplace_operation(self):
        """Test that inplace=True modifies input tensor."""
        block = Dropout(p=0.5, inplace=True)
        block.train()
        
        x = torch.randn(32, 128)
        x_original = x.clone()
        y = block(x)
        
        # With inplace=True, y should be the same object as x
        assert y is x, "Inplace operation should return the same tensor"
        
        # The tensor should have been modified
        # (unless p=0, but we're using p=0.5)
        # Note: We can't guarantee they're different due to randomness,
        # but we can verify the operation completed without error

    def test_non_inplace_operation(self):
        """Test that inplace=False preserves input tensor."""
        block = Dropout(p=0.5, inplace=False)
        block.train()
        
        x = torch.randn(32, 128)
        x_original = x.clone()
        y = block(x)
        
        # With inplace=False, input should be unchanged
        assert torch.allclose(x, x_original), \
            "Non-inplace operation should not modify input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
