"""Tests for Embedding block implementation."""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from blocks.embedding.module import Embedding
from src.core.block_interface import BlockInterface
from src.core.capability_parser import CapabilityParser
from src.core.block_registry import BlockRegistry


class TestEmbeddingBlock:
    """Test suite for Embedding block."""

    def test_embedding_initialization(self):
        """Test that Embedding block can be initialized with valid parameters."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        assert block.vocab_size == 1000
        assert block.embedding_dim == 128
        assert block.padding_idx is None

    def test_embedding_initialization_with_padding(self):
        """Test Embedding block initialization with padding_idx."""
        block = Embedding(vocab_size=1000, embedding_dim=128, padding_idx=0)
        assert block.padding_idx == 0

    def test_embedding_parameter_validation_vocab_size(self):
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
            Embedding(vocab_size=0, embedding_dim=128)

        with pytest.raises(ValueError, match="vocab_size must be a positive integer"):
            Embedding(vocab_size=-1, embedding_dim=128)

    def test_embedding_parameter_validation_embedding_dim(self):
        """Test that invalid embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
            Embedding(vocab_size=1000, embedding_dim=0)

        with pytest.raises(ValueError, match="embedding_dim must be a positive integer"):
            Embedding(vocab_size=1000, embedding_dim=-1)

    def test_embedding_parameter_range_upper_bound(self):
        """Test that parameters exceeding upper bound raise ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be <= 1000000"):
            Embedding(vocab_size=1000001, embedding_dim=128)

        with pytest.raises(ValueError, match="embedding_dim must be <= 10000"):
            Embedding(vocab_size=1000, embedding_dim=10001)

    def test_embedding_padding_idx_validation(self):
        """Test that invalid padding_idx raises ValueError."""
        with pytest.raises(ValueError, match="padding_idx must be in range"):
            Embedding(vocab_size=1000, embedding_dim=128, padding_idx=-1)

        with pytest.raises(ValueError, match="padding_idx must be in range"):
            Embedding(vocab_size=1000, embedding_dim=128, padding_idx=1000)

    def test_embedding_forward_2d_tensor(self):
        """Test forward pass with 2D tensor [batch, seq]."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.randint(0, 1000, (32, 50))  # [batch=32, seq=50]
        embeddings = block(indices)

        assert embeddings.shape == (32, 50, 128)  # [batch=32, seq=50, embedding_dim=128]
        assert embeddings.dtype == torch.float32

    def test_embedding_forward_1d_tensor(self):
        """Test forward pass with 1D tensor [seq]."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.randint(0, 1000, (50,))  # [seq=50]
        embeddings = block(indices)

        assert embeddings.shape == (50, 128)  # [seq=50, embedding_dim=128]
        assert embeddings.dtype == torch.float32

    def test_embedding_forward_3d_tensor(self):
        """Test forward pass with 3D tensor [batch, num_sequences, seq]."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.randint(0, 1000, (16, 10, 50))  # [batch=16, num_seq=10, seq=50]
        embeddings = block(indices)

        assert embeddings.shape == (16, 10, 50, 128)  # [batch=16, num_seq=10, seq=50, embedding_dim=128]
        assert embeddings.dtype == torch.float32

    def test_embedding_dtype_validation(self):
        """Test that non-integer input raises ValueError."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.randn(32, 50)  # float32 instead of long

        with pytest.raises(ValueError, match="Input indices must have dtype torch.long"):
            block(indices)

    def test_embedding_index_range_validation_negative(self):
        """Test that negative indices raise ValueError."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.tensor([1, 2, -1, 4], dtype=torch.long)

        with pytest.raises(ValueError, match="Input indices contain negative values"):
            block(indices)

    def test_embedding_index_range_validation_exceeds_vocab(self):
        """Test that indices exceeding vocab_size raise ValueError."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.tensor([1, 2, 1000, 4], dtype=torch.long)

        with pytest.raises(ValueError, match="Input indices exceed vocab_size"):
            block(indices)

    def test_embedding_with_padding_idx(self):
        """Test that padding_idx works correctly."""
        block = Embedding(vocab_size=1000, embedding_dim=128, padding_idx=0)
        indices = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)
        embeddings = block(indices)

        # Padding embeddings should be zero
        assert torch.allclose(embeddings[:, :, :][indices == 0], torch.zeros(128))

    def test_embedding_discrete_to_dense_transformation(self):
        """Test that embedding transforms discrete indices to dense vectors."""
        block = Embedding(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([5, 10, 15, 20], dtype=torch.long)
        embeddings = block(indices)

        # Output should be dense (continuous values)
        assert embeddings.shape == (4, 64)
        assert embeddings.dtype == torch.float32
        
        # Different indices should produce different embeddings
        assert not torch.allclose(embeddings[0], embeddings[1])
        assert not torch.allclose(embeddings[1], embeddings[2])

    def test_embedding_same_index_same_embedding(self):
        """Test that the same index produces the same embedding."""
        block = Embedding(vocab_size=100, embedding_dim=64)
        indices = torch.tensor([5, 5, 10, 10], dtype=torch.long)
        embeddings = block(indices)

        # Same indices should produce identical embeddings
        assert torch.allclose(embeddings[0], embeddings[1])
        assert torch.allclose(embeddings[2], embeddings[3])

    def test_embedding_implements_block_interface(self):
        """Test that Embedding block implements BlockInterface protocol."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        assert isinstance(block, BlockInterface)

    def test_get_capabilities_returns_block_capability(self):
        """Test that get_capabilities() returns a BlockCapability object."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        capabilities = block.get_capabilities()

        assert capabilities.name == "Embedding"
        assert capabilities.version == "1.0.0"
        assert "indices" in capabilities.inputs
        assert "embeddings" in capabilities.outputs

    def test_get_capabilities_input_shape(self):
        """Test that input shape pattern is correctly specified."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        capabilities = block.get_capabilities()

        indices_input = capabilities.inputs["indices"]
        assert indices_input.pattern == ["*", "seq"]
        assert indices_input.dtype == ["int64"]

    def test_get_capabilities_output_shape(self):
        """Test that output shape pattern is correctly specified."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        capabilities = block.get_capabilities()

        embeddings_output = capabilities.outputs["embeddings"]
        assert embeddings_output.pattern == ["*", "seq", "embedding_dim"]
        assert set(embeddings_output.dtype) == {"float32", "float16", "bfloat16"}

    def test_get_capabilities_parameters(self):
        """Test that parameters are correctly specified."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        capabilities = block.get_capabilities()

        assert "vocab_size" in capabilities.params
        assert "embedding_dim" in capabilities.params
        assert "padding_idx" in capabilities.params

        vocab_size_param = capabilities.params["vocab_size"]
        assert vocab_size_param.type == "int"
        assert vocab_size_param.required is True
        assert vocab_size_param.range == (1, 1000000)

        embedding_dim_param = capabilities.params["embedding_dim"]
        assert embedding_dim_param.type == "int"
        assert embedding_dim_param.required is True
        assert embedding_dim_param.range == (1, 10000)

        padding_idx_param = capabilities.params["padding_idx"]
        assert padding_idx_param.type == "int"
        assert padding_idx_param.required is False
        assert padding_idx_param.default is None

    def test_get_capabilities_constraints(self):
        """Test that constraints are correctly specified."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        capabilities = block.get_capabilities()

        assert len(capabilities.constraints) == 2
        constraint_strs = [c.constraint_str for c in capabilities.constraints]
        assert "vocab_size >= 1" in constraint_strs
        assert "embedding_dim >= 1" in constraint_strs

    def test_capability_parser_integration(self):
        """Test that CapabilityParser can parse Embedding block.yaml."""
        block_dir = Path(__file__).parent
        block_yaml = block_dir / "block.yaml"

        parser = CapabilityParser()
        capabilities = parser.parse_file(str(block_yaml))

        assert capabilities.name == "Embedding"
        assert capabilities.version == "1.0.0"
        assert "indices" in capabilities.inputs
        assert "embeddings" in capabilities.outputs
        assert "vocab_size" in capabilities.params
        assert "embedding_dim" in capabilities.params
        assert "padding_idx" in capabilities.params

    def test_block_registry_discovery(self):
        """Test that BlockRegistry can discover Embedding block."""
        # BlockRegistry looks for blocks in blocks/ directory
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Check that embedding block was discovered
        assert "embedding" in registry.list_blocks()

    def test_block_registry_get_block(self):
        """Test that BlockRegistry can retrieve Embedding block capabilities."""
        blocks_dir = Path(__file__).parent.parent
        registry = BlockRegistry(blocks_dir=str(blocks_dir))

        # Get embedding block capabilities
        capabilities = registry.get_block("embedding")

        assert capabilities is not None
        assert capabilities.name == "Embedding"
        assert capabilities.version == "1.0.0"

    def test_repr(self):
        """Test string representation of Embedding block."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        repr_str = repr(block)

        assert "Embedding" in repr_str
        assert "vocab_size=1000" in repr_str
        assert "embedding_dim=128" in repr_str

    def test_repr_with_padding(self):
        """Test string representation with padding_idx."""
        block = Embedding(vocab_size=1000, embedding_dim=128, padding_idx=0)
        repr_str = repr(block)

        assert "padding_idx=0" in repr_str

    def test_small_dimensions(self):
        """Test with minimum allowed dimensions."""
        block = Embedding(vocab_size=1, embedding_dim=1)
        indices = torch.tensor([0], dtype=torch.long)
        embeddings = block(indices)

        assert embeddings.shape == (1, 1)

    def test_large_dimensions(self):
        """Test with large dimensions."""
        block = Embedding(vocab_size=100000, embedding_dim=1024)
        indices = torch.randint(0, 100000, (2, 100))
        embeddings = block(indices)

        assert embeddings.shape == (2, 100, 1024)

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.randint(0, 1000, (32, 50))
        embeddings = block(indices)
        loss = embeddings.sum()
        loss.backward()

        assert block.embedding.weight.grad is not None

    def test_empty_tensor(self):
        """Test with empty tensor."""
        block = Embedding(vocab_size=1000, embedding_dim=128)
        indices = torch.tensor([], dtype=torch.long).reshape(0, 10)
        embeddings = block(indices)

        assert embeddings.shape == (0, 10, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
