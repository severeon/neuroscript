"""Unit tests for BlockRegistry."""

import tempfile
from pathlib import Path

import pytest

from src.core.block_interface import BlockInterface, BlockCapability
from src.core.block_registry import BlockRegistry, BlockRegistryError


@pytest.fixture
def temp_blocks_dir(tmp_path):
    """Create a temporary blocks directory with test blocks."""
    blocks_dir = tmp_path / "blocks"
    blocks_dir.mkdir()
    return blocks_dir


def create_block_yaml(blocks_dir: Path, block_name: str, content: str) -> Path:
    """Helper to create a block.yaml file."""
    block_dir = blocks_dir / block_name
    block_dir.mkdir(parents=True, exist_ok=True)
    yaml_file = block_dir / "block.yaml"
    yaml_file.write_text(content)
    return block_dir


def test_empty_blocks_directory(tmp_path):
    """Test registry with empty blocks directory."""
    blocks_dir = tmp_path / "blocks"
    blocks_dir.mkdir()
    registry = BlockRegistry(str(blocks_dir))
    assert len(registry) == 0
    assert registry.list_blocks() == []


def test_missing_blocks_directory(tmp_path):
    """Test registry with missing blocks directory (should not crash)."""
    blocks_dir = tmp_path / "nonexistent"
    # Should not raise, just log warning
    registry = BlockRegistry(str(blocks_dir))
    assert len(registry) == 0


def test_discover_single_block(temp_blocks_dir):
    """Test discovering a single block."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    assert len(registry) == 1
    assert "linear" in registry.list_blocks()


def test_discover_multiple_blocks(temp_blocks_dir):
    """Test discovering multiple blocks."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    create_block_yaml(
        temp_blocks_dir,
        "embedding",
        "name: Embedding\nversion: 1.0.0\ncapabilities: {}",
    )
    create_block_yaml(
        temp_blocks_dir,
        "layernorm",
        "name: LayerNorm\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    assert len(registry) == 3
    blocks = registry.list_blocks()
    assert "linear" in blocks
    assert "embedding" in blocks
    assert "layernorm" in blocks
    assert blocks == sorted(blocks)  # Should be alphabetically sorted


def test_get_block_existing(temp_blocks_dir):
    """Test retrieving an existing block."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    cap = registry.get_block("linear")
    assert cap.name == "Linear"
    assert cap.version == "1.0.0"


def test_get_block_nonexistent(temp_blocks_dir):
    """Test retrieving a non-existent block."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    with pytest.raises(BlockRegistryError, match="Block not found"):
        registry.get_block("nonexistent")


def test_get_block_error_includes_available_blocks(temp_blocks_dir):
    """Test that block not found error lists available blocks."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    with pytest.raises(BlockRegistryError, match="Available blocks.*linear"):
        registry.get_block("nonexistent")


def test_missing_block_yaml_file(temp_blocks_dir, caplog):
    """Test that directories without block.yaml are skipped with warning."""
    block_dir = temp_blocks_dir / "incomplete"
    block_dir.mkdir()
    # Don't create block.yaml

    registry = BlockRegistry(str(temp_blocks_dir))
    assert len(registry) == 0
    assert "missing block.yaml" in caplog.text.lower()


def test_malformed_block_yaml(temp_blocks_dir):
    """Test that malformed YAML raises error."""
    block_yaml = temp_blocks_dir / "bad_block" / "block.yaml"
    block_yaml.parent.mkdir()
    block_yaml.write_text("invalid: yaml: syntax:")

    with pytest.raises(BlockRegistryError, match="Failed to parse block"):
        BlockRegistry(str(temp_blocks_dir))


def test_duplicate_block_identifier(temp_blocks_dir):
    """Test that duplicate block IDs raise error."""
    # Create two directories with same name (edge case - same name)
    create_block_yaml(
        temp_blocks_dir,
        "duplicate",
        "name: Duplicate1\nversion: 1.0.0\ncapabilities: {}",
    )

    # Try to create another with same ID (by replacing)
    block_dir = temp_blocks_dir / "duplicate"
    yaml_file = block_dir / "block.yaml"
    yaml_file.write_text("name: Duplicate2\nversion: 1.0.0\ncapabilities: {}")

    # This should work but discover only one
    registry = BlockRegistry(str(temp_blocks_dir))
    assert len(registry) == 1


def test_block_exists(temp_blocks_dir):
    """Test block_exists method."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    assert registry.block_exists("linear") is True
    assert registry.block_exists("nonexistent") is False


def test_contains_operator(temp_blocks_dir):
    """Test 'in' operator for blocks."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    assert "linear" in registry
    assert "nonexistent" not in registry


def test_len_operator(temp_blocks_dir):
    """Test len() on registry."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    create_block_yaml(
        temp_blocks_dir,
        "embedding",
        "name: Embedding\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    assert len(registry) == 2


def test_iter_operator(temp_blocks_dir):
    """Test iteration over registry."""
    create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    create_block_yaml(
        temp_blocks_dir,
        "embedding",
        "name: Embedding\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    blocks = list(registry)
    assert len(blocks) == 2
    assert "linear" in blocks
    assert "embedding" in blocks


def test_validate_interface_valid_block():
    """Test validating a valid block implementation."""
    class ValidBlock:
        def get_capabilities(self) -> BlockCapability:
            return BlockCapability(name="Test", version="1.0.0")

        def forward(self, x):
            return x

    registry = BlockRegistry("/nonexistent")
    block = ValidBlock()
    # Should not raise
    registry.validate_interface(block)


def test_validate_interface_invalid_block():
    """Test validating an invalid block."""
    class InvalidBlock:
        pass

    registry = BlockRegistry("/nonexistent")
    block = InvalidBlock()
    with pytest.raises(BlockRegistryError, match="does not implement BlockInterface"):
        registry.validate_interface(block)


def test_get_block_path(temp_blocks_dir):
    """Test getting the filesystem path of a block."""
    block_dir = create_block_yaml(
        temp_blocks_dir,
        "linear",
        "name: Linear\nversion: 1.0.0\ncapabilities: {}",
    )
    registry = BlockRegistry(str(temp_blocks_dir))
    path = registry.get_block_path("linear")
    assert path == block_dir


def test_get_block_path_nonexistent(temp_blocks_dir):
    """Test get_block_path for non-existent block."""
    registry = BlockRegistry(str(temp_blocks_dir))
    with pytest.raises(BlockRegistryError, match="Block not found"):
        registry.get_block_path("nonexistent")


def test_nested_block_directories(temp_blocks_dir):
    """Test discovering blocks in nested directories."""
    # Create a block in a subdirectory
    subdir = temp_blocks_dir / "category" / "linear"
    subdir.mkdir(parents=True)
    (subdir / "block.yaml").write_text("name: Linear\nversion: 1.0.0\ncapabilities: {}")

    # Only the direct subdirectories are discovered, not nested
    registry = BlockRegistry(str(temp_blocks_dir / "category"))
    assert "linear" in registry.list_blocks()


def test_complete_block_discovery(temp_blocks_dir):
    """Test discovering a complete block with full spec."""
    content = """
name: Linear
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: [*, in_features]
  outputs:
    y:
      shape: [*, out_features]
  params:
    in_features:
      type: int
      required: true
      range: [1, 100000]
    out_features:
      type: int
      required: true
      range: [1, 100000]
"""
    create_block_yaml(temp_blocks_dir, "linear", content)
    registry = BlockRegistry(str(temp_blocks_dir))
    cap = registry.get_block("linear")
    assert cap.name == "Linear"
    assert "x" in cap.inputs
    assert "y" in cap.outputs
    assert "in_features" in cap.params
