"""Unit tests for GraphLoader."""

from pathlib import Path

import pytest

from src.core.block_registry import BlockRegistry
from src.core.graph_loader import (
    ArchitectureGraph,
    GraphEdge,
    GraphLoader,
    GraphLoaderError,
    GraphNode,
)


@pytest.fixture
def temp_blocks_dir(tmp_path):
    """Create a temporary blocks directory with test blocks."""
    blocks_dir = tmp_path / "blocks"
    blocks_dir.mkdir()

    # Create linear block
    linear_dir = blocks_dir / "linear"
    linear_dir.mkdir()
    (linear_dir / "block.yaml").write_text(
        """
name: Linear
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: ["*", in_features]
  outputs:
    y:
      shape: ["*", out_features]
  params:
    in_features:
      type: int
      required: true
    out_features:
      type: int
      required: true
"""
    )

    # Create embedding block
    embedding_dir = blocks_dir / "embedding"
    embedding_dir.mkdir()
    (embedding_dir / "block.yaml").write_text(
        """
name: Embedding
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: ["*", vocab_size]
  outputs:
    y:
      shape: ["*", embedding_dim]
  params:
    vocab_size:
      type: int
      required: true
    embedding_dim:
      type: int
      required: true
"""
    )

    # Create layernorm block
    ln_dir = blocks_dir / "layernorm"
    ln_dir.mkdir()
    (ln_dir / "block.yaml").write_text(
        """
name: LayerNorm
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: ["*", hidden_dim]
  outputs:
    y:
      shape: ["*", hidden_dim]
  params:
    hidden_dim:
      type: int
      required: true
"""
    )

    return blocks_dir


@pytest.fixture
def registry(temp_blocks_dir):
    """Create a BlockRegistry from temp blocks."""
    return BlockRegistry(str(temp_blocks_dir))


@pytest.fixture
def loader(registry):
    """Create a GraphLoader with registry."""
    return GraphLoader(registry)


def create_arch_yaml(tmp_path, content: str) -> Path:
    """Helper to create an architecture YAML file."""
    arch_file = tmp_path / "architecture.yaml"
    arch_file.write_text(content)
    return arch_file


def test_graph_node_creation():
    """Test GraphNode dataclass creation."""
    node = GraphNode(id="layer1", block_id="linear", params={"in_features": 10})
    assert node.id == "layer1"
    assert node.block_id == "linear"
    assert node.params["in_features"] == 10


def test_graph_edge_creation():
    """Test GraphEdge dataclass creation."""
    edge = GraphEdge(source="layer1", target="layer2")
    assert edge.source == "layer1"
    assert edge.target == "layer2"
    assert edge.source_output == "y"
    assert edge.target_input == "x"


def test_graph_edge_custom_ports():
    """Test GraphEdge with custom port names."""
    edge = GraphEdge(
        source="layer1",
        target="layer2",
        source_output="hidden",
        target_input="embedding",
    )
    assert edge.source_output == "hidden"
    assert edge.target_input == "embedding"


def test_architecture_graph_creation():
    """Test ArchitectureGraph dataclass creation."""
    node1 = GraphNode(id="input", block_id="linear")
    node2 = GraphNode(id="output", block_id="linear")
    edge = GraphEdge(source="input", target="output")

    graph = ArchitectureGraph(nodes=[node1, node2], edges=[edge])
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1


def test_load_file_not_found(loader, tmp_path):
    """Test loading non-existent file."""
    with pytest.raises(GraphLoaderError, match="not found"):
        loader.load("/nonexistent/path/architecture.yaml")


def test_load_malformed_yaml(loader, tmp_path):
    """Test loading malformed YAML."""
    arch_file = create_arch_yaml(tmp_path, "invalid: yaml: content:")
    with pytest.raises(GraphLoaderError, match="Malformed YAML"):
        loader.load(str(arch_file))


def test_load_not_dict(loader, tmp_path):
    """Test loading YAML that's not a dict."""
    arch_file = create_arch_yaml(tmp_path, "- item1\n- item2")
    with pytest.raises(GraphLoaderError, match="Expected YAML dict"):
        loader.load(str(arch_file))


def test_load_minimal_architecture(loader, tmp_path):
    """Test loading minimal valid architecture."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
topology: []
"""
    arch_file = create_arch_yaml(tmp_path, content)
    graph = loader.load(str(arch_file))
    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "layer1"
    assert graph.nodes[0].block_id == "linear"


def test_load_shorthand_component(loader, tmp_path):
    """Test loading components with shorthand syntax."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
topology: []
"""
    arch_file = create_arch_yaml(tmp_path, content)
    graph = loader.load(str(arch_file))
    assert graph.nodes[0].block_id == "linear"


def test_parse_topology_simple_chain(loader):
    """Test parsing simple topology chain."""
    edges = loader.parse_topology("A -> B")
    assert len(edges) == 1
    assert edges[0].source == "A"
    assert edges[0].target == "B"
    assert edges[0].source_output == "y"
    assert edges[0].target_input == "x"


def test_parse_topology_long_chain(loader):
    """Test parsing longer topology chain."""
    edges = loader.parse_topology("A -> B -> C -> D")
    assert len(edges) == 3
    assert edges[0].source == "A"
    assert edges[1].source == "B"
    assert edges[2].source == "C"
    assert edges[2].target == "D"


def test_parse_topology_with_port_names(loader):
    """Test parsing topology with custom port names."""
    edges = loader.parse_topology("A.output -> B.input")
    assert len(edges) == 1
    assert edges[0].source_output == "output"
    assert edges[0].target_input == "input"


def test_parse_topology_mixed_ports(loader):
    """Test parsing topology with mixed port specifications."""
    edges = loader.parse_topology("A.hidden -> B -> C.feat")
    assert len(edges) == 2
    assert edges[0].source_output == "hidden"
    assert edges[0].target_input == "x"
    assert edges[1].source_output == "y"
    assert edges[1].target_input == "feat"


def test_parse_topology_invalid_no_arrow(loader):
    """Test parsing invalid topology without arrow."""
    with pytest.raises(GraphLoaderError, match="Invalid topology format"):
        loader.parse_topology("A B C")


def test_parse_topology_empty_nodes(loader):
    """Test parsing topology with empty node names."""
    with pytest.raises(GraphLoaderError, match="cannot be empty"):
        loader.parse_topology("A -> ")


def test_validate_references_valid(loader, tmp_path):
    """Test validating references with valid blocks."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
  layer2:
    block: embedding
    params:
      vocab_size: 100
      embedding_dim: 50
topology: []
"""
    arch_file = create_arch_yaml(tmp_path, content)
    graph = loader.load(str(arch_file))
    # load() includes validation, so no error should be raised
    assert len(graph.nodes) == 2


def test_validate_references_invalid_block(loader, tmp_path):
    """Test validating references with non-existent block."""
    content = """
components:
  layer1:
    block: nonexistent_block
    params: {}
topology: []
"""
    arch_file = create_arch_yaml(tmp_path, content)
    with pytest.raises(GraphLoaderError, match="Block.*not found"):
        loader.load(str(arch_file))


def test_validate_parameters_unknown_param(loader, tmp_path):
    """Test validating parameters with unknown param."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
      unknown_param: 123
topology: []
"""
    arch_file = create_arch_yaml(tmp_path, content)
    with pytest.raises(GraphLoaderError, match="Unknown parameter"):
        loader.load(str(arch_file))


def test_validate_parameters_missing_required(loader, tmp_path):
    """Test validating parameters with missing required param."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
topology: []
"""
    arch_file = create_arch_yaml(tmp_path, content)
    with pytest.raises(GraphLoaderError, match="Missing required parameter"):
        loader.load(str(arch_file))


def test_detect_cycles_simple_cycle(loader, tmp_path):
    """Test detecting simple cycle."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
  layer2:
    block: linear
    params:
      in_features: 20
      out_features: 10
topology:
  - layer1 -> layer2 -> layer1
"""
    arch_file = create_arch_yaml(tmp_path, content)
    with pytest.raises(GraphLoaderError, match="Cycle detected"):
        loader.load(str(arch_file))


def test_detect_cycles_self_loop(loader, tmp_path):
    """Test detecting self-loop."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
topology:
  - layer1 -> layer1
"""
    arch_file = create_arch_yaml(tmp_path, content)
    with pytest.raises(GraphLoaderError, match="Cycle detected"):
        loader.load(str(arch_file))


def test_detect_cycles_no_cycle(loader, tmp_path):
    """Test architecture without cycles."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
  layer2:
    block: linear
    params:
      in_features: 20
      out_features: 30
  layer3:
    block: linear
    params:
      in_features: 30
      out_features: 40
topology:
  - layer1 -> layer2 -> layer3
"""
    arch_file = create_arch_yaml(tmp_path, content)
    # Should not raise
    graph = loader.load(str(arch_file))
    assert len(graph.nodes) == 3


def test_complex_valid_architecture(loader, tmp_path):
    """Test loading a complex valid architecture."""
    content = """
components:
  embedding:
    block: embedding
    params:
      vocab_size: 10000
      embedding_dim: 512
  layer1:
    block: linear
    params:
      in_features: 512
      out_features: 1024
  layer2:
    block: linear
    params:
      in_features: 1024
      out_features: 512
  norm:
    block: layernorm
    params:
      hidden_dim: 512
topology:
  - embedding -> layer1 -> layer2 -> norm
inputs:
  - embedding
outputs:
  - norm
"""
    arch_file = create_arch_yaml(tmp_path, content)
    graph = loader.load(str(arch_file))
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
    assert graph.inputs == ["embedding"]
    assert graph.outputs == ["norm"]


def test_topology_as_list(loader, tmp_path):
    """Test topology specified as list."""
    content = """
components:
  layer1:
    block: linear
    params:
      in_features: 10
      out_features: 20
  layer2:
    block: linear
    params:
      in_features: 20
      out_features: 30
topology:
  - layer1 -> layer2
"""
    arch_file = create_arch_yaml(tmp_path, content)
    graph = loader.load(str(arch_file))
    assert len(graph.edges) == 1


def test_validate_graph_io_valid(loader):
    """Test validating valid graph I/O."""
    graph = ArchitectureGraph(
        nodes=[
            GraphNode(id="input", block_id="linear"),
            GraphNode(id="output", block_id="linear"),
        ],
        inputs=["input"],
        outputs=["output"],
    )
    # Should not raise
    loader.validate_graph_io(graph)


def test_validate_graph_io_invalid_input(loader):
    """Test validating with non-existent input node."""
    graph = ArchitectureGraph(
        nodes=[GraphNode(id="layer1", block_id="linear")],
        inputs=["nonexistent"],
    )
    with pytest.raises(GraphLoaderError, match="Input node.*not found"):
        loader.validate_graph_io(graph)


def test_validate_graph_io_invalid_output(loader):
    """Test validating with non-existent output node."""
    graph = ArchitectureGraph(
        nodes=[GraphNode(id="layer1", block_id="linear")],
        outputs=["nonexistent"],
    )
    with pytest.raises(GraphLoaderError, match="Output node.*not found"):
        loader.validate_graph_io(graph)
