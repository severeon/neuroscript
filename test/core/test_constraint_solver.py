"""Unit tests for ConstraintSolver."""

from pathlib import Path

import pytest

from src.core.block_registry import BlockRegistry
from src.core.constraint_solver import (
    Configuration,
    Constraint,
    ConstraintSolver,
    ConstraintSolverError,
)
from src.core.graph_loader import ArchitectureGraph, GraphLoader


@pytest.fixture
def temp_blocks_dir(tmp_path):
    """Create a temporary blocks directory with test blocks."""
    blocks_dir = tmp_path / "blocks"
    blocks_dir.mkdir()

    # Create linear block with constraints
    linear_dir = blocks_dir / "linear"
    linear_dir.mkdir()
    (linear_dir / "block.yaml").write_text(
        """
name: Linear
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: [batch, in_features]
      dtype: [float32, float16]
  outputs:
    y:
      shape: [batch, out_features]
      dtype: [float32, float16]
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
    - batch >= 1
    - in_features % 8 == 0
    - out_features % 8 == 0
"""
    )

    # Create transformer block with complex constraints
    transformer_dir = blocks_dir / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "block.yaml").write_text(
        """
name: Transformer
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: [batch, seq, dim]
      dtype: [float32, float16]
  outputs:
    y:
      shape: [batch, seq, dim]
      dtype: [float32, float16]
  params:
    dim:
      type: int
      required: false
      default: 512
    heads:
      type: int
      required: false
      default: 8
    layers:
      type: int
      range: [1, 24]
      default: 6
  constraints:
    - dim in [256, 512, 768, 1024]
    - heads in [4, 8, 12, 16]
    - seq >= 1
"""
    )

    # Create block with unsatisfiable constraints
    conflicting_dir = blocks_dir / "conflicting"
    conflicting_dir.mkdir()
    (conflicting_dir / "block.yaml").write_text(
        """
name: Conflicting
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: [batch, dim]
  outputs:
    y:
      shape: [batch, dim]
  params:
    dim:
      type: int
      required: false
      default: 256
  constraints:
    - dim == 256
    - dim == 512
"""
    )

    # Create block with modulo constraint
    attention_dir = blocks_dir / "attention"
    attention_dir.mkdir()
    (attention_dir / "block.yaml").write_text(
        """
name: Attention
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: [batch, seq, dim]
  outputs:
    y:
      shape: [batch, seq, dim]
  params:
    dim:
      type: int
      required: false
      default: 512
    heads:
      type: int
      required: false
      default: 8
  constraints:
    - dim % 64 == 0
    - heads in [8, 16]
"""
    )

    return blocks_dir


@pytest.fixture
def registry(temp_blocks_dir):
    """Create a BlockRegistry from temp blocks."""
    return BlockRegistry(str(temp_blocks_dir))


@pytest.fixture
def loader(registry):
    """Create a GraphLoader."""
    return GraphLoader(registry)


def test_constraint_solver_initialization(registry):
    """Test ConstraintSolver initialization."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    assert solver.graph == graph
    assert solver.registry == registry
    assert solver.constraints == []
    assert solver.dimensions == set()


def test_parse_enum_constraint(registry):
    """Test parsing enum constraints."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    constraint = solver._parse_constraint("dim in [256, 512, 768]")

    assert constraint.constraint_type == "enum"
    assert constraint.dimension == "dim"
    assert constraint.values == [256, 512, 768]


def test_parse_modulo_constraint(registry):
    """Test parsing modulo constraints."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    constraint = solver._parse_constraint("dim % 8 == 0")

    assert constraint.constraint_type == "modulo"
    assert constraint.dimension == "dim"
    assert constraint.modulo == (8, 0)


def test_parse_equality_constraint(registry):
    """Test parsing equality constraints."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    constraint = solver._parse_constraint("dim == 512")

    assert constraint.constraint_type == "equality"
    assert constraint.dimension == "dim"
    assert constraint.values == [512]


def test_parse_range_constraint(registry):
    """Test parsing range constraints."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    # Test >= constraint
    constraint = solver._parse_constraint("batch >= 1")
    assert constraint.constraint_type == "range"
    assert constraint.dimension == "batch"
    assert constraint.min_value == 1
    assert constraint.max_value is None

    # Test <= constraint
    constraint = solver._parse_constraint("layers <= 24")
    assert constraint.constraint_type == "range"
    assert constraint.dimension == "layers"
    assert constraint.min_value is None
    assert constraint.max_value == 24


def test_solve_simple_architecture(registry, loader, tmp_path):
    """Test solving constraints for a simple architecture."""
    arch_file = tmp_path / "arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  layer1:
    block: linear
    params:
      in_features: 256
      out_features: 512

topology:
  - input -> layer1
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    configs = solver.solve(max_configs=10)

    assert len(configs) > 0
    assert all(isinstance(config, Configuration) for config in configs)


def test_solve_transformer_architecture(registry, loader, tmp_path):
    """Test solving constraints for transformer architecture."""
    arch_file = tmp_path / "transformer_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  encoder:
    block: transformer
    params:
      layers: 6

topology:
  - input -> encoder
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    configs = solver.solve(max_configs=50)

    assert len(configs) > 0

    # Check that all configurations have valid dimension bindings
    for config in configs:
        if "dim" in config.bindings:
            assert config.bindings["dim"] in [256, 512, 768, 1024]
        if "heads" in config.bindings:
            assert config.bindings["heads"] in [4, 8, 12, 16]


def test_check_satisfiable_valid(registry, loader, tmp_path):
    """Test check_satisfiable for valid architecture."""
    arch_file = tmp_path / "valid_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  layer1:
    block: linear
    params:
      in_features: 256
      out_features: 512

topology:
  - input -> layer1
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    satisfiable, unsatisfiable = solver.check_satisfiable()

    assert satisfiable is True
    assert len(unsatisfiable) == 0


def test_check_satisfiable_invalid(registry, loader, tmp_path):
    """Test check_satisfiable for conflicting constraints."""
    arch_file = tmp_path / "conflicting_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  layer1:
    block: conflicting
    params: {}

topology:
  - input -> layer1
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    satisfiable, unsatisfiable = solver.check_satisfiable()

    assert satisfiable is False
    assert len(unsatisfiable) > 0


def test_detect_conflicts(registry, loader, tmp_path):
    """Test conflict detection."""
    arch_file = tmp_path / "conflicting_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  layer1:
    block: conflicting
    params: {}

topology:
  - input -> layer1
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    # Collect constraints first
    solver._collect_constraints()
    solver._collected = True

    conflicts = solver.detect_conflicts()

    assert len(conflicts) > 0
    # Should detect conflict between dim == 256 and dim == 512


def test_apply_configuration(registry):
    """Test applying configuration."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    config = Configuration(bindings={"dim": 512, "batch": 32})
    solver.apply_configuration(config)

    assert solver.dimension_domains["dim"] == {512}
    assert solver.dimension_domains["batch"] == {32}


def test_constraint_propagation(registry, loader, tmp_path):
    """Test constraint propagation reduces domains."""
    arch_file = tmp_path / "attention_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  attn:
    block: attention
    params: {}

topology:
  - input -> attn
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    # Collect constraints and initialize domains
    solver._collect_constraints()
    solver._collected = True
    solver._initialize_domains()

    # Check initial domain size
    if "dim" in solver.dimension_domains:
        initial_size = len(solver.dimension_domains["dim"])

        # Apply constraint propagation
        solver._propagate_constraints()

        # Domain should be reduced by dim % 64 == 0 constraint
        final_size = len(solver.dimension_domains["dim"])
        assert final_size < initial_size


def test_configuration_with_modulo_constraint(registry, loader, tmp_path):
    """Test that configurations satisfy modulo constraints."""
    arch_file = tmp_path / "attention_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  attn:
    block: attention
    params: {}

topology:
  - input -> attn
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    configs = solver.solve(max_configs=20)

    # All dim values should be divisible by 64
    for config in configs:
        if "dim" in config.bindings:
            assert config.bindings["dim"] % 64 == 0


def test_resource_estimation(registry, loader, tmp_path):
    """Test resource estimation for configurations."""
    arch_file = tmp_path / "arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  layer1:
    block: linear
    params:
      in_features: 256
      out_features: 512

topology:
  - input -> layer1
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    configs = solver.solve(max_configs=5)

    for config in configs:
        # Check that estimates are computed
        assert config.estimated_params >= 0
        assert config.estimated_memory_gb >= 0.0


def test_empty_graph(registry):
    """Test solver with empty graph."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    configs = solver.solve()

    # Empty graph should have one empty configuration
    assert len(configs) == 1
    assert configs[0].bindings == {}


def test_multiple_nodes(registry, loader, tmp_path):
    """Test solver with multiple connected nodes."""
    arch_file = tmp_path / "multi_node_arch.yaml"
    arch_file.write_text(
        """
components:
  layer1:
    block: linear
    params:
      in_features: 256
      out_features: 512
  layer2:
    block: linear
    params:
      in_features: 512
      out_features: 256

topology:
  - layer1 -> layer2
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    configs = solver.solve(max_configs=10)

    assert len(configs) > 0


def test_max_configs_limit(registry, loader, tmp_path):
    """Test that max_configs limits the number of configurations."""
    arch_file = tmp_path / "transformer_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  encoder:
    block: transformer
    params:
      layers: 6

topology:
  - input -> encoder
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    # Request only 5 configurations
    configs = solver.solve(max_configs=5)

    assert len(configs) <= 5


def test_configuration_repr():
    """Test Configuration string representation."""
    config = Configuration(
        bindings={"dim": 512, "batch": 32}, estimated_params=1000000, estimated_memory_gb=1.5
    )

    repr_str = repr(config)

    assert "batch=32" in repr_str
    assert "dim=512" in repr_str
    assert "params=1,000,000" in repr_str
    assert "memory=1.50GB" in repr_str


def test_constraints_conflict_enum(registry):
    """Test conflict detection for enum constraints."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    c1 = Constraint(
        expression="dim in [256, 512]",
        dimension="dim",
        constraint_type="enum",
        values=[256, 512],
    )

    c2 = Constraint(
        expression="dim in [768, 1024]",
        dimension="dim",
        constraint_type="enum",
        values=[768, 1024],
    )

    # These constraints conflict (no overlap)
    assert solver._constraints_conflict(c1, c2) is True


def test_constraints_no_conflict_enum(registry):
    """Test no conflict for overlapping enum constraints."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    c1 = Constraint(
        expression="dim in [256, 512]",
        dimension="dim",
        constraint_type="enum",
        values=[256, 512],
    )

    c2 = Constraint(
        expression="dim in [512, 768]",
        dimension="dim",
        constraint_type="enum",
        values=[512, 768],
    )

    # These constraints don't conflict (512 is in both)
    assert solver._constraints_conflict(c1, c2) is False


def test_constraints_conflict_equality(registry):
    """Test conflict detection for equality constraints."""
    graph = ArchitectureGraph()
    solver = ConstraintSolver(graph, registry)

    c1 = Constraint(
        expression="dim == 256", dimension="dim", constraint_type="equality", values=[256]
    )

    c2 = Constraint(
        expression="dim == 512", dimension="dim", constraint_type="equality", values=[512]
    )

    # These constraints conflict
    assert solver._constraints_conflict(c1, c2) is True


def test_solve_raises_on_unsatisfiable(registry, loader, tmp_path):
    """Test that solve raises error for unsatisfiable constraints."""
    arch_file = tmp_path / "conflicting_arch.yaml"
    arch_file.write_text(
        """
components:
  input:
    block: linear
    params:
      in_features: 256
      out_features: 256
  layer1:
    block: conflicting
    params: {}

topology:
  - input -> layer1
"""
    )

    graph = loader.load(str(arch_file))
    solver = ConstraintSolver(graph, registry)

    with pytest.raises(ConstraintSolverError) as exc_info:
        solver.solve()

    assert "Unsatisfiable" in str(exc_info.value)
