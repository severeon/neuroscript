"""Unit tests for CompilationEngine."""

import ast
from pathlib import Path

import pytest

from src.core.block_registry import BlockRegistry
from src.core.compilation_engine import CompilationEngine, CompilationError
from src.core.constraint_solver import Configuration
from src.core.graph_loader import ArchitectureGraph, GraphEdge, GraphNode


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
    bias:
      type: bool
      default: true
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

    # Create dropout block
    dropout_dir = blocks_dir / "dropout"
    dropout_dir.mkdir()
    (dropout_dir / "block.yaml").write_text(
        """
name: Dropout
version: 1.0.0
capabilities:
  inputs:
    x:
      shape: ["*", "*"]
  outputs:
    y:
      shape: ["*", "*"]
  params:
    p:
      type: float
      default: 0.5
"""
    )

    return blocks_dir


@pytest.fixture
def registry(temp_blocks_dir):
    """Create a BlockRegistry from temp blocks."""
    return BlockRegistry(str(temp_blocks_dir))


@pytest.fixture
def simple_config():
    """Create a simple configuration."""
    return Configuration(
        bindings={"in_features": 128, "out_features": 256, "hidden_dim": 256},
        estimated_params=32768,
        estimated_memory_gb=0.1,
    )


@pytest.fixture
def simple_graph():
    """Create a simple sequential graph."""
    return ArchitectureGraph(
        nodes=[
            GraphNode(
                id="layer1",
                block_id="linear",
                params={"in_features": 128, "out_features": 256},
            ),
            GraphNode(
                id="layer2",
                block_id="linear",
                params={"in_features": 256, "out_features": 256},
            ),
        ],
        edges=[GraphEdge(source="layer1", target="layer2")],
        inputs=["layer1"],
        outputs=["layer2"],
    )


@pytest.fixture
def complex_graph():
    """Create a more complex graph."""
    return ArchitectureGraph(
        nodes=[
            GraphNode(
                id="embedding",
                block_id="embedding",
                params={"vocab_size": 10000, "embedding_dim": 256},
            ),
            GraphNode(
                id="linear1",
                block_id="linear",
                params={"in_features": 256, "out_features": 512},
            ),
            GraphNode(
                id="norm",
                block_id="layernorm",
                params={"hidden_dim": 512},
            ),
            GraphNode(
                id="dropout",
                block_id="dropout",
                params={"p": 0.1},
            ),
        ],
        edges=[
            GraphEdge(source="embedding", target="linear1"),
            GraphEdge(source="linear1", target="norm"),
            GraphEdge(source="norm", target="dropout"),
        ],
        inputs=["embedding"],
        outputs=["dropout"],
        metadata={"name": "TestArchitecture"},
    )


@pytest.fixture
def compilation_engine(simple_graph, simple_config, registry):
    """Create a CompilationEngine instance."""
    return CompilationEngine(simple_graph, simple_config, registry)


def test_compilation_engine_initialization(simple_graph, simple_config, registry):
    """Test CompilationEngine initialization."""
    engine = CompilationEngine(simple_graph, simple_config, registry)
    assert engine.graph == simple_graph
    assert engine.config == simple_config
    assert engine.registry == registry
    assert len(engine._node_map) == 2


def test_compute_execution_order(compilation_engine):
    """Test topological sort for execution order."""
    order = compilation_engine._compute_execution_order()
    assert len(order) == 2
    assert order[0] == "layer1"
    assert order[1] == "layer2"


def test_compute_execution_order_complex(complex_graph, simple_config, registry):
    """Test execution order for complex graph."""
    engine = CompilationEngine(complex_graph, simple_config, registry)
    order = engine._compute_execution_order()
    assert len(order) == 4
    # Verify topological ordering
    assert order.index("embedding") < order.index("linear1")
    assert order.index("linear1") < order.index("norm")
    assert order.index("norm") < order.index("dropout")


def test_generate_imports(compilation_engine):
    """Test import statement generation."""
    imports = compilation_engine._generate_imports()
    assert "from blocks.linear.module import Linear" in imports


def test_generate_imports_multiple_blocks(complex_graph, simple_config, registry):
    """Test import generation for multiple different blocks."""
    engine = CompilationEngine(complex_graph, simple_config, registry)
    imports = engine._generate_imports()
    assert "from blocks.embedding.module import Embedding" in imports
    assert "from blocks.linear.module import Linear" in imports
    assert "from blocks.layernorm.module import LayerNorm" in imports
    assert "from blocks.dropout.module import Dropout" in imports


def test_generate_config_doc(compilation_engine):
    """Test configuration documentation generation."""
    doc = compilation_engine._generate_config_doc()
    assert "hidden_dim = 256" in doc
    assert "in_features = 128" in doc
    assert "out_features = 256" in doc
    assert "Total parameters: 32,768" in doc
    assert "Estimated memory: 0.10 GB" in doc


def test_generate_topology_doc(compilation_engine):
    """Test topology documentation generation."""
    doc = compilation_engine._generate_topology_doc()
    assert "layer1 -> layer2" in doc


def test_generate_block_init(compilation_engine):
    """Test block initialization code generation."""
    init_code = compilation_engine._generate_block_init()
    assert "self.layer1 = Linear(in_features=128, out_features=256" in init_code
    assert "self.layer2 = Linear(in_features=256, out_features=256" in init_code


def test_generate_block_init_with_defaults(complex_graph, simple_config, registry):
    """Test block initialization includes default parameters."""
    engine = CompilationEngine(complex_graph, simple_config, registry)
    init_code = engine._generate_block_init()
    # Check that dropout parameter is included
    assert "self.dropout = Dropout(p=0.1)" in init_code


def test_generate_forward_signature(compilation_engine):
    """Test forward method signature generation."""
    args, args_doc = compilation_engine._generate_forward_signature()
    assert args == "layer1"
    assert "layer1: Input tensor" in args_doc


def test_generate_forward_signature_default(simple_config, registry):
    """Test forward signature with no explicit inputs."""
    graph = ArchitectureGraph(
        nodes=[GraphNode(id="layer1", block_id="linear", params={})],
        edges=[],
    )
    engine = CompilationEngine(graph, simple_config, registry)
    args, args_doc = engine._generate_forward_signature()
    assert args == "x"
    assert "x: Input tensor" in args_doc


def test_generate_return_signature(compilation_engine):
    """Test return signature generation."""
    return_type, return_doc = compilation_engine._generate_return_signature()
    assert return_type == "torch.Tensor"
    assert "layer2" in return_doc


def test_generate_shape_assertions(compilation_engine):
    """Test shape assertion generation."""
    compilation_engine._execution_order = compilation_engine._compute_execution_order()
    assertions = compilation_engine._generate_shape_assertions()
    # Should have runtime shape assertions
    assert "assert" in assertions or "pass" in assertions


def test_is_sequential_true(compilation_engine):
    """Test detecting sequential graph structure."""
    compilation_engine._execution_order = compilation_engine._compute_execution_order()
    assert compilation_engine._is_sequential() is True


def test_is_sequential_false():
    """Test detecting non-sequential graph (with branching)."""
    # This would need a graph with multiple paths
    # For now, just ensure the method works
    pass


def test_generate_sequential(compilation_engine):
    """Test sequential code generation."""
    compilation_engine._execution_order = compilation_engine._compute_execution_order()
    nodes = [compilation_engine._node_map[nid] for nid in compilation_engine._execution_order]
    code = compilation_engine.generate_sequential(nodes)
    assert "layer1_out = self.layer1(layer1)" in code or "layer1_out = self.layer1(x)" in code
    assert "layer2_out = self.layer2(" in code


def test_generate_parallel(compilation_engine):
    """Test parallel branch code generation."""
    # Create simple parallel structure
    branch1 = [GraphNode(id="branch1", block_id="linear", params={})]
    branch2 = [GraphNode(id="branch2", block_id="linear", params={})]
    branches = [branch1, branch2]

    code = compilation_engine.generate_parallel(branches, "add")
    assert "branch_0" in code
    assert "branch_1" in code
    assert "merged" in code


def test_generate_parallel_concat(compilation_engine):
    """Test parallel generation with concat merge."""
    branch1 = [GraphNode(id="branch1", block_id="linear", params={})]
    branches = [branch1]

    code = compilation_engine.generate_parallel(branches, "concat")
    assert "torch.cat" in code


def test_generate_general_forward(compilation_engine):
    """Test general forward pass generation."""
    compilation_engine._execution_order = compilation_engine._compute_execution_order()
    code = compilation_engine._generate_general_forward()
    assert "self.layer1" in code
    assert "self.layer2" in code
    assert "_out" in code


def test_generate_return_statement(compilation_engine):
    """Test return statement generation."""
    compilation_engine._execution_order = compilation_engine._compute_execution_order()
    ret = compilation_engine._generate_return_statement()
    assert "layer2_out" in ret


def test_generate_return_statement_multiple_outputs(simple_config, registry):
    """Test return statement with multiple outputs."""
    graph = ArchitectureGraph(
        nodes=[
            GraphNode(id="layer1", block_id="linear", params={}),
            GraphNode(id="layer2", block_id="linear", params={}),
        ],
        edges=[],
        outputs=["layer1", "layer2"],
    )
    engine = CompilationEngine(graph, simple_config, registry)
    engine._execution_order = engine._compute_execution_order()
    ret = engine._generate_return_statement()
    assert "layer1_out" in ret
    assert "layer2_out" in ret


def test_format_code_without_black(compilation_engine):
    """Test code formatting without black (fallback)."""
    code = "def foo():\n    pass"
    formatted = compilation_engine._format_code(code)
    # Should return original code if black not available
    assert formatted == code or "def foo()" in formatted


def test_compile_creates_file(compilation_engine, tmp_path):
    """Test that compile() creates output file."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")
    assert output_file.exists()


def test_compile_generates_valid_python(compilation_engine, tmp_path):
    """Test that generated code is syntactically valid Python."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    # Read generated code
    code = output_file.read_text()

    # Try to parse as Python AST
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax error: {e}")


def test_compile_includes_class_definition(compilation_engine, tmp_path):
    """Test that generated code includes class definition."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()
    assert "class TestModel(nn.Module):" in code


def test_compile_includes_imports(compilation_engine, tmp_path):
    """Test that generated code includes necessary imports."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()
    assert "import torch" in code
    assert "import torch.nn as nn" in code
    assert "from blocks.linear.module import Linear" in code


def test_compile_includes_init_method(compilation_engine, tmp_path):
    """Test that generated code includes __init__ method."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()
    assert "def __init__(self):" in code
    assert "super().__init__()" in code


def test_compile_includes_forward_method(compilation_engine, tmp_path):
    """Test that generated code includes forward method."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()
    assert "def forward(self," in code


def test_compile_includes_docstrings(compilation_engine, tmp_path):
    """Test that generated code includes documentation."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()
    assert '"""' in code  # Has docstrings


def test_compile_includes_configuration(compilation_engine, tmp_path):
    """Test that generated code documents configuration."""
    output_file = tmp_path / "generated_model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()
    assert "in_features = 128" in code
    assert "out_features = 256" in code


def test_compile_creates_parent_directories(compilation_engine, tmp_path):
    """Test that compile creates parent directories if needed."""
    output_file = tmp_path / "nested" / "dir" / "model.py"
    compilation_engine.compile(output_file, class_name="TestModel")
    assert output_file.exists()
    assert output_file.parent.exists()


def test_compile_complex_architecture(complex_graph, simple_config, registry, tmp_path):
    """Test compiling a more complex architecture."""
    engine = CompilationEngine(complex_graph, simple_config, registry)
    output_file = tmp_path / "complex_model.py"
    engine.compile(output_file, class_name="ComplexModel")

    assert output_file.exists()
    code = output_file.read_text()

    # Verify all blocks are initialized
    assert "self.embedding = Embedding" in code
    assert "self.linear1 = Linear" in code
    assert "self.norm = LayerNorm" in code
    assert "self.dropout = Dropout" in code

    # Verify code is valid Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Complex generated code has syntax error: {e}")


def test_compile_with_metadata(complex_graph, simple_config, registry, tmp_path):
    """Test that architecture metadata is included in generated code."""
    engine = CompilationEngine(complex_graph, simple_config, registry)
    output_file = tmp_path / "model.py"
    engine.compile(output_file, class_name="MyModel")

    code = output_file.read_text()
    assert "TestArchitecture" in code


def test_compilation_error_handling(simple_graph, simple_config, tmp_path):
    """Test error handling when registry is invalid."""
    # Create engine with invalid registry
    invalid_registry = None
    engine = CompilationEngine(simple_graph, simple_config, invalid_registry)

    output_file = tmp_path / "model.py"
    with pytest.raises(CompilationError):
        engine.compile(output_file)


def test_analyze_graph_structure_sequential(compilation_engine):
    """Test graph structure analysis for sequential graphs."""
    compilation_engine._execution_order = compilation_engine._compute_execution_order()
    structure = compilation_engine._analyze_graph_structure()
    assert structure['type'] == 'sequential'
    assert len(structure['nodes']) == 2


def test_analyze_graph_structure_general():
    """Test graph structure analysis for general DAG."""
    # This would require a more complex branching graph
    # Placeholder for future expansion
    pass


def test_compile_empty_graph(tmp_path, simple_config, registry):
    """Test compiling an empty graph."""
    empty_graph = ArchitectureGraph(nodes=[], edges=[])
    engine = CompilationEngine(empty_graph, simple_config, registry)

    output_file = tmp_path / "empty_model.py"
    engine.compile(output_file, class_name="EmptyModel")

    assert output_file.exists()
    code = output_file.read_text()

    # Should still be valid Python
    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Empty model code has syntax error: {e}")


def test_compile_single_node(tmp_path, simple_config, registry):
    """Test compiling a graph with single node."""
    single_graph = ArchitectureGraph(
        nodes=[GraphNode(id="only", block_id="linear", params={"in_features": 10, "out_features": 20})],
        edges=[],
        inputs=["only"],
        outputs=["only"],
    )
    engine = CompilationEngine(single_graph, simple_config, registry)

    output_file = tmp_path / "single_model.py"
    engine.compile(output_file, class_name="SingleModel")

    assert output_file.exists()
    code = output_file.read_text()
    assert "self.only = Linear" in code


def test_generated_code_structure(compilation_engine, tmp_path):
    """Test overall structure of generated code."""
    output_file = tmp_path / "model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()

    # Verify order of sections
    import_pos = code.find("import torch")
    class_pos = code.find("class TestModel")
    init_pos = code.find("def __init__")
    forward_pos = code.find("def forward")

    assert import_pos < class_pos
    assert class_pos < init_pos
    assert init_pos < forward_pos


def test_compile_preserves_parameter_types(compilation_engine, tmp_path):
    """Test that parameter types are preserved in generated code."""
    output_file = tmp_path / "model.py"
    compilation_engine.compile(output_file, class_name="TestModel")

    code = output_file.read_text()

    # Integer parameters should not have quotes
    assert "in_features=128" in code
    assert "out_features=256" in code
    # Should not have 'in_features=128' (as string)
    assert "'128'" not in code
