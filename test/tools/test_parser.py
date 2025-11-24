import pytest
from src.tools.parser import parse_mermaid_flow


with open("src/blocks/apoptotic_model.mmd") as f:
    trm = f.read()

with open("src/blocks/tiny_recursive_mamba.mmd") as f:
    apop = f.read()


def test_basic():
    result = parse_mermaid_flow(trm)
    nodes = result["nodes"]

    assert result is not None

    # 'MatMul', 'Input', 'ApopLayer1', 'HiddenLayers',
    # 'Mamba1', 'TRMamba', 'Mamba2', 'LargeOutput', 'SmallOutput'
    assert len(nodes) is 9


def test_bool_param_parsing():
    """boolTrue should be parsed as if it was boolTrue=True"""
    result = parse_mermaid_flow(trm)
    nodes = result["nodes"]

    assert result is not None

    # MatMul
    assert nodes[0]["params"]
    assert not(isinstance(nodes[0]["params"], str)) 
    assert isinstance(nodes[0]["params"]["boolTrue"], bool)
