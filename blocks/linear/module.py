"""Linear block implementation for NeuroScript."""

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Import from src/core
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.block_interface import BlockCapability
from src.core.capability_parser import CapabilityParser


class Linear(nn.Module):
    """
    Linear transformation block: y = xW^T + b

    Implements the BlockInterface protocol for NeuroScript compatibility.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize Linear block.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, adds a learnable bias to the output
        """
        super().__init__()

        # Validate parameters against block specification
        if not isinstance(in_features, int) or in_features < 1:
            raise ValueError(f"in_features must be a positive integer, got {in_features}")
        if not isinstance(out_features, int) or out_features < 1:
            raise ValueError(f"out_features must be a positive integer, got {out_features}")
        if in_features > 100000:
            raise ValueError(f"in_features must be <= 100000, got {in_features}")
        if out_features > 100000:
            raise ValueError(f"out_features must be <= 100000, got {out_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Create the underlying PyTorch linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def get_capabilities(self) -> BlockCapability:
        """
        Return the capability specification of this block.

        Loads and parses the block.yaml file in the same directory.

        Returns:
            BlockCapability: Block's input/output shapes, parameters, and constraints
        """
        # Get the directory where this module lives
        block_dir = Path(__file__).parent
        block_yaml_path = block_dir / "block.yaml"

        # Use CapabilityParser to load the specification
        parser = CapabilityParser()
        return parser.parse_file(str(block_yaml_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: applies linear transformation.

        Args:
            x: Input tensor of shape [*, in_features] where * can be any batch dimensions

        Returns:
            Output tensor of shape [*, out_features]
        """
        # Validate input shape
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: expected {self.in_features}, "
                f"got {x.shape[-1]}"
            )

        return self.linear(x)

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.has_bias})"
        )
