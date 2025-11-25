"""Dropout block implementation for NeuroScript."""

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# Import from src/core
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.block_interface import BlockCapability
from src.core.capability_parser import CapabilityParser


class Dropout(nn.Module):
    """
    Dropout block: randomly zeroes elements during training for regularization.

    Implements the BlockInterface protocol for NeuroScript compatibility.
    This block preserves input shapes while applying dropout regularization.
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False
    ):
        """
        Initialize Dropout block.

        Args:
            p: Probability of an element to be zeroed (default: 0.5)
            inplace: If True, will do this operation in-place (default: False)
        """
        super().__init__()

        # Validate parameters against block specification
        if not isinstance(p, (int, float)):
            raise ValueError(f"p must be a number, got {type(p)}")
        if p < 0.0 or p > 1.0:
            raise ValueError(f"p must be in range [0.0, 1.0], got {p}")
        
        if not isinstance(inplace, bool):
            raise ValueError(
                f"inplace must be a boolean, got {type(inplace)}"
            )

        self.p = p
        self.inplace = inplace

        # Create the underlying PyTorch Dropout layer
        self.dropout = nn.Dropout(
            p=p,
            inplace=inplace
        )

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
        Forward pass: applies dropout to input tensor.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor of the same shape as input
        """
        # Validate input dtype
        if x.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise ValueError(
                f"Input must have dtype float32, float16, or bfloat16, got {x.dtype}"
            )

        # Store original shape for verification
        original_shape = x.shape

        # Apply dropout
        output = self.dropout(x)

        # Verify shape preservation (output shape == input shape)
        assert output.shape == original_shape, (
            f"Shape preservation violated: input shape {original_shape} != output shape {output.shape}"
        )

        return output

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"Dropout(p={self.p}, "
            f"inplace={self.inplace})"
        )
