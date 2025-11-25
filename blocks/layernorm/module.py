"""LayerNorm block implementation for NeuroScript."""

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


class LayerNorm(nn.Module):
    """
    LayerNorm block: normalizes activations over the last dimension.

    Implements the BlockInterface protocol for NeuroScript compatibility.
    This block preserves input shapes while normalizing over the specified dimension.
    """

    def __init__(
        self,
        normalized_dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        Initialize LayerNorm block.

        Args:
            normalized_dim: Size of the dimension to normalize over
            eps: Small value for numerical stability (default: 1e-5)
            elementwise_affine: If True, include learnable affine parameters (default: True)
        """
        super().__init__()

        # Validate parameters against block specification
        if not isinstance(normalized_dim, int) or normalized_dim < 1:
            raise ValueError(f"normalized_dim must be a positive integer, got {normalized_dim}")
        if normalized_dim > 10000:
            raise ValueError(f"normalized_dim must be <= 10000, got {normalized_dim}")
        
        if not isinstance(eps, (int, float)) or eps < 0:
            raise ValueError(f"eps must be a non-negative number, got {eps}")
        
        if not isinstance(elementwise_affine, bool):
            raise ValueError(
                f"elementwise_affine must be a boolean, got {type(elementwise_affine)}"
            )

        self.normalized_dim = normalized_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Create the underlying PyTorch LayerNorm layer
        self.layer_norm = nn.LayerNorm(
            normalized_dim,
            eps=eps,
            elementwise_affine=elementwise_affine
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
        Forward pass: normalizes input over the last dimension.

        Args:
            x: Input tensor of shape [*, normalized_dim] where * represents
               any number of leading dimensions

        Returns:
            Output tensor of shape [*, normalized_dim] (same as input)
        """
        # Validate input dtype
        if x.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise ValueError(
                f"Input must have dtype float32, float16, or bfloat16, got {x.dtype}"
            )

        # Validate input shape - last dimension must match normalized_dim
        if x.shape[-1] != self.normalized_dim:
            raise ValueError(
                f"Input last dimension must be {self.normalized_dim}, "
                f"got shape {x.shape} with last dimension {x.shape[-1]}"
            )

        # Apply layer normalization
        output = self.layer_norm(x)

        # Verify shape preservation (output shape == input shape)
        assert output.shape == x.shape, (
            f"Shape preservation violated: input shape {x.shape} != output shape {output.shape}"
        )

        return output

    def __repr__(self) -> str:
        """String representation of the block."""
        return (
            f"LayerNorm(normalized_dim={self.normalized_dim}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine})"
        )
