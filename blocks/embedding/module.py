"""Embedding block implementation for NeuroScript."""

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


class Embedding(nn.Module):
    """
    Embedding block: transforms discrete token indices to dense vectors.

    Implements the BlockInterface protocol for NeuroScript compatibility.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Initialize Embedding block.

        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            embedding_dim: Dimension of the embedding vectors
            padding_idx: If specified, entries at padding_idx do not contribute to gradient
        """
        super().__init__()

        # Validate parameters against block specification
        if not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim}")
        if vocab_size > 1000000:
            raise ValueError(f"vocab_size must be <= 1000000, got {vocab_size}")
        if embedding_dim > 10000:
            raise ValueError(f"embedding_dim must be <= 10000, got {embedding_dim}")
        
        # Validate padding_idx if provided
        if padding_idx is not None:
            if not isinstance(padding_idx, int):
                raise ValueError(f"padding_idx must be an integer or None, got {type(padding_idx)}")
            if padding_idx < 0 or padding_idx >= vocab_size:
                raise ValueError(
                    f"padding_idx must be in range [0, {vocab_size-1}], got {padding_idx}"
                )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Create the underlying PyTorch embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

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

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: transforms token indices to embeddings.

        Args:
            indices: Input tensor of shape [*, seq] containing token indices (long/int64)

        Returns:
            Output tensor of shape [*, seq, embedding_dim] containing embeddings
        """
        # Validate input dtype
        if indices.dtype not in [torch.long, torch.int64]:
            raise ValueError(
                f"Input indices must have dtype torch.long or torch.int64, "
                f"got {indices.dtype}"
            )

        # Validate input values are within vocab range
        if indices.numel() > 0:  # Only check if tensor is not empty
            min_idx = indices.min().item()
            max_idx = indices.max().item()
            if min_idx < 0:
                raise ValueError(
                    f"Input indices contain negative values: min={min_idx}"
                )
            if max_idx >= self.vocab_size:
                raise ValueError(
                    f"Input indices exceed vocab_size: max={max_idx}, vocab_size={self.vocab_size}"
                )

        return self.embedding(indices)

    def __repr__(self) -> str:
        """String representation of the block."""
        padding_str = f", padding_idx={self.padding_idx}" if self.padding_idx is not None else ""
        return (
            f"Embedding(vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}"
            f"{padding_str})"
        )
