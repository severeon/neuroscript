"""BlockRegistry for discovering and loading blocks."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .block_interface import BlockCapability, BlockInterface
from .capability_parser import CapabilityParser, CapabilityParseError

logger = logging.getLogger(__name__)


class BlockRegistryError(Exception):
    """Raised when block registry operations fail."""

    pass


class BlockRegistry:
    """
    Registry for discovering and managing neural network blocks.

    Automatically discovers blocks in a filesystem directory tree and loads
    their capability specifications using CapabilityParser.
    """

    def __init__(self, blocks_dir: str = "blocks"):
        """
        Initialize the BlockRegistry and discover available blocks.

        Args:
            blocks_dir: Root directory containing block subdirectories
        """
        self.blocks_dir = Path(blocks_dir)
        self._registry: Dict[str, BlockCapability] = {}
        self._block_paths: Dict[str, Path] = {}
        self._parser = CapabilityParser()
        self._discover_blocks()

    def _discover_blocks(self) -> None:
        """
        Recursively scan blocks_dir for block.yaml files and load them.

        A valid block directory contains a block.yaml file. The directory name
        becomes the block identifier.

        Logs warnings for:
        - Directories without block.yaml files
        - Duplicate block identifiers
        """
        if not self.blocks_dir.exists():
            logger.warning(f"Blocks directory does not exist: {self.blocks_dir}")
            return

        if not self.blocks_dir.is_dir():
            logger.warning(f"Blocks path is not a directory: {self.blocks_dir}")
            return

        seen_blocks: Set[str] = set()

        # Walk the blocks directory
        for item in self.blocks_dir.iterdir():
            if not item.is_dir():
                continue

            block_yaml = item / "block.yaml"

            if not block_yaml.exists():
                logger.warning(
                    f"Block directory missing block.yaml: {item.name}"
                )
                continue

            block_id = item.name

            # Check for duplicates
            if block_id in seen_blocks:
                logger.error(
                    f"Duplicate block identifier: {block_id}. "
                    f"Found in {self._block_paths[block_id]} and {item}"
                )
                raise BlockRegistryError(
                    f"Duplicate block identifier: {block_id}"
                )

            seen_blocks.add(block_id)

            # Try to parse the block
            try:
                capability = self._parser.parse_file(str(block_yaml))
                self._registry[block_id] = capability
                self._block_paths[block_id] = item
                logger.debug(f"Registered block: {block_id} ({capability.name})")
            except CapabilityParseError as e:
                logger.error(
                    f"Failed to parse block {block_id} at {block_yaml}: {e}"
                )
                raise BlockRegistryError(
                    f"Failed to parse block {block_id}: {e}"
                )

    def get_block(self, block_id: str) -> BlockCapability:
        """
        Retrieve a block's capability specification by identifier.

        Args:
            block_id: Block identifier (directory name)

        Returns:
            BlockCapability: The block's parsed capability specification

        Raises:
            BlockRegistryError: If block not found
        """
        if block_id not in self._registry:
            available = self.list_blocks()
            raise BlockRegistryError(
                f"Block not found: {block_id}. "
                f"Available blocks: {', '.join(available) if available else 'none'}"
            )

        return self._registry[block_id]

    def list_blocks(self) -> List[str]:
        """
        Return list of all registered block identifiers.

        Returns:
            List of block identifiers sorted alphabetically
        """
        return sorted(self._registry.keys())

    def validate_interface(self, block_instance: Any) -> None:
        """
        Validate that a block instance implements BlockInterface.

        Args:
            block_instance: Object to validate

        Raises:
            BlockRegistryError: If block doesn't implement required methods
        """
        if not isinstance(block_instance, BlockInterface):
            raise BlockRegistryError(
                f"Block does not implement BlockInterface. "
                f"Must implement get_capabilities() and forward() methods. "
                f"Got {type(block_instance)}"
            )

    def get_block_path(self, block_id: str) -> Path:
        """
        Get the filesystem path for a block directory.

        Args:
            block_id: Block identifier

        Returns:
            Path to block directory

        Raises:
            BlockRegistryError: If block not found
        """
        if block_id not in self._block_paths:
            raise BlockRegistryError(f"Block not found: {block_id}")

        return self._block_paths[block_id]

    def block_exists(self, block_id: str) -> bool:
        """
        Check if a block is registered.

        Args:
            block_id: Block identifier

        Returns:
            True if block exists, False otherwise
        """
        return block_id in self._registry

    def __contains__(self, block_id: str) -> bool:
        """Support 'in' operator for block existence checks."""
        return self.block_exists(block_id)

    def __len__(self) -> int:
        """Return number of registered blocks."""
        return len(self._registry)

    def __iter__(self):
        """Iterate over registered block identifiers."""
        return iter(self._registry)
