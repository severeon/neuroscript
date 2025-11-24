"""Hardware capability detection for filtering compatible blocks.

This module detects available hardware (CUDA capability, CPU cores, memory)
and checks if blocks are compatible with the detected capabilities.

Implements Requirements 5.1, 5.2, 5.3, 5.4, 5.5 from specs/02-requirements.md
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

# Import torch for CUDA detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# Import psutil for CPU/memory detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities.

    Attributes:
        cuda_available: Whether CUDA is available
        cuda_compute_capability: CUDA compute capability as (major, minor) tuple
        gpu_memory_gb: Available GPU memory in GB
        cpu_cores: Number of CPU cores
        system_memory_gb: Available system memory in GB
    """

    cuda_available: bool
    cuda_compute_capability: Optional[Tuple[int, int]]
    gpu_memory_gb: Optional[float]
    cpu_cores: int
    system_memory_gb: float


class HardwareDetector:
    """Detects hardware capabilities and checks block compatibility.

    This class detects available hardware resources including CUDA GPUs,
    CPU cores, and system memory. It can check if blocks are compatible
    with the detected hardware and warn about resource constraints.

    Implements Requirements:
        - 5.1: CUDA availability detection
        - 5.2: Compute capability detection
        - 5.3: Block compatibility checking
        - 5.4: Memory detection
        - 5.5: Resource warning generation

    Attributes:
        capabilities: Detected hardware capabilities
    """

    def __init__(self):
        """Initialize detector and detect hardware.

        Implements Req 5.1, 5.2, 5.4
        """
        self.capabilities = self._detect()
        logger.info(f"Hardware detected: {self.capabilities}")

    def _detect(self) -> HardwareCapabilities:
        """Detect all hardware capabilities.

        Queries:
        - CUDA availability and compute capability
        - GPU memory if CUDA is available
        - CPU core count
        - System memory

        Implements Req 5.1, 5.2, 5.4

        Returns:
            HardwareCapabilities: Complete hardware specification
        """
        # Detect CUDA
        cuda_available = self.check_cuda_available()
        cuda_compute_capability = self.get_compute_capability()
        gpu_memory_gb = self._get_gpu_memory() if cuda_available else None

        # Detect CPU and memory
        cpu_cores = self._get_cpu_cores()
        system_memory_gb = self._get_system_memory()

        return HardwareCapabilities(
            cuda_available=cuda_available,
            cuda_compute_capability=cuda_compute_capability,
            gpu_memory_gb=gpu_memory_gb,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory_gb
        )

    def check_cuda_available(self) -> bool:
        """Check if CUDA is available.

        Implements Req 5.1

        Returns:
            bool: True if CUDA is available, False otherwise
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, CUDA detection disabled")
            return False

        try:
            return torch.cuda.is_available()
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Error checking CUDA availability: {e}")
            return False

    def get_compute_capability(self) -> Optional[Tuple[int, int]]:
        """Get CUDA compute capability.

        Implements Req 5.2

        Returns:
            Optional[Tuple[int, int]]: (major, minor) compute capability or None if no CUDA
        """
        if not self.check_cuda_available():
            return None

        try:
            # Get capability for device 0 (primary GPU)
            return torch.cuda.get_device_capability(0)
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Error getting CUDA compute capability: {e}")
            return None

    def _get_gpu_memory(self) -> Optional[float]:
        """Get available GPU memory in GB.

        Implements Req 5.4

        Returns:
            Optional[float]: GPU memory in GB or None if unavailable
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        try:
            # Get total memory for device 0
            total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_memory_bytes / (1024 ** 3)  # Convert to GB
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Error getting GPU memory: {e}")
            return None

    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores.

        Implements Req 5.4

        Returns:
            int: Number of CPU cores (defaults to 1 if detection fails)
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, defaulting to 1 CPU core")
            return 1

        try:
            # Use physical cores (not hyperthreads)
            cores = psutil.cpu_count(logical=False)
            return cores if cores is not None else 1
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"Error detecting CPU cores: {e}")
            return 1

    def _get_system_memory(self) -> float:
        """Get available system memory in GB.

        Implements Req 5.4

        Returns:
            float: System memory in GB (defaults to 1.0 if detection fails)
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, defaulting to 1GB system memory")
            return 1.0

        try:
            memory = psutil.virtual_memory()
            return memory.total / (1024 ** 3)  # Convert to GB
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"Error detecting system memory: {e}")
            return 1.0

    def is_block_compatible(
        self,
        block_requirements: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Check if block is compatible with available hardware.

        Checks:
        - CUDA availability if required
        - Compute capability if specified
        - Estimated memory usage

        Implements Req 5.3, 5.5

        Args:
            block_requirements: Hardware requirements from block capability spec.
                Expected keys:
                - 'requires_cuda' (bool): Whether CUDA is required
                - 'min_cuda_capability' (Tuple[int, int]): Minimum compute capability
                - 'estimated_memory_gb' (float): Estimated memory usage
                - 'prefer_gpu' (bool): Whether GPU is preferred but not required

        Returns:
            Tuple[bool, List[str]]: (is_compatible, warning_messages)
                is_compatible: True if block can run on this hardware
                warning_messages: List of warnings about resource constraints
        """
        warnings = []

        # Check CUDA requirements
        requires_cuda = block_requirements.get('requires_cuda', False)
        if requires_cuda and not self.capabilities.cuda_available:
            return False, ["Block requires CUDA but CUDA is not available"]

        # Check compute capability
        min_cuda_capability = block_requirements.get('min_cuda_capability')
        if min_cuda_capability is not None:
            if not self.capabilities.cuda_available:
                return False, ["Block requires CUDA compute capability but CUDA is not available"]

            current_capability = self.capabilities.cuda_compute_capability
            if current_capability is None:
                return False, ["Could not detect CUDA compute capability"]

            # Compare (major, minor) tuples
            if current_capability < min_cuda_capability:
                return False, [
                    f"Block requires CUDA compute capability {min_cuda_capability} "
                    f"but only {current_capability} is available"
                ]

        # Check memory requirements
        estimated_memory_gb = block_requirements.get('estimated_memory_gb')
        if estimated_memory_gb is not None:
            prefer_gpu = block_requirements.get('prefer_gpu', False)

            # Check GPU memory if GPU is preferred and available
            if prefer_gpu and self.capabilities.cuda_available:
                if self.capabilities.gpu_memory_gb is not None:
                    if estimated_memory_gb > self.capabilities.gpu_memory_gb:
                        warnings.append(
                            f"Block requires ~{estimated_memory_gb:.2f}GB GPU memory "
                            f"but only {self.capabilities.gpu_memory_gb:.2f}GB available. "
                            f"May cause out-of-memory errors."
                        )
            else:
                # Check system memory
                if estimated_memory_gb > self.capabilities.system_memory_gb:
                    warnings.append(
                        f"Block requires ~{estimated_memory_gb:.2f}GB system memory "
                        f"but only {self.capabilities.system_memory_gb:.2f}GB available. "
                        f"May cause out-of-memory errors."
                    )

        # Block is compatible, but may have warnings
        return True, warnings

    def estimate_block_memory(
        self,
        param_count: int,
        dtype_size: int = 4,
        activation_multiplier: float = 3.0
    ) -> float:
        """Estimate memory usage for a block.

        Provides a rough estimate of memory requirements based on parameter count
        and typical activation memory multiplier.

        Implements Req 5.5

        Args:
            param_count: Number of parameters in the block
            dtype_size: Size of each parameter in bytes (default: 4 for float32)
            activation_multiplier: Multiplier for activation memory (default: 3.0)
                Accounts for gradients, optimizer states, and activations

        Returns:
            float: Estimated memory usage in GB
        """
        # Parameter memory
        param_memory_bytes = param_count * dtype_size

        # Total memory including activations/gradients (rough estimate)
        total_memory_bytes = param_memory_bytes * activation_multiplier

        # Convert to GB
        return total_memory_bytes / (1024 ** 3)
