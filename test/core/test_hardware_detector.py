"""Unit tests for HardwareDetector.

Tests hardware capability detection and block compatibility checking
with mocked hardware environments.

Tests Requirements 5.1, 5.2, 5.3, 5.4, 5.5 from specs/02-requirements.md
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple

from src.core.hardware_detector import (
    HardwareDetector,
    HardwareCapabilities,
    TORCH_AVAILABLE,
    PSUTIL_AVAILABLE
)


class TestHardwareCapabilities:
    """Test HardwareCapabilities dataclass."""

    def test_create_capabilities(self):
        """Test creating HardwareCapabilities instance."""
        caps = HardwareCapabilities(
            cuda_available=True,
            cuda_compute_capability=(8, 0),
            gpu_memory_gb=16.0,
            cpu_cores=8,
            system_memory_gb=32.0
        )

        assert caps.cuda_available is True
        assert caps.cuda_compute_capability == (8, 0)
        assert caps.gpu_memory_gb == 16.0
        assert caps.cpu_cores == 8
        assert caps.system_memory_gb == 32.0

    def test_capabilities_with_no_cuda(self):
        """Test HardwareCapabilities for CPU-only system."""
        caps = HardwareCapabilities(
            cuda_available=False,
            cuda_compute_capability=None,
            gpu_memory_gb=None,
            cpu_cores=4,
            system_memory_gb=8.0
        )

        assert caps.cuda_available is False
        assert caps.cuda_compute_capability is None
        assert caps.gpu_memory_gb is None


class TestHardwareDetectorCUDA:
    """Test CUDA detection functionality."""

    @patch('src.core.hardware_detector.torch')
    def test_check_cuda_available_true(self, mock_torch):
        """Test CUDA detection when CUDA is available.

        Implements Req 5.1
        """
        mock_torch.cuda.is_available.return_value = True

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            detector = HardwareDetector()
            assert detector.check_cuda_available() is True

    @patch('src.core.hardware_detector.torch')
    def test_check_cuda_available_false(self, mock_torch):
        """Test CUDA detection when CUDA is not available.

        Implements Req 5.1
        """
        mock_torch.cuda.is_available.return_value = False

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            detector = HardwareDetector()
            assert detector.check_cuda_available() is False

    def test_check_cuda_no_torch(self):
        """Test CUDA detection when PyTorch is not installed.

        Implements Req 5.1
        """
        with patch('src.core.hardware_detector.TORCH_AVAILABLE', False):
            # Need to mock _detect to avoid initialization issues
            with patch.object(HardwareDetector, '_detect') as mock_detect:
                mock_detect.return_value = HardwareCapabilities(
                    cuda_available=False,
                    cuda_compute_capability=None,
                    gpu_memory_gb=None,
                    cpu_cores=1,
                    system_memory_gb=1.0
                )
                detector = HardwareDetector()
                assert detector.check_cuda_available() is False

    @patch('src.core.hardware_detector.torch')
    def test_get_compute_capability(self, mock_torch):
        """Test compute capability detection.

        Implements Req 5.2
        """
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            detector = HardwareDetector()
            capability = detector.get_compute_capability()
            assert capability == (8, 0)

    @patch('src.core.hardware_detector.torch')
    def test_get_compute_capability_no_cuda(self, mock_torch):
        """Test compute capability when CUDA is not available.

        Implements Req 5.2
        """
        mock_torch.cuda.is_available.return_value = False

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            detector = HardwareDetector()
            capability = detector.get_compute_capability()
            assert capability is None

    @patch('src.core.hardware_detector.torch')
    def test_get_compute_capability_multiple_versions(self, mock_torch):
        """Test various compute capability versions.

        Implements Req 5.2
        """
        test_cases = [
            (7, 0),  # Volta
            (7, 5),  # Turing
            (8, 0),  # Ampere
            (8, 6),  # Ampere (mobile)
            (9, 0),  # Hopper
        ]

        for expected_capability in test_cases:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_capability.return_value = expected_capability

            with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
                detector = HardwareDetector()
                capability = detector.get_compute_capability()
                assert capability == expected_capability


class TestHardwareDetectorMemory:
    """Test memory detection functionality."""

    @patch('src.core.hardware_detector.torch')
    def test_get_gpu_memory(self, mock_torch):
        """Test GPU memory detection.

        Implements Req 5.4
        """
        mock_torch.cuda.is_available.return_value = True
        mock_properties = Mock()
        mock_properties.total_memory = 16 * (1024 ** 3)  # 16 GB in bytes
        mock_torch.cuda.get_device_properties.return_value = mock_properties

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.psutil'):
                detector = HardwareDetector()
                # Access private method for testing
                gpu_memory = detector._get_gpu_memory()
                assert gpu_memory == pytest.approx(16.0, rel=0.01)

    @patch('src.core.hardware_detector.psutil')
    def test_get_cpu_cores(self, mock_psutil):
        """Test CPU core detection.

        Implements Req 5.4
        """
        mock_psutil.cpu_count.return_value = 8

        with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
            with patch('src.core.hardware_detector.torch'):
                detector = HardwareDetector()
                # CPU cores should be detected during init
                assert detector.capabilities.cpu_cores == 8

    @patch('src.core.hardware_detector.psutil')
    def test_get_system_memory(self, mock_psutil):
        """Test system memory detection.

        Implements Req 5.4
        """
        mock_memory = Mock()
        mock_memory.total = 32 * (1024 ** 3)  # 32 GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory

        with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
            with patch('src.core.hardware_detector.torch'):
                detector = HardwareDetector()
                assert detector.capabilities.system_memory_gb == pytest.approx(32.0, rel=0.01)

    def test_fallback_when_psutil_unavailable(self):
        """Test fallback values when psutil is not available.

        Implements Req 5.4
        """
        with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', False):
            with patch('src.core.hardware_detector.TORCH_AVAILABLE', False):
                detector = HardwareDetector()
                # Should fall back to minimal values
                assert detector.capabilities.cpu_cores == 1
                assert detector.capabilities.system_memory_gb == 1.0


class TestHardwareDetectorCompatibility:
    """Test block compatibility checking."""

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_is_block_compatible_cuda_required_available(self, mock_psutil, mock_torch):
        """Test block requiring CUDA when CUDA is available.

        Implements Req 5.3
        """
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_memory = Mock()
        mock_memory.total = 16 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 8

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                block_reqs = {'requires_cuda': True}
                compatible, warnings = detector.is_block_compatible(block_reqs)

                assert compatible is True
                assert len(warnings) == 0

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_is_block_compatible_cuda_required_unavailable(self, mock_psutil, mock_torch):
        """Test block requiring CUDA when CUDA is not available.

        Implements Req 5.3
        """
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_memory = Mock()
        mock_memory.total = 16 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 8

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                block_reqs = {'requires_cuda': True}
                compatible, warnings = detector.is_block_compatible(block_reqs)

                assert compatible is False
                assert len(warnings) == 1
                assert "CUDA" in warnings[0]

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_is_block_compatible_compute_capability_sufficient(self, mock_psutil, mock_torch):
        """Test block with compute capability requirement met.

        Implements Req 5.3
        """
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_memory = Mock()
        mock_memory.total = 16 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 8

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                block_reqs = {'min_cuda_capability': (7, 0)}
                compatible, warnings = detector.is_block_compatible(block_reqs)

                assert compatible is True
                assert len(warnings) == 0

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_is_block_compatible_compute_capability_insufficient(self, mock_psutil, mock_torch):
        """Test block with compute capability requirement not met.

        Implements Req 5.3
        """
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (7, 0)
        mock_memory = Mock()
        mock_memory.total = 16 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 8

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                block_reqs = {'min_cuda_capability': (8, 0)}
                compatible, warnings = detector.is_block_compatible(block_reqs)

                assert compatible is False
                assert len(warnings) == 1
                assert "compute capability" in warnings[0].lower()

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_is_block_compatible_memory_warning_gpu(self, mock_psutil, mock_torch):
        """Test block with memory exceeding GPU capacity generates warning.

        Implements Req 5.5
        """
        # Setup mocks for GPU system
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_properties = Mock()
        mock_properties.total_memory = 8 * (1024 ** 3)  # 8 GB GPU
        mock_torch.cuda.get_device_properties.return_value = mock_properties

        mock_memory = Mock()
        mock_memory.total = 32 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 8

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                # Block requiring 16 GB on GPU
                block_reqs = {
                    'estimated_memory_gb': 16.0,
                    'prefer_gpu': True
                }
                compatible, warnings = detector.is_block_compatible(block_reqs)

                assert compatible is True  # Still compatible, just warning
                assert len(warnings) == 1
                assert "GPU memory" in warnings[0]
                assert "out-of-memory" in warnings[0]

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_is_block_compatible_memory_warning_system(self, mock_psutil, mock_torch):
        """Test block with memory exceeding system capacity generates warning.

        Implements Req 5.5
        """
        # Setup mocks for CPU-only system
        mock_torch.cuda.is_available.return_value = False

        mock_memory = Mock()
        mock_memory.total = 8 * (1024 ** 3)  # 8 GB system RAM
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 4

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                # Block requiring 16 GB
                block_reqs = {'estimated_memory_gb': 16.0}
                compatible, warnings = detector.is_block_compatible(block_reqs)

                assert compatible is True  # Still compatible, just warning
                assert len(warnings) == 1
                assert "system memory" in warnings[0]

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_is_block_compatible_no_requirements(self, mock_psutil, mock_torch):
        """Test block with no special requirements.

        Implements Req 5.3
        """
        # Setup basic mocks
        mock_torch.cuda.is_available.return_value = False
        mock_memory = Mock()
        mock_memory.total = 8 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 4

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                block_reqs = {}  # No special requirements
                compatible, warnings = detector.is_block_compatible(block_reqs)

                assert compatible is True
                assert len(warnings) == 0


class TestHardwareDetectorMemoryEstimation:
    """Test memory estimation functionality."""

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_estimate_block_memory_default(self, mock_psutil, mock_torch):
        """Test basic memory estimation.

        Implements Req 5.5
        """
        # Setup minimal mocks
        mock_torch.cuda.is_available.return_value = False
        mock_memory = Mock()
        mock_memory.total = 8 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 4

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                # 1M parameters, float32 (4 bytes), 3x multiplier
                # Expected: 1M * 4 bytes * 3 = 12 MB = 0.0114 GB
                param_count = 1_000_000
                estimated = detector.estimate_block_memory(param_count)

                assert estimated == pytest.approx(0.0114, abs=0.001)

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_estimate_block_memory_custom_dtype(self, mock_psutil, mock_torch):
        """Test memory estimation with custom dtype size.

        Implements Req 5.5
        """
        # Setup minimal mocks
        mock_torch.cuda.is_available.return_value = False
        mock_memory = Mock()
        mock_memory.total = 8 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 4

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                # 1M parameters, float16 (2 bytes)
                param_count = 1_000_000
                estimated = detector.estimate_block_memory(param_count, dtype_size=2)

                # Expected: 1M * 2 bytes * 3 = 6 MB = 0.0057 GB
                assert estimated == pytest.approx(0.0057, abs=0.001)

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_estimate_block_memory_large_model(self, mock_psutil, mock_torch):
        """Test memory estimation for large model.

        Implements Req 5.5
        """
        # Setup minimal mocks
        mock_torch.cuda.is_available.return_value = False
        mock_memory = Mock()
        mock_memory.total = 64 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 16

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()

                # 7B parameters (like Llama-7B)
                param_count = 7_000_000_000
                estimated = detector.estimate_block_memory(param_count)

                # Expected: 7B * 4 bytes * 3 = 84 GB
                assert estimated == pytest.approx(84.0, rel=0.01)


class TestHardwareDetectorIntegration:
    """Integration tests for complete detection workflow."""

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_full_detection_gpu_system(self, mock_psutil, mock_torch):
        """Test complete hardware detection on GPU system.

        Implements Req 5.1, 5.2, 5.4
        """
        # Setup complete GPU system
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 6)
        mock_properties = Mock()
        mock_properties.total_memory = 24 * (1024 ** 3)  # 24 GB GPU
        mock_torch.cuda.get_device_properties.return_value = mock_properties

        mock_memory = Mock()
        mock_memory.total = 64 * (1024 ** 3)  # 64 GB RAM
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 16

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()
                caps = detector.capabilities

                assert caps.cuda_available is True
                assert caps.cuda_compute_capability == (8, 6)
                assert caps.gpu_memory_gb == pytest.approx(24.0, rel=0.01)
                assert caps.cpu_cores == 16
                assert caps.system_memory_gb == pytest.approx(64.0, rel=0.01)

    @patch('src.core.hardware_detector.torch')
    @patch('src.core.hardware_detector.psutil')
    def test_full_detection_cpu_only_system(self, mock_psutil, mock_torch):
        """Test complete hardware detection on CPU-only system.

        Implements Req 5.1, 5.4
        """
        # Setup CPU-only system
        mock_torch.cuda.is_available.return_value = False

        mock_memory = Mock()
        mock_memory.total = 16 * (1024 ** 3)  # 16 GB RAM
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_count.return_value = 8

        with patch('src.core.hardware_detector.TORCH_AVAILABLE', True):
            with patch('src.core.hardware_detector.PSUTIL_AVAILABLE', True):
                detector = HardwareDetector()
                caps = detector.capabilities

                assert caps.cuda_available is False
                assert caps.cuda_compute_capability is None
                assert caps.gpu_memory_gb is None
                assert caps.cpu_cores == 8
                assert caps.system_memory_gb == pytest.approx(16.0, rel=0.01)
