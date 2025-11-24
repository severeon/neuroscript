"""Unit tests for ContainerRuntime."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from src.core.container_runtime import (
    ContainerRuntime,
    ContainerRuntimeError,
    ExecutionResult,
)
from src.core.hardware_detector import HardwareCapabilities, HardwareDetector


@pytest.fixture
def mock_hardware_detector():
    """Create a mock hardware detector."""
    detector = Mock(spec=HardwareDetector)
    detector.capabilities = HardwareCapabilities(
        cuda_available=True,
        cuda_compute_capability=(8, 0),
        gpu_memory_gb=8.0,
        cpu_cores=8,
        system_memory_gb=16.0,
    )
    return detector


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with test files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create a simple test model
    model_file = workspace / "model.py"
    model_file.write_text("""
import sys
print("Model executed successfully")
""")

    return workspace


@pytest.fixture
def runtime(mock_hardware_detector):
    """Create ContainerRuntime instance with mocked Docker."""
    with patch.object(ContainerRuntime, "_verify_docker_available"):
        runtime = ContainerRuntime(
            hardware_detector=mock_hardware_detector, image_name="test-runtime:latest"
        )
    return runtime


class TestContainerRuntime:
    """Test suite for ContainerRuntime class."""

    def test_init(self, mock_hardware_detector):
        """Test ContainerRuntime initialization."""
        with patch.object(ContainerRuntime, "_verify_docker_available"):
            runtime = ContainerRuntime(
                hardware_detector=mock_hardware_detector, image_name="test:latest"
            )

        assert runtime.hardware_detector == mock_hardware_detector
        assert runtime.image_name == "test:latest"

    def test_init_creates_hardware_detector_if_not_provided(self):
        """Test that hardware detector is created if not provided."""
        with patch.object(ContainerRuntime, "_verify_docker_available"):
            with patch("src.core.container_runtime.HardwareDetector") as mock_detector:
                runtime = ContainerRuntime()
                assert runtime.hardware_detector is not None

    def test_verify_docker_available_success(self):
        """Test Docker availability check succeeds."""
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            # Should not raise
            with patch.object(ContainerRuntime, "_verify_docker_available"):
                runtime = ContainerRuntime()
            runtime._verify_docker_available()

    def test_verify_docker_available_not_running(self):
        """Test Docker availability check fails when Docker not running."""
        mock_result = Mock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(
                ContainerRuntimeError, match="Docker is not available or not running"
            ):
                runtime = ContainerRuntime()

    def test_verify_docker_available_not_installed(self):
        """Test Docker availability check fails when Docker not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(
                ContainerRuntimeError, match="Docker command not found"
            ):
                runtime = ContainerRuntime()

    def test_verify_docker_available_timeout(self):
        """Test Docker availability check fails on timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 5)):
            with pytest.raises(ContainerRuntimeError, match="Docker command timed out"):
                runtime = ContainerRuntime()

    def test_verify_image_exists_success(self, runtime):
        """Test image existence check succeeds."""
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            runtime._verify_image_exists()  # Should not raise

    def test_verify_image_exists_not_found(self, runtime):
        """Test image existence check fails when image not found."""
        mock_result = Mock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ContainerRuntimeError, match="Docker image.*not found"):
                runtime._verify_image_exists()

    def test_get_resource_limits_auto_detect(self, runtime):
        """Test automatic resource limit detection."""
        cpu_limit, memory_limit = runtime.get_resource_limits()

        # Should use half of available resources
        assert cpu_limit == "4.0"  # Half of 8 cores
        assert memory_limit == "8g"  # Half of 16GB

    def test_get_resource_limits_custom(self, runtime):
        """Test custom resource limits."""
        cpu_limit, memory_limit = runtime.get_resource_limits(
            cpu_limit=2.5, memory_limit="4g"
        )

        assert cpu_limit == "2.5"
        assert memory_limit == "4g"

    def test_get_resource_limits_invalid_memory_format(self, runtime):
        """Test invalid memory format raises error."""
        with pytest.raises(ContainerRuntimeError, match="Invalid memory limit format"):
            runtime.get_resource_limits(memory_limit="invalid")

    def test_validate_memory_format_valid(self, runtime):
        """Test valid memory format validation."""
        assert runtime._validate_memory_format("1g")
        assert runtime._validate_memory_format("512m")
        assert runtime._validate_memory_format("1024k")
        assert runtime._validate_memory_format("1073741824b")

    def test_validate_memory_format_invalid(self, runtime):
        """Test invalid memory format validation."""
        assert not runtime._validate_memory_format("1gb")
        assert not runtime._validate_memory_format("512")
        assert not runtime._validate_memory_format("invalid")

    def test_create_container_success(self, runtime, temp_workspace):
        """Test successful container creation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "container123456\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            container_id = runtime.create_container(
                code_dir=temp_workspace,
                command=["python", "model.py"],
                cpu_limit="2.0",
                memory_limit="2g",
            )

        assert container_id == "container123456"
        mock_run.assert_called_once()

        # Verify docker create command
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "create" in call_args
        assert "--cpus" in call_args
        assert "2.0" in call_args
        assert "--memory" in call_args
        assert "2g" in call_args

    def test_create_container_with_gpu(self, runtime, temp_workspace):
        """Test container creation with GPU passthrough."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "container123456\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            container_id = runtime.create_container(
                code_dir=temp_workspace,
                command=["python", "model.py"],
                cpu_limit="2.0",
                memory_limit="2g",
                enable_gpu=True,
            )

        # Verify --gpus flag is present
        call_args = mock_run.call_args[0][0]
        assert "--gpus" in call_args
        assert "all" in call_args

    def test_create_container_gpu_not_available(self, runtime, temp_workspace):
        """Test GPU request when CUDA not available."""
        runtime.hardware_detector.capabilities.cuda_available = False

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "container123456\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            container_id = runtime.create_container(
                code_dir=temp_workspace,
                command=["python", "model.py"],
                cpu_limit="2.0",
                memory_limit="2g",
                enable_gpu=True,
            )

        # Verify --gpus flag is NOT present
        call_args = mock_run.call_args[0][0]
        assert "--gpus" not in call_args

    def test_create_container_with_env_vars(self, runtime, temp_workspace):
        """Test container creation with environment variables."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "container123456\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            runtime.create_container(
                code_dir=temp_workspace,
                command=["python", "model.py"],
                cpu_limit="2.0",
                memory_limit="2g",
                env_vars={"FOO": "bar", "BAZ": "qux"},
            )

        call_args = mock_run.call_args[0][0]
        assert "-e" in call_args
        assert "FOO=bar" in call_args
        assert "BAZ=qux" in call_args

    def test_create_container_failure(self, runtime, temp_workspace):
        """Test container creation failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Error creating container"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(
                ContainerRuntimeError, match="Failed to create container"
            ):
                runtime.create_container(
                    code_dir=temp_workspace,
                    command=["python", "model.py"],
                    cpu_limit="2.0",
                    memory_limit="2g",
                )

    def test_execute_success(self, runtime, temp_workspace):
        """Test successful model execution."""
        model_path = temp_workspace / "model.py"

        # Mock all subprocess calls
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0] if args else []

            if "image" in cmd and "inspect" in cmd:
                # Image exists
                result = Mock()
                result.returncode = 0
                return result
            elif "create" in cmd:
                # Container creation
                result = Mock()
                result.returncode = 0
                result.stdout = "container123456\n"
                return result
            elif "start" in cmd:
                # Container execution
                result = Mock()
                result.returncode = 0
                result.stdout = "Model executed successfully\n"
                result.stderr = ""
                return result
            elif "inspect" in cmd and "ExitCode" in " ".join(cmd):
                # Exit code check
                result = Mock()
                result.returncode = 0
                result.stdout = "0"
                return result
            elif "inspect" in cmd and "OOMKilled" in " ".join(cmd):
                # OOM check
                result = Mock()
                result.returncode = 0
                result.stdout = "false"
                return result
            elif "rm" in cmd:
                # Container cleanup
                result = Mock()
                result.returncode = 0
                return result
            else:
                result = Mock()
                result.returncode = 0
                return result

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            result = runtime.execute(
                model_path=model_path, code_dir=temp_workspace, cpu_limit=2.0
            )

        assert result.exit_code == 0
        assert "Model executed successfully" in result.stdout
        assert result.oom_killed is False
        assert result.execution_time_seconds >= 0

    def test_execute_model_not_found(self, runtime, temp_workspace):
        """Test execution fails when model file not found."""
        model_path = temp_workspace / "nonexistent.py"

        with patch.object(runtime, "_verify_image_exists"):
            with pytest.raises(ContainerRuntimeError, match="Model file not found"):
                runtime.execute(model_path=model_path, code_dir=temp_workspace)

    def test_execute_code_dir_not_found(self, runtime, temp_workspace):
        """Test execution fails when code directory not found."""
        model_path = temp_workspace / "model.py"
        nonexistent_dir = temp_workspace / "nonexistent"

        with patch.object(runtime, "_verify_image_exists"):
            with pytest.raises(ContainerRuntimeError, match="Code directory not found"):
                runtime.execute(model_path=model_path, code_dir=nonexistent_dir)

    def test_execute_model_outside_code_dir(self, runtime, tmp_path):
        """Test execution fails when model is outside code directory."""
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        model_path = tmp_path / "outside" / "model.py"
        model_path.parent.mkdir()
        model_path.write_text("print('test')")

        # Mock image check
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(
                ContainerRuntimeError, match="must be inside code directory"
            ):
                runtime.execute(model_path=model_path, code_dir=code_dir)

    def test_execute_with_timeout(self, runtime, temp_workspace):
        """Test execution timeout."""
        model_path = temp_workspace / "model.py"

        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0] if args else []

            if "image" in cmd and "inspect" in cmd:
                result = Mock()
                result.returncode = 0
                return result
            elif "create" in cmd:
                result = Mock()
                result.returncode = 0
                result.stdout = "container123456\n"
                return result
            elif "start" in cmd:
                # Simulate timeout
                raise subprocess.TimeoutExpired("docker start", 10)
            elif "kill" in cmd:
                # Container kill
                result = Mock()
                result.returncode = 0
                return result
            elif "rm" in cmd:
                # Cleanup
                result = Mock()
                result.returncode = 0
                return result
            else:
                result = Mock()
                result.returncode = 0
                return result

        with patch("subprocess.run", side_effect=mock_run_side_effect):
            with pytest.raises(
                ContainerRuntimeError, match="Execution timed out after 10 seconds"
            ):
                runtime.execute(
                    model_path=model_path, code_dir=temp_workspace, timeout=10
                )

    def test_check_oom_killed_true(self, runtime):
        """Test OOM detection returns True."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "true\n"

        with patch("subprocess.run", return_value=mock_result):
            assert runtime._check_oom_killed("container123") is True

    def test_check_oom_killed_false(self, runtime):
        """Test OOM detection returns False."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "false\n"

        with patch("subprocess.run", return_value=mock_result):
            assert runtime._check_oom_killed("container123") is False

    def test_check_oom_killed_error(self, runtime):
        """Test OOM detection handles errors gracefully."""
        with patch("subprocess.run", side_effect=Exception("Error")):
            assert runtime._check_oom_killed("container123") is False

    def test_handle_oom(self, runtime):
        """Test OOM handler raises appropriate error."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "500MiB / 512MiB\n"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(ContainerRuntimeError) as exc_info:
                runtime.handle_oom("container123", "512m")

        assert "exceeded memory limit" in str(exc_info.value).lower()
        assert "512m" in str(exc_info.value)
        assert "Suggestions:" in str(exc_info.value)

    def test_cleanup_container_success(self, runtime):
        """Test successful container cleanup."""
        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            runtime._cleanup_container("container123")

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "rm" in call_args
        assert "container123" in call_args

    def test_cleanup_container_failure(self, runtime):
        """Test container cleanup handles errors gracefully."""
        with patch("subprocess.run", side_effect=Exception("Error")):
            # Should not raise, just log warning
            runtime._cleanup_container("container123")

    def test_build_runtime_image_success(self, runtime, tmp_path):
        """Test successful Docker image build."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.9\n")

        mock_result = Mock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            runtime.build_runtime_image(dockerfile)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "build" in call_args
        assert "-t" in call_args

    def test_build_runtime_image_not_found(self, runtime, tmp_path):
        """Test image build fails when Dockerfile not found."""
        dockerfile = tmp_path / "Dockerfile"

        with pytest.raises(ContainerRuntimeError, match="Dockerfile not found"):
            runtime.build_runtime_image(dockerfile)

    def test_build_runtime_image_failure(self, runtime, tmp_path):
        """Test image build failure."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.9\n")

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Build error"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(
                ContainerRuntimeError, match="Failed to build Docker image"
            ):
                runtime.build_runtime_image(dockerfile)


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_creation(self):
        """Test ExecutionResult can be created with all fields."""
        result = ExecutionResult(
            exit_code=0,
            stdout="output",
            stderr="",
            container_id="container123",
            oom_killed=False,
            execution_time_seconds=1.5,
        )

        assert result.exit_code == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.container_id == "container123"
        assert result.oom_killed is False
        assert result.execution_time_seconds == 1.5

    def test_execution_result_oom(self):
        """Test ExecutionResult with OOM condition."""
        result = ExecutionResult(
            exit_code=137,
            stdout="",
            stderr="",
            container_id="container123",
            oom_killed=True,
            execution_time_seconds=0.5,
        )

        assert result.oom_killed is True
        assert result.exit_code == 137  # Common OOM exit code


# Integration tests (require Docker)
@pytest.mark.integration
@pytest.mark.skipif(
    not Path("/var/run/docker.sock").exists(), reason="Docker not available"
)
class TestContainerRuntimeIntegration:
    """Integration tests for ContainerRuntime (require Docker)."""

    def test_docker_version_check(self):
        """Test actual Docker version check."""
        result = subprocess.run(
            ["docker", "version"], capture_output=True, timeout=5
        )
        assert result.returncode == 0

    def test_real_container_lifecycle(self, tmp_path):
        """Test real container creation and cleanup."""
        # Create test workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        model = workspace / "test.py"
        model.write_text('print("Hello from container")')

        # Use Python image instead of custom runtime
        runtime = ContainerRuntime(image_name="python:3.9-slim")

        try:
            # Create container
            container_id = runtime.create_container(
                code_dir=workspace,
                command=["python", "test.py"],
                cpu_limit="1.0",
                memory_limit="512m",
            )

            assert container_id
            assert len(container_id) == 64  # Docker container ID length

            # Verify container exists
            result = subprocess.run(
                ["docker", "inspect", container_id], capture_output=True
            )
            assert result.returncode == 0

        finally:
            # Cleanup
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
