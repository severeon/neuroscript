"""ContainerRuntime for executing models in isolated Docker containers with resource limits.

This module executes generated PyTorch models in Docker containers with
configurable CPU, memory, and GPU resource limits.

Implements Requirements 9.1, 9.2, 9.3, 9.4, 9.5 from specs/02-requirements.md
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hardware_detector import HardwareDetector

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of model execution in container.

    Attributes:
        exit_code: Container exit code (0 for success)
        stdout: Standard output captured from container
        stderr: Standard error captured from container
        container_id: Docker container ID
        oom_killed: Whether container was killed due to OOM
        execution_time_seconds: Time taken for execution
    """

    exit_code: int
    stdout: str
    stderr: str
    container_id: str
    oom_killed: bool
    execution_time_seconds: float


class ContainerRuntimeError(Exception):
    """Raised when container execution fails."""

    pass


class ContainerRuntime:
    """
    Executes models in isolated Docker containers with resource limits.

    This class creates Docker containers with mounted code directories,
    applies CPU and memory limits, enables GPU passthrough when available,
    and captures execution output.

    Implements Requirements:
        - 9.1: Create containers with mounted code
        - 9.2: Apply CPU and memory limits
        - 9.3: GPU passthrough via --gpus flag
        - 9.4: Handle OOM termination
        - 9.5: Capture stdout/stderr

    Attributes:
        hardware_detector: Hardware detection for resource limit calculation
        image_name: Docker image name for runtime
    """

    def __init__(
        self,
        hardware_detector: Optional[HardwareDetector] = None,
        image_name: str = "neuroscript-runtime:latest"
    ):
        """
        Initialize ContainerRuntime.

        Args:
            hardware_detector: Hardware detector for resource limits
                (will create new one if not provided)
            image_name: Docker image to use for execution
        """
        self.hardware_detector = hardware_detector or HardwareDetector()
        self.image_name = image_name
        self._verify_docker_available()

    def _verify_docker_available(self) -> None:
        """
        Verify Docker is available on the system.

        Raises:
            ContainerRuntimeError: If Docker is not available
        """
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise ContainerRuntimeError(
                    "Docker is not available or not running. "
                    "Please install Docker and ensure the Docker daemon is running."
                )
        except FileNotFoundError:
            raise ContainerRuntimeError(
                "Docker command not found. Please install Docker."
            )
        except subprocess.TimeoutExpired:
            raise ContainerRuntimeError(
                "Docker command timed out. Docker daemon may not be running."
            )

    def _verify_image_exists(self) -> None:
        """
        Verify runtime Docker image exists.

        Raises:
            ContainerRuntimeError: If image doesn't exist
        """
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise ContainerRuntimeError(
                    f"Docker image '{self.image_name}' not found. "
                    f"Please build the image first using: "
                    f"docker build -t {self.image_name} ."
                )
        except subprocess.TimeoutExpired:
            raise ContainerRuntimeError(
                "Docker image inspection timed out."
            )

    def execute(
        self,
        model_path: Path,
        code_dir: Path,
        command: Optional[List[str]] = None,
        cpu_limit: Optional[float] = None,
        memory_limit: Optional[str] = None,
        enable_gpu: bool = False,
        timeout: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute model in Docker container.

        Implements Req 9.1, 9.2, 9.3, 9.4, 9.5

        Args:
            model_path: Path to the generated model.py file
            code_dir: Directory to mount in container (should contain model and blocks)
            command: Command to execute in container (default: ["python", "model.py"])
            cpu_limit: CPU limit in cores (default: from hardware detection)
            memory_limit: Memory limit string (e.g., "2g", "512m")
            enable_gpu: Whether to enable GPU passthrough
            timeout: Execution timeout in seconds (default: None)
            env_vars: Environment variables to set in container

        Returns:
            ExecutionResult: Execution results including stdout, stderr, exit code

        Raises:
            ContainerRuntimeError: If execution fails
        """
        logger.info(f"Executing {model_path} in container")

        # Verify prerequisites
        self._verify_image_exists()

        # Validate inputs
        if not model_path.exists():
            raise ContainerRuntimeError(f"Model file not found: {model_path}")
        if not code_dir.exists() or not code_dir.is_dir():
            raise ContainerRuntimeError(f"Code directory not found: {code_dir}")

        # Get resource limits
        cpu_limit_str, memory_limit_str = self.get_resource_limits(
            cpu_limit=cpu_limit,
            memory_limit=memory_limit
        )

        # Build command
        if command is None:
            # Get relative path from code_dir to model_path
            try:
                relative_model_path = model_path.relative_to(code_dir)
                command = ["python", str(relative_model_path)]
            except ValueError:
                # model_path is not relative to code_dir
                raise ContainerRuntimeError(
                    f"Model path {model_path} must be inside code directory {code_dir}"
                )

        # Create container
        container_id = self.create_container(
            code_dir=code_dir,
            command=command,
            cpu_limit=cpu_limit_str,
            memory_limit=memory_limit_str,
            enable_gpu=enable_gpu,
            env_vars=env_vars
        )

        try:
            # Run container and capture output
            import time
            start_time = time.time()

            result = subprocess.run(
                ["docker", "start", "-a", container_id],
                capture_output=True,
                timeout=timeout,
                text=True
            )

            execution_time = time.time() - start_time

            # Check if OOM killed
            oom_killed = self._check_oom_killed(container_id)

            if oom_killed:
                self.handle_oom(container_id, memory_limit_str)

            # Get final exit code
            inspect_result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            exit_code = int(inspect_result.stdout.strip()) if inspect_result.returncode == 0 else result.returncode

            return ExecutionResult(
                exit_code=exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                container_id=container_id,
                oom_killed=oom_killed,
                execution_time_seconds=execution_time
            )

        except subprocess.TimeoutExpired:
            # Kill container on timeout
            logger.warning(f"Execution timed out after {timeout}s, killing container")
            subprocess.run(["docker", "kill", container_id], capture_output=True)
            raise ContainerRuntimeError(
                f"Execution timed out after {timeout} seconds"
            )

        finally:
            # Cleanup container
            self._cleanup_container(container_id)

    def create_container(
        self,
        code_dir: Path,
        command: List[str],
        cpu_limit: str,
        memory_limit: str,
        enable_gpu: bool = False,
        env_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create Docker container with resource limits.

        Implements Req 9.1, 9.2, 9.3

        Args:
            code_dir: Directory to mount in container
            command: Command to execute
            cpu_limit: CPU limit string (e.g., "2.0")
            memory_limit: Memory limit string (e.g., "2g")
            enable_gpu: Whether to enable GPU passthrough
            env_vars: Environment variables to set

        Returns:
            str: Container ID

        Raises:
            ContainerRuntimeError: If container creation fails
        """
        docker_cmd = [
            "docker", "create",
            "--cpus", cpu_limit,
            "--memory", memory_limit,
            "--mount", f"type=bind,source={code_dir.absolute()},target=/workspace",
            "--workdir", "/workspace",
        ]

        # Add GPU support if requested and available
        if enable_gpu:
            if self.hardware_detector.capabilities.cuda_available:
                docker_cmd.extend(["--gpus", "all"])
                logger.info("GPU passthrough enabled")
            else:
                logger.warning(
                    "GPU requested but CUDA not available, running on CPU"
                )

        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        docker_cmd.append(self.image_name)
        docker_cmd.extend(command)

        logger.debug(f"Creating container: {' '.join(docker_cmd)}")

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise ContainerRuntimeError(
                    f"Failed to create container: {result.stderr}"
                )

            container_id = result.stdout.strip()
            logger.info(f"Container created: {container_id[:12]}")
            return container_id

        except subprocess.TimeoutExpired:
            raise ContainerRuntimeError("Container creation timed out")

    def get_resource_limits(
        self,
        cpu_limit: Optional[float] = None,
        memory_limit: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Get resource limits based on hardware detection.

        Implements Req 9.2

        Args:
            cpu_limit: CPU limit in cores (optional, auto-detected if not provided)
            memory_limit: Memory limit string (optional, auto-detected if not provided)

        Returns:
            Tuple[str, str]: (cpu_limit_str, memory_limit_str)
                cpu_limit_str: CPU limit as string (e.g., "2.0")
                memory_limit_str: Memory limit string (e.g., "2g")
        """
        # Determine CPU limit
        if cpu_limit is None:
            # Use half of available CPU cores by default
            detected_cores = self.hardware_detector.capabilities.cpu_cores
            cpu_limit = max(1.0, detected_cores / 2.0)
        cpu_limit_str = f"{cpu_limit:.1f}"

        # Determine memory limit
        if memory_limit is None:
            # Use half of available system memory by default
            detected_memory_gb = self.hardware_detector.capabilities.system_memory_gb
            memory_gb = max(1.0, detected_memory_gb / 2.0)
            memory_limit = f"{int(memory_gb)}g"
        else:
            # Validate memory limit format
            if not self._validate_memory_format(memory_limit):
                raise ContainerRuntimeError(
                    f"Invalid memory limit format: {memory_limit}. "
                    f"Expected format: <number>[b|k|m|g] (e.g., '2g', '512m')"
                )

        logger.debug(
            f"Resource limits: CPU={cpu_limit_str} cores, Memory={memory_limit}"
        )

        return cpu_limit_str, memory_limit

    def _validate_memory_format(self, memory_limit: str) -> bool:
        """
        Validate memory limit string format.

        Args:
            memory_limit: Memory limit string to validate

        Returns:
            bool: True if format is valid
        """
        import re
        # Docker memory format: <number>[b|k|m|g]
        pattern = r'^\d+[bkmg]$'
        return bool(re.match(pattern, memory_limit.lower()))

    def _check_oom_killed(self, container_id: str) -> bool:
        """
        Check if container was killed due to OOM.

        Args:
            container_id: Container ID to check

        Returns:
            bool: True if container was OOM killed
        """
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.OOMKilled}}", container_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip().lower() == "true"
        except Exception as e:
            logger.warning(f"Failed to check OOM status: {e}")
            return False

    def handle_oom(self, container_id: str, memory_limit: str) -> None:
        """
        Handle out-of-memory container termination.

        Implements Req 9.4

        Args:
            container_id: Container ID that was OOM killed
            memory_limit: Memory limit that was exceeded

        Raises:
            ContainerRuntimeError: Always raises with OOM information
        """
        logger.error(f"Container {container_id[:12]} was killed due to OOM")

        # Get container stats for diagnostics
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format",
                 "{{.MemUsage}}", container_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            mem_usage = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            mem_usage = "unknown"

        raise ContainerRuntimeError(
            f"Container exceeded memory limit of {memory_limit}. "
            f"Memory usage: {mem_usage}. "
            f"Suggestions:\n"
            f"  1. Increase memory limit using --memory-limit flag\n"
            f"  2. Reduce batch size in model configuration\n"
            f"  3. Use a smaller model variant\n"
            f"  4. Enable gradient checkpointing if training"
        )

    def _cleanup_container(self, container_id: str) -> None:
        """
        Remove container after execution.

        Args:
            container_id: Container ID to remove
        """
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=10
            )
            logger.debug(f"Container {container_id[:12]} removed")
        except Exception as e:
            logger.warning(f"Failed to remove container {container_id[:12]}: {e}")

    def build_runtime_image(self, dockerfile_path: Path) -> None:
        """
        Build the runtime Docker image.

        Args:
            dockerfile_path: Path to Dockerfile

        Raises:
            ContainerRuntimeError: If image build fails
        """
        logger.info(f"Building Docker image: {self.image_name}")

        if not dockerfile_path.exists():
            raise ContainerRuntimeError(
                f"Dockerfile not found: {dockerfile_path}"
            )

        try:
            result = subprocess.run(
                [
                    "docker", "build",
                    "-t", self.image_name,
                    "-f", str(dockerfile_path),
                    str(dockerfile_path.parent)
                ],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for image build
            )

            if result.returncode != 0:
                raise ContainerRuntimeError(
                    f"Failed to build Docker image:\n{result.stderr}"
                )

            logger.info(f"Successfully built image: {self.image_name}")

        except subprocess.TimeoutExpired:
            raise ContainerRuntimeError(
                "Docker image build timed out after 10 minutes"
            )
