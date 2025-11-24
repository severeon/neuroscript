"""ContainerRuntime for executing models in isolated Docker containers with resource limits.

This module executes generated PyTorch models in Docker containers with
configurable CPU, memory, and GPU resource limits.

Implements Requirements 9.1, 9.2, 9.3, 9.4, 9.5 from specs/02-requirements.md
"""

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .hardware_detector import HardwareDetector
from .monitoring import MonitorEvent, ResourceSample, parse_event_line

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
        resource_samples: Periodic resource usage samples during execution
        monitoring_events: Parsed monitoring events from model execution
    """

    exit_code: int
    stdout: str
    stderr: str
    container_id: str
    oom_killed: bool
    execution_time_seconds: float
    resource_samples: List[ResourceSample] = field(default_factory=list)
    monitoring_events: List[MonitorEvent] = field(default_factory=list)


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
        env_vars: Optional[Dict[str, str]] = None,
        stream_output: bool = True,
        output_callback: Optional[Callable[[str], None]] = None
    ) -> ExecutionResult:
        """
        Execute model in Docker container with resource monitoring.

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
            stream_output: Whether to stream output line-by-line (default: True)
            output_callback: Callback function for each output line (default: None)

        Returns:
            ExecutionResult: Execution results including stdout, stderr, exit code,
                resource samples, and monitoring events

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

        # Start resource monitoring in background thread
        resource_samples: List[ResourceSample] = []
        monitoring_events: List[MonitorEvent] = []
        stop_monitoring = threading.Event()

        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(container_id, resource_samples, stop_monitoring, enable_gpu),
            daemon=True
        )

        try:
            # Start monitoring thread
            monitor_thread.start()

            # Run container and capture output
            start_time = time.time()

            if stream_output and output_callback:
                # Stream output line-by-line
                stdout_lines, stderr_lines = self._stream_container_output(
                    container_id,
                    timeout,
                    output_callback,
                    monitoring_events
                )
                stdout = "\n".join(stdout_lines)
                stderr = "\n".join(stderr_lines)
                exit_code = self._get_container_exit_code(container_id)
            else:
                # Buffered output (original behavior)
                result = subprocess.run(
                    ["docker", "start", "-a", container_id],
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )
                stdout = result.stdout
                stderr = result.stderr
                exit_code = result.returncode

                # Parse monitoring events from stdout
                for line in stdout.splitlines():
                    event = parse_event_line(line)
                    if event:
                        monitoring_events.append(event)

            execution_time = time.time() - start_time

            # Stop resource monitoring
            stop_monitoring.set()
            monitor_thread.join(timeout=2.0)

            # Check if OOM killed
            oom_killed = self._check_oom_killed(container_id)

            if oom_killed:
                self.handle_oom(container_id, memory_limit_str)

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                container_id=container_id,
                oom_killed=oom_killed,
                execution_time_seconds=execution_time,
                resource_samples=resource_samples,
                monitoring_events=monitoring_events
            )

        except subprocess.TimeoutExpired:
            # Kill container on timeout
            logger.warning(f"Execution timed out after {timeout}s, killing container")
            stop_monitoring.set()
            subprocess.run(["docker", "kill", container_id], capture_output=True)
            raise ContainerRuntimeError(
                f"Execution timed out after {timeout} seconds"
            )

        finally:
            # Ensure monitoring stopped
            stop_monitoring.set()
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

    def _monitor_resources(
        self,
        container_id: str,
        samples: List[ResourceSample],
        stop_event: threading.Event,
        enable_gpu: bool = False
    ) -> None:
        """
        Monitor container resource usage in background thread.

        Polls docker stats every second and records CPU, memory, and GPU usage.

        Args:
            container_id: Container ID to monitor
            samples: List to append resource samples to
            stop_event: Event to signal monitoring should stop
            enable_gpu: Whether to query GPU stats
        """
        while not stop_event.is_set():
            try:
                # Get CPU and memory stats from docker stats
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format",
                     "{{.CPUPerc}},{{.MemUsage}}", container_id],
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode == 0:
                    stats = result.stdout.strip()
                    if stats:
                        # Parse docker stats output
                        # Format: "15.23%,1.5GiB / 2GiB"
                        parts = stats.split(',')
                        if len(parts) >= 2:
                            cpu_str = parts[0].strip().rstrip('%')
                            mem_str = parts[1].strip()

                            try:
                                cpu_percent = float(cpu_str)
                            except ValueError:
                                cpu_percent = 0.0

                            # Parse memory usage: "1.5GiB / 2GiB"
                            mem_parts = mem_str.split('/')
                            if len(mem_parts) >= 2:
                                mem_used_str = mem_parts[0].strip()
                                mem_limit_str = mem_parts[1].strip()

                                mem_used_mb = self._parse_memory_to_mb(mem_used_str)
                                mem_limit_mb = self._parse_memory_to_mb(mem_limit_str)

                                mem_percent = (mem_used_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0.0
                            else:
                                mem_used_mb = 0.0
                                mem_percent = 0.0

                            # Get GPU stats if enabled
                            gpu_util = None
                            gpu_mem = None
                            if enable_gpu:
                                try:
                                    gpu_result = subprocess.run(
                                        ["docker", "exec", container_id,
                                         "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                         "--format=csv,noheader,nounits"],
                                        capture_output=True,
                                        text=True,
                                        timeout=2
                                    )
                                    if gpu_result.returncode == 0:
                                        gpu_stats = gpu_result.stdout.strip().split(',')
                                        if len(gpu_stats) >= 2:
                                            gpu_util = float(gpu_stats[0].strip())
                                            gpu_mem = float(gpu_stats[1].strip())
                                except Exception:
                                    pass  # GPU monitoring is best-effort

                            sample = ResourceSample(
                                timestamp=time.time(),
                                cpu_percent=cpu_percent,
                                memory_mb=mem_used_mb,
                                memory_percent=mem_percent,
                                gpu_utilization=gpu_util,
                                gpu_memory_mb=gpu_mem
                            )
                            samples.append(sample)

            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")

            # Sample every 1 second
            stop_event.wait(1.0)

    def _parse_memory_to_mb(self, mem_str: str) -> float:
        """
        Parse Docker memory string to megabytes.

        Args:
            mem_str: Memory string like "1.5GiB", "512MiB", "1024KiB"

        Returns:
            Memory in megabytes
        """
        mem_str = mem_str.strip()
        match = re.match(r'([0-9.]+)\s*([A-Za-z]+)', mem_str)
        if not match:
            return 0.0

        value = float(match.group(1))
        unit = match.group(2).lower()

        # Convert to MB
        if unit in ['gib', 'gb', 'g']:
            return value * 1024
        elif unit in ['mib', 'mb', 'm']:
            return value
        elif unit in ['kib', 'kb', 'k']:
            return value / 1024
        elif unit in ['b']:
            return value / (1024 * 1024)
        else:
            return value

    def _stream_container_output(
        self,
        container_id: str,
        timeout: Optional[int],
        output_callback: Callable[[str], None],
        monitoring_events: List[MonitorEvent]
    ) -> tuple[List[str], List[str]]:
        """
        Stream container output line-by-line and call callback.

        Args:
            container_id: Container ID
            timeout: Execution timeout
            output_callback: Function to call for each output line
            monitoring_events: List to append parsed events to

        Returns:
            Tuple of (stdout_lines, stderr_lines)
        """
        # Start container in detached mode
        subprocess.run(["docker", "start", container_id], capture_output=True)

        stdout_lines = []
        stderr_lines = []

        # Follow logs
        start_time = time.time()
        process = subprocess.Popen(
            ["docker", "logs", "-f", container_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        try:
            while True:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    process.kill()
                    raise subprocess.TimeoutExpired(process.args, timeout)

                # Read stdout line
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        line = line.rstrip('\n')
                        stdout_lines.append(line)

                        # Parse monitoring events
                        event = parse_event_line(line)
                        if event:
                            monitoring_events.append(event)

                        # Call output callback
                        output_callback(line)

                # Check if process finished
                if process.poll() is not None:
                    # Read remaining lines
                    if process.stdout:
                        for line in process.stdout:
                            line = line.rstrip('\n')
                            stdout_lines.append(line)
                            event = parse_event_line(line)
                            if event:
                                monitoring_events.append(event)
                            output_callback(line)
                    break

                time.sleep(0.01)  # Small sleep to avoid busy-waiting

        finally:
            if process.poll() is None:
                process.kill()
            process.wait()

        return stdout_lines, stderr_lines

    def _get_container_exit_code(self, container_id: str) -> int:
        """
        Get container exit code.

        Args:
            container_id: Container ID

        Returns:
            Exit code (0 for success)
        """
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 1
        except Exception:
            return 1

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
