"""DirectRuntime for executing models directly on the host system.

This module executes generated PyTorch models directly using subprocess,
without Docker containers. This is the default execution mode for compatibility
with all platforms including arm64/MPS systems.

Provides resource monitoring via psutil when available.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .monitoring import MonitorEvent, ResourceSample, parse_event_line

# Optional psutil for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of direct model execution.

    Attributes:
        exit_code: Process exit code (0 for success)
        stdout: Standard output captured from process
        stderr: Standard error captured from process
        execution_time_seconds: Time taken for execution
        resource_samples: Periodic resource usage samples during execution
        monitoring_events: Parsed monitoring events from model execution
        container_id: Always empty for direct execution (for compatibility)
        oom_killed: Always False for direct execution (for compatibility)
    """

    exit_code: int
    stdout: str
    stderr: str
    execution_time_seconds: float
    resource_samples: List[ResourceSample] = field(default_factory=list)
    monitoring_events: List[MonitorEvent] = field(default_factory=list)
    container_id: str = ""  # For compatibility with ContainerRuntime
    oom_killed: bool = False  # For compatibility with ContainerRuntime


class DirectRuntimeError(Exception):
    """Raised when direct execution fails."""

    pass


class DirectRuntime:
    """
    Executes models directly on the host system without containers.

    This class runs PyTorch models using subprocess.Popen and monitors
    resource usage via psutil when available. This is the default execution
    mode for compatibility with all platforms.

    Features:
        - Direct subprocess execution
        - Real-time output streaming
        - Resource monitoring (CPU, memory, GPU via nvidia-smi)
        - Monitoring event parsing
        - No Docker dependencies

    Attributes:
        python_executable: Path to Python interpreter (default: sys.executable)
    """

    def __init__(self, python_executable: Optional[str] = None):
        """
        Initialize DirectRuntime.

        Args:
            python_executable: Path to Python interpreter to use
                (default: sys.executable)
        """
        self.python_executable = python_executable or sys.executable

        if not PSUTIL_AVAILABLE:
            logger.warning(
                "psutil not installed. Resource monitoring will be limited. "
                "Install with: pip install psutil"
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
        Execute model directly on host system with resource monitoring.

        Args:
            model_path: Path to the generated model.py file
            code_dir: Working directory for execution
            command: Command to execute (default: ["python", str(model_path)])
            cpu_limit: CPU limit in cores (informational only, not enforced)
            memory_limit: Memory limit string (informational only, not enforced)
            enable_gpu: Whether GPU is available (informational only)
            timeout: Execution timeout in seconds (default: None)
            env_vars: Environment variables to set for the process
            stream_output: Whether to stream output line-by-line (default: True)
            output_callback: Callback function for each output line (default: None)

        Returns:
            ExecutionResult: Execution results including stdout, stderr, exit code,
                resource samples, and monitoring events

        Raises:
            DirectRuntimeError: If execution fails
        """
        logger.info(f"Executing {model_path} directly on host")

        # Validate inputs
        if not model_path.exists():
            raise DirectRuntimeError(f"Model file not found: {model_path}")
        if not code_dir.exists() or not code_dir.is_dir():
            raise DirectRuntimeError(f"Code directory not found: {code_dir}")

        # Build command
        if command is None:
            command = [self.python_executable, str(model_path)]

        # Log resource limits (informational only in direct mode)
        if cpu_limit:
            logger.info(f"CPU limit requested: {cpu_limit} cores (not enforced in direct mode)")
        if memory_limit:
            logger.info(f"Memory limit requested: {memory_limit} (not enforced in direct mode)")

        # Prepare environment
        exec_env = os.environ.copy()
        if env_vars:
            exec_env.update(env_vars)

        # Start resource monitoring in background thread
        resource_samples: List[ResourceSample] = []
        monitoring_events: List[MonitorEvent] = []
        stop_monitoring = threading.Event()
        process_handle = None

        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(lambda: process_handle, resource_samples, stop_monitoring, enable_gpu),
            daemon=True
        )

        try:
            # Start monitoring thread
            monitor_thread.start()

            # Execute model
            start_time = time.time()

            if stream_output and output_callback:
                # Stream output line-by-line
                stdout_lines, stderr_lines, exit_code, process_handle = self._stream_process_output(
                    command,
                    code_dir,
                    exec_env,
                    timeout,
                    output_callback,
                    monitoring_events
                )
                stdout = "\n".join(stdout_lines)
                stderr = "\n".join(stderr_lines)
            else:
                # Buffered output
                process_handle = subprocess.Popen(
                    command,
                    cwd=code_dir,
                    env=exec_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                try:
                    stdout, stderr = process_handle.communicate(timeout=timeout)
                    exit_code = process_handle.returncode

                    # Parse monitoring events from stdout
                    for line in stdout.splitlines():
                        event = parse_event_line(line)
                        if event:
                            monitoring_events.append(event)

                except subprocess.TimeoutExpired:
                    process_handle.kill()
                    stdout, stderr = process_handle.communicate()
                    raise DirectRuntimeError(
                        f"Execution timed out after {timeout} seconds"
                    )

            execution_time = time.time() - start_time

            # Stop resource monitoring
            stop_monitoring.set()
            monitor_thread.join(timeout=2.0)

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time_seconds=execution_time,
                resource_samples=resource_samples,
                monitoring_events=monitoring_events
            )

        except subprocess.TimeoutExpired:
            logger.warning(f"Execution timed out after {timeout}s")
            stop_monitoring.set()
            if process_handle:
                process_handle.kill()
            raise DirectRuntimeError(
                f"Execution timed out after {timeout} seconds"
            )

        finally:
            # Ensure monitoring stopped
            stop_monitoring.set()

    def _stream_process_output(
        self,
        command: List[str],
        cwd: Path,
        env: Dict[str, str],
        timeout: Optional[int],
        output_callback: Callable[[str], None],
        monitoring_events: List[MonitorEvent]
    ) -> tuple[List[str], List[str], int, subprocess.Popen]:
        """
        Execute process and stream output line-by-line.

        Args:
            command: Command to execute
            cwd: Working directory
            env: Environment variables
            timeout: Execution timeout
            output_callback: Function to call for each output line
            monitoring_events: List to append parsed events to

        Returns:
            Tuple of (stdout_lines, stderr_lines, exit_code, process)
        """
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        stdout_lines = []
        stderr_lines = []

        start_time = time.time()

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
                    if process.stderr:
                        for line in process.stderr:
                            line = line.rstrip('\n')
                            stderr_lines.append(line)
                    break

                time.sleep(0.01)  # Small sleep to avoid busy-waiting

        finally:
            if process.poll() is None:
                process.kill()
            process.wait()

        return stdout_lines, stderr_lines, process.returncode, process

    def _monitor_resources(
        self,
        process_getter: Callable[[], Optional[subprocess.Popen]],
        samples: List[ResourceSample],
        stop_event: threading.Event,
        enable_gpu: bool = False
    ) -> None:
        """
        Monitor process resource usage in background thread.

        Uses psutil to collect CPU and memory stats every second.

        Args:
            process_getter: Function that returns the process handle
            samples: List to append resource samples to
            stop_event: Event to signal monitoring should stop
            enable_gpu: Whether to query GPU stats via nvidia-smi
        """
        if not PSUTIL_AVAILABLE:
            logger.debug("psutil not available, skipping resource monitoring")
            return

        while not stop_event.is_set():
            try:
                process = process_getter()
                if process is None or process.poll() is not None:
                    # Process hasn't started yet or has finished
                    stop_event.wait(0.5)
                    continue

                # Get process stats via psutil
                try:
                    ps_process = psutil.Process(process.pid)

                    # CPU percent (averaged over interval)
                    cpu_percent = ps_process.cpu_percent(interval=0.1)

                    # Memory usage
                    mem_info = ps_process.memory_info()
                    mem_used_mb = mem_info.rss / (1024 * 1024)  # Convert to MB

                    # Get total system memory for percentage
                    mem_total_mb = psutil.virtual_memory().total / (1024 * 1024)
                    mem_percent = (mem_used_mb / mem_total_mb * 100) if mem_total_mb > 0 else 0.0

                    # Get GPU stats if enabled
                    gpu_util = None
                    gpu_mem = None
                    if enable_gpu:
                        try:
                            result = subprocess.run(
                                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            if result.returncode == 0:
                                gpu_stats = result.stdout.strip().split(',')
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

                except psutil.NoSuchProcess:
                    # Process ended
                    break
                except psutil.AccessDenied:
                    logger.warning("Access denied when monitoring process")
                    break

            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")

            # Sample every 1 second
            stop_event.wait(1.0)
