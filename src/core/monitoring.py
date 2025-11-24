"""
Monitoring protocol for NeuroScript model execution.

Provides structured event emission and timing utilities for tracking
model execution, resource usage, and training progress.
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager


@dataclass
class MonitorEvent:
    """
    Structured monitoring event emitted during model execution.

    All events are JSON-serializable for parsing by CLI and logging tools.
    """
    type: str  # Event type: execution_start, block_forward, epoch_start, etc.
    timestamp: float  # Unix timestamp (seconds since epoch)
    data: Dict[str, Any]  # Event-specific data

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'MonitorEvent':
        """Deserialize event from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class ResourceSample:
    """
    Resource usage snapshot at a point in time.

    Captured periodically during execution by ContainerRuntime.
    """
    timestamp: float  # Unix timestamp
    cpu_percent: float  # CPU utilization (0-100 per core)
    memory_mb: float  # Memory usage in MB
    memory_percent: float  # Memory usage as percentage of limit
    gpu_utilization: Optional[float] = None  # GPU utilization (0-100)
    gpu_memory_mb: Optional[float] = None  # GPU memory usage in MB


@dataclass
class ExecutionMetrics:
    """
    Aggregated metrics from a model execution run.

    Includes timing, resource usage peaks, and training metrics.
    """
    total_time_seconds: float
    peak_memory_mb: float
    peak_cpu_percent: float
    peak_gpu_utilization: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None

    # Training metrics (if applicable)
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    final_train_accuracy: Optional[float] = None
    final_val_accuracy: Optional[float] = None

    # Timing breakdown
    layer_timings: Dict[str, float] = field(default_factory=dict)  # layer_name -> avg_time_ms


def emit_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Emit a monitoring event to stdout as JSON.

    Events are prefixed with 'NEUROSCRIPT_EVENT:' for parsing by CLI.
    Stdout is flushed immediately to ensure real-time delivery.

    Args:
        event_type: Type of event (execution_start, block_forward, etc.)
        data: Event-specific data dictionary

    Example:
        emit_event('block_forward', {
            'block_name': 'encoder',
            'input_shape': [32, 512],
            'output_shape': [32, 256],
            'time_ms': 15.3
        })
    """
    event = MonitorEvent(
        type=event_type,
        timestamp=time.time(),
        data=data
    )
    # Prefix with marker for easy parsing
    print(f"NEUROSCRIPT_EVENT:{event.to_json()}", flush=True)


def format_event(event: MonitorEvent) -> str:
    """
    Format a monitoring event as human-readable text for CLI display.

    Args:
        event: MonitorEvent to format

    Returns:
        Formatted string for terminal display

    Example output:
        [12:34:56] Block 'encoder' forward: (32, 512) → (32, 256) [15.3ms]
        [12:34:57] Epoch 1/10 | Batch 5/100 | Loss: 0.452 | Acc: 0.873
    """
    timestamp = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
    data = event.data

    if event.type == 'execution_start':
        mode = data.get('mode', 'inference')
        input_shape = data.get('input_shape', 'unknown')
        return f"[{timestamp}] Starting {mode} | Input shape: {input_shape}"

    elif event.type == 'block_forward':
        name = data.get('block_name', 'unknown')
        input_shape = tuple(data.get('input_shape', []))
        output_shape = tuple(data.get('output_shape', []))
        time_ms = data.get('time_ms', 0.0)
        return f"[{timestamp}] Block '{name}': {input_shape} → {output_shape} [{time_ms:.1f}ms]"

    elif event.type == 'epoch_start':
        epoch = data.get('epoch', 0)
        total = data.get('total_epochs', 0)
        return f"[{timestamp}] Epoch {epoch}/{total} started"

    elif event.type == 'batch_complete':
        epoch = data.get('epoch', 0)
        batch = data.get('batch', 0)
        total_batches = data.get('total_batches', 0)
        loss = data.get('loss', 0.0)
        metrics = data.get('metrics', {})

        # Format metrics
        metric_str = " | ".join([
            f"{k.capitalize()}: {v:.3f}" for k, v in metrics.items()
        ])
        base = f"[{timestamp}] Epoch {epoch} | Batch {batch}/{total_batches} | Loss: {loss:.4f}"
        if metric_str:
            return f"{base} | {metric_str}"
        return base

    elif event.type == 'epoch_complete':
        epoch = data.get('epoch', 0)
        total = data.get('total_epochs', 0)
        train_loss = data.get('train_loss', 0.0)
        val_loss = data.get('val_loss')
        metrics = data.get('metrics', {})

        base = f"[{timestamp}] Epoch {epoch}/{total} complete | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            base += f" | Val Loss: {val_loss:.4f}"
        if metrics:
            metric_str = " | ".join([
                f"{k.capitalize()}: {v:.3f}" for k, v in metrics.items()
            ])
            base += f" | {metric_str}"
        return base

    elif event.type == 'metric_update':
        metrics = data.get('metrics', {})
        metric_str = " | ".join([
            f"{k}: {v:.4f}" for k, v in metrics.items()
        ])
        return f"[{timestamp}] Metrics: {metric_str}"

    elif event.type == 'execution_end':
        total_time = data.get('total_time_seconds', 0.0)
        success = data.get('success', True)
        status = "completed successfully" if success else "failed"
        return f"[{timestamp}] Execution {status} | Total time: {total_time:.2f}s"

    else:
        # Generic format for unknown event types
        return f"[{timestamp}] {event.type}: {data}"


@contextmanager
def timing_context(block_name: str, emit_events: bool = True):
    """
    Context manager for timing a block's forward pass.

    Args:
        block_name: Name of the block being timed
        emit_events: If True, emit monitoring events (default: True)

    Yields:
        Dictionary for storing input/output shape information

    Example:
        with timing_context('encoder', emit_events=True) as ctx:
            ctx['input_shape'] = list(x.shape)
            output = self.encoder(x)
            ctx['output_shape'] = list(output.shape)
    """
    context = {}
    start_time = time.time()

    yield context

    elapsed_ms = (time.time() - start_time) * 1000

    if emit_events:
        emit_event('block_forward', {
            'block_name': block_name,
            'input_shape': context.get('input_shape', []),
            'output_shape': context.get('output_shape', []),
            'time_ms': elapsed_ms
        })


def parse_event_line(line: str) -> Optional[MonitorEvent]:
    """
    Parse a line of output to extract monitoring event.

    Args:
        line: Line of stdout/stderr from model execution

    Returns:
        MonitorEvent if line contains event, None otherwise

    Example:
        line = "NEUROSCRIPT_EVENT:{...json...}"
        event = parse_event_line(line)
    """
    prefix = "NEUROSCRIPT_EVENT:"
    if line.startswith(prefix):
        json_str = line[len(prefix):].strip()
        try:
            return MonitorEvent.from_json(json_str)
        except (json.JSONDecodeError, TypeError, KeyError):
            # Invalid event format - ignore
            return None
    return None


def format_execution_summary(metrics: ExecutionMetrics) -> str:
    """
    Format execution summary for display at end of run.

    Args:
        metrics: Aggregated execution metrics

    Returns:
        Multi-line formatted summary string

    Example output:
        Execution Summary:
        ─────────────────
        Total Time: 45.3s
        Peak Memory: 1024.5 MB
        Peak CPU: 87.3%
        Peak GPU: 95.2% (2048.0 MB)

        Final Metrics:
          Train Loss: 0.234
          Val Loss: 0.267
          Train Accuracy: 0.912
          Val Accuracy: 0.889
    """
    lines = [
        "\nExecution Summary:",
        "─────────────────",
        f"Total Time: {metrics.total_time_seconds:.1f}s",
        f"Peak Memory: {metrics.peak_memory_mb:.1f} MB",
        f"Peak CPU: {metrics.peak_cpu_percent:.1f}%"
    ]

    if metrics.peak_gpu_utilization is not None:
        gpu_mem = metrics.peak_gpu_memory_mb or 0.0
        lines.append(
            f"Peak GPU: {metrics.peak_gpu_utilization:.1f}% ({gpu_mem:.1f} MB)"
        )

    # Add training metrics if present
    has_metrics = any([
        metrics.final_train_loss is not None,
        metrics.final_val_loss is not None,
        metrics.final_train_accuracy is not None,
        metrics.final_val_accuracy is not None
    ])

    if has_metrics:
        lines.append("\nFinal Metrics:")
        if metrics.final_train_loss is not None:
            lines.append(f"  Train Loss: {metrics.final_train_loss:.4f}")
        if metrics.final_val_loss is not None:
            lines.append(f"  Val Loss: {metrics.final_val_loss:.4f}")
        if metrics.final_train_accuracy is not None:
            lines.append(f"  Train Accuracy: {metrics.final_train_accuracy:.4f}")
        if metrics.final_val_accuracy is not None:
            lines.append(f"  Val Accuracy: {metrics.final_val_accuracy:.4f}")

    # Add layer timing breakdown if available
    if metrics.layer_timings:
        lines.append("\nLayer Timings (avg):")
        for layer_name, avg_time_ms in sorted(metrics.layer_timings.items()):
            lines.append(f"  {layer_name}: {avg_time_ms:.1f}ms")

    return "\n".join(lines)
