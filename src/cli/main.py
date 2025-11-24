#!/usr/bin/env python3
"""NeuroScript CLI interface."""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Core imports
from core.block_registry import BlockRegistry
from core.constraint_solver import Configuration, ConstraintSolver
from core.graph_loader import GraphLoader
from core.graph_validator import GraphValidator
from core.compilation_engine import CompilationEngine
from core.direct_runtime import DirectRuntime
from core.container_runtime import ContainerRuntime
from core.hardware_detector import HardwareDetector
from core.shape_validator import ShapeValidator
from core.monitoring import (
    ExecutionMetrics,
    MonitorEvent,
    ResourceSample,
    format_event,
    format_execution_summary,
    parse_event_line
)

logger = logging.getLogger()

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description='NeuroScript: Validate, compile, and run neural architectures.'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Validate command
    validate_parser = subparsers.add_parser(
        'validate', help='Validate architecture graph'
    )
    validate_parser.add_argument(
        'architecture', type=Path,
        help='Path to architecture YAML file'
    )
    validate_parser.add_argument(
        '--input-shape', nargs='+', type=int, default=None,
        help='Input shape for constraint solving (e.g., 32 128 512)'
    )

    # Compile command
    compile_parser = subparsers.add_parser(
        'compile', help='Compile architecture to PyTorch module'
    )
    compile_parser.add_argument(
        'architecture', type=Path,
        help='Path to architecture YAML file'
    )
    compile_parser.add_argument(
        '--input-shape', nargs='+', type=int, default=None,
        help='Input shape for constraint solving (e.g., 32 128 512)'
    )
    compile_parser.add_argument(
        '--output', '-o', type=Path, default='model.py',
        help='Output PyTorch module file (default: model.py)'
    )

    # Run command
    run_parser = subparsers.add_parser('run', help='Run compiled model')
    run_parser.add_argument('model', type=Path, help='Path to compiled PyTorch model')
    run_parser.add_argument('--local', action='store_true', default=True,
                            help='Run locally without Docker (default: True)')
    run_parser.add_argument('--container', dest='local', action='store_false',
                            help='Run in Docker container instead of locally')
    run_parser.add_argument('--cpu-limit', type=float, default=1.0, help='CPU limit (cores)')
    run_parser.add_argument('--memory-limit', type=str, default='1g', help='Memory limit (e.g., 1g, 512m)')
    run_parser.add_argument('--mode', choices=['inference', 'training'], default='inference',
                            help='Execution mode (default: inference)')
    run_parser.add_argument('--epochs', type=int, default=5,
                            help='Number of training epochs (default: 5)')
    run_parser.add_argument('--batch-size', type=int, default=32,
                            help='Training batch size (default: 32)')
    run_parser.add_argument('--save-logs', type=str, default=None,
                            help='Save execution logs to file (default: neuroscript_run_<timestamp>.log if not specified)')
    run_parser.add_argument('--enable-save-logs', action='store_true',
                            help='Enable log saving with automatic filename')
    run_parser.add_argument('--quiet', action='store_true',
                            help='Suppress monitoring output, only show final results')

    args = parser.parse_args()

    if args.command == 'validate':
        _run_validate(args)
    elif args.command == 'compile':
        _run_compile(args)
    elif args.command == 'run':
        _run_run(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_validate(args):
    registry = BlockRegistry()
    hardware = HardwareDetector()
    loader = GraphLoader(registry)
    shapeValidator = ShapeValidator()
    graph = loader.load(args.architecture)
    constraint_solver = ConstraintSolver(graph=graph, registry=registry)
    validator = GraphValidator(graph, registry, hardware, shape_validator=shapeValidator, constraint_solver=constraint_solver)
    result = validator.validate()
    if result.valid:
        logger.debug(result)
        print('✅ Architecture is valid!')
    else:
        print('❌ Validation failed:')
        for error in result.errors:
            print(f'  - {error.message}')
        sys.exit(1)


def _run_compile(args):
    registry = BlockRegistry()
    hardware = HardwareDetector()
    loader = GraphLoader(registry)
    shapeValidator = ShapeValidator()
    graph = loader.load(args.architecture)
    constraint_solver = ConstraintSolver(graph=graph, registry=registry)
    validator = GraphValidator(graph, registry, hardware, shape_validator=shapeValidator, constraint_solver=constraint_solver)
    result = validator.validate()
    if not result.valid:
        print('❌ Validation failed. Cannot compile.')
        for error in result.errors:
            print(f'  - {error.message}')
        sys.exit(1)

    engine = CompilationEngine(graph=graph, config=Configuration(), registry=registry)
    engine.compile(Path(args.output))
    print(f'✅ Compiled to {args.output}')


def _run_run(args):
    """Execute model with monitoring and optional log saving."""

    # Determine log file path
    log_file = None
    if args.save_logs:
        log_file = Path(args.save_logs)
    elif args.enable_save_logs:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(f'neuroscript_run_{timestamp}.log')

    # Prepare log data
    log_data = {
        'command': {
            'model': str(args.model),
            'mode': args.mode,
            'cpu_limit': args.cpu_limit,
            'memory_limit': args.memory_limit,
            'epochs': args.epochs if args.mode == 'training' else None,
            'batch_size': args.batch_size if args.mode == 'training' else None
        },
        'start_time': datetime.now().isoformat(),
        'output_lines': [],
        'monitoring_events': [],
        'resource_samples': []
    }

    # Create output callback for streaming
    def output_callback(line: str):
        """Handle each line of output."""
        log_data['output_lines'].append(line)

        # Parse monitoring event
        event = parse_event_line(line)
        if event:
            # Format and display monitoring event
            if not args.quiet:
                formatted = format_event(event)
                print(formatted)
        else:
            # Regular output line (not a monitoring event)
            if not args.quiet and not line.startswith('NEUROSCRIPT_EVENT:'):
                print(line)

    # Build command with mode arguments
    command = ["python", str(args.model), "--mode", args.mode]
    if args.mode == 'training':
        command.extend(["--epochs", str(args.epochs)])
        command.extend(["--batch-size", str(args.batch_size)])

    # Choose runtime based on --local flag
    if args.local:
        # Direct execution on host
        runtime = DirectRuntime()
        result = runtime.execute(
            args.model,
            code_dir=Path("."),
            command=command,
            cpu_limit=args.cpu_limit,
            memory_limit=args.memory_limit,
            stream_output=True,
            output_callback=output_callback
        )
    else:
        # Container execution with Docker
        runtime = ContainerRuntime()
        result = runtime.execute(
            args.model,
            code_dir=Path("."),
            command=command,
            cpu_limit=args.cpu_limit,
            memory_limit=args.memory_limit,
            stream_output=True,
            output_callback=output_callback
        )

    # Store execution results in log
    log_data['end_time'] = datetime.now().isoformat()
    log_data['exit_code'] = result.exit_code
    log_data['execution_time_seconds'] = result.execution_time_seconds
    log_data['oom_killed'] = result.oom_killed

    # Convert monitoring events to serializable format
    log_data['monitoring_events'] = [
        {
            'type': event.type,
            'timestamp': event.timestamp,
            'data': event.data
        }
        for event in result.monitoring_events
    ]

    # Convert resource samples to serializable format
    log_data['resource_samples'] = [
        {
            'timestamp': sample.timestamp,
            'cpu_percent': sample.cpu_percent,
            'memory_mb': sample.memory_mb,
            'memory_percent': sample.memory_percent,
            'gpu_utilization': sample.gpu_utilization,
            'gpu_memory_mb': sample.gpu_memory_mb
        }
        for sample in result.resource_samples
    ]

    # Calculate execution metrics
    metrics = _calculate_execution_metrics(result)
    log_data['metrics'] = {
        'total_time_seconds': metrics.total_time_seconds,
        'peak_memory_mb': metrics.peak_memory_mb,
        'peak_cpu_percent': metrics.peak_cpu_percent,
        'peak_gpu_utilization': metrics.peak_gpu_utilization,
        'peak_gpu_memory_mb': metrics.peak_gpu_memory_mb,
        'final_train_loss': metrics.final_train_loss,
        'final_val_loss': metrics.final_val_loss,
        'final_train_accuracy': metrics.final_train_accuracy,
        'final_val_accuracy': metrics.final_val_accuracy,
        'layer_timings': metrics.layer_timings
    }

    # Display execution summary
    if not args.quiet:
        summary = format_execution_summary(metrics)
        print(summary)

    # Save logs to file if requested
    if log_file:
        try:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"\nLogs saved to: {log_file}")
        except IOError as e:
            print(f"Warning: Failed to save logs to {log_file}: {e}", file=sys.stderr)

    # Handle stderr
    if result.stderr and not args.quiet:
        print('\nSTDERR:', result.stderr, file=sys.stderr)

    sys.exit(result.exit_code)


def _calculate_execution_metrics(result) -> ExecutionMetrics:
    """Calculate execution metrics from execution result."""

    # Calculate peak resource usage
    peak_memory_mb = 0.0
    peak_cpu_percent = 0.0
    peak_gpu_utilization = None
    peak_gpu_memory_mb = None

    for sample in result.resource_samples:
        peak_memory_mb = max(peak_memory_mb, sample.memory_mb)
        peak_cpu_percent = max(peak_cpu_percent, sample.cpu_percent)
        if sample.gpu_utilization is not None:
            if peak_gpu_utilization is None:
                peak_gpu_utilization = sample.gpu_utilization
            else:
                peak_gpu_utilization = max(peak_gpu_utilization, sample.gpu_utilization)
        if sample.gpu_memory_mb is not None:
            if peak_gpu_memory_mb is None:
                peak_gpu_memory_mb = sample.gpu_memory_mb
            else:
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, sample.gpu_memory_mb)

    # Extract training metrics from monitoring events
    final_train_loss = None
    final_val_loss = None
    final_train_accuracy = None
    final_val_accuracy = None

    for event in reversed(result.monitoring_events):
        if event.type == 'epoch_complete':
            if final_train_loss is None:
                final_train_loss = event.data.get('train_loss')
            if final_val_loss is None:
                final_val_loss = event.data.get('val_loss')
            metrics = event.data.get('metrics', {})
            if final_train_accuracy is None:
                final_train_accuracy = metrics.get('accuracy')
            break

    # Calculate average layer timings
    layer_timings = {}
    layer_counts = {}

    for event in result.monitoring_events:
        if event.type == 'block_forward':
            block_name = event.data.get('block_name')
            time_ms = event.data.get('time_ms', 0.0)
            if block_name:
                if block_name not in layer_timings:
                    layer_timings[block_name] = 0.0
                    layer_counts[block_name] = 0
                layer_timings[block_name] += time_ms
                layer_counts[block_name] += 1

    # Calculate averages
    for block_name in layer_timings:
        if layer_counts[block_name] > 0:
            layer_timings[block_name] /= layer_counts[block_name]

    return ExecutionMetrics(
        total_time_seconds=result.execution_time_seconds,
        peak_memory_mb=peak_memory_mb,
        peak_cpu_percent=peak_cpu_percent,
        peak_gpu_utilization=peak_gpu_utilization,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        final_train_accuracy=final_train_accuracy,
        final_val_accuracy=final_val_accuracy,
        layer_timings=layer_timings
    )


if __name__ == '__main__':
    main()
