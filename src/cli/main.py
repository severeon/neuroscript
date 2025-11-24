#!/usr/bin/env python3
"""NeuroScript CLI interface."""
import argparse
from pathlib import Path
import sys

# Core imports (adjust paths if needed)
from core.graph_loader import GraphLoader
from core.graph_validator import GraphValidator
from core.compilation_engine import CompilationEngine
from core.container_runtime import ContainerRuntime


def main():
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
    run_parser = subparsers.add_parser('run', help='Run compiled model in container')
    run_parser.add_argument('model', type=Path, help='Path to compiled PyTorch model')
    run_parser.add_argument('--cpu-limit', type=float, default=1.0, help='CPU limit')
    run_parser.add_argument('--memory-limit', type=str, default='1g', help='Memory limit')

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
    loader = GraphLoader()
    graph = loader.load(args.architecture)
    validator = GraphValidator()
    result = validator.validate(graph, input_shape=args.input_shape)
    if result.valid:
        print('✅ Architecture is valid!')
    else:
        print('❌ Validation failed:')
        for error in result.errors:
            print(f'  - {error.message}')
        sys.exit(1)


def _run_compile(args):
    loader = GraphLoader()
    graph = loader.load(args.architecture)
    validator = GraphValidator()
    result = validator.validate(graph, input_shape=args.input_shape)
    if not result.valid:
        print('❌ Validation failed. Cannot compile.')
        for error in result.errors:
            print(f'  - {error.message}')
        sys.exit(1)

    engine = CompilationEngine()
    code = engine.compile(graph)
    args.output.write_text(code)
    print(f'✅ Compiled to {args.output}')


def _run_run(args):
    runtime = ContainerRuntime()
    result = runtime.execute(args.model, cpu_limit=args.cpu_limit, memory_limit=args.memory_limit)
    print(result.stdout)
    if result.stderr:
        print('STDERR:', result.stderr, file=sys.stderr)
    sys.exit(0 if result.success else 1)


if __name__ == '__main__':
    main()
