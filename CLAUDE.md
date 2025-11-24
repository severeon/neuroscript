# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroScript v2 is a type-safe neural architecture composition system that validates block compatibility through capability-based shape inference before execution. It automatically detects hardware constraints (CUDA/CPU/memory) and executes validated architectures in resource-controlled containers.

**Core Philosophy**: "Validation is the product. The rest is tooling."

**Current State**: Phases 1-3 complete (all 10 core components implemented, 298 passing tests including monitoring). Ready for Phase 4 (example blocks) and Phase 5 (integration testing).

## Development Commands

### Testing

```bash
# Run all tests (298 tests across all components)
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest test/core/test_graph_validator.py

# Run specific test function
pytest test/core/test_shape_validator.py::test_shape_unification_with_wildcards

# Run with verbose output
pytest -v
```

### Installation

```bash
# Install basic package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with CUDA support (optional)
pip install -e ".[cuda]"

# Install with all dependencies
pip install -e ".[full]"
```

### CLI Usage

```bash
# Validate architecture
neuroscript validate architecture.yaml --input-shape 32 128 512

# Compile to PyTorch
neuroscript compile architecture.yaml --input-shape 32 128 512 -o model.py

# Run inference with monitoring (default)
neuroscript run model.py --cpu-limit 2.0 --memory-limit 2g

# Run training mode with monitoring
neuroscript run model.py --mode training --epochs 10 --batch-size 32

# Run with log saving
neuroscript run model.py --enable-save-logs

# Run quietly (minimal output)
neuroscript run model.py --quiet
```

## Architecture Overview

NeuroScript v2 has 10 core components organized into 3 phases:

**Phase 1: Core Infrastructure** (Complete)
- `BlockInterface` - Protocol defining block contract
- `CapabilityParser` - Parses `block.yaml` capability specifications
- `BlockRegistry` - Discovers and loads blocks from filesystem
- `GraphLoader` - Loads `architecture.yaml` files

**Phase 2: Validation Pipeline** (Complete)
- `ShapeValidator` - Validates tensor shape compatibility via unification
- `HardwareDetector` - Detects CUDA/CPU/memory capabilities
- `ConstraintSolver` - Enumerates valid shape configurations
- `GraphValidator` - Orchestrates complete validation with actionable errors

**Phase 3: Compilation & Execution** (Complete)
- `CompilationEngine` - Generates PyTorch code from validated graphs
- `ContainerRuntime` - Executes models in Docker with resource limits
- `Monitoring` - Real-time execution tracking with resource monitoring

All components are in `src/core/` with corresponding tests in `test/core/`.

## Execution Monitoring System

NeuroScript includes comprehensive execution monitoring for both inference and training:

**Real-time Metrics**:
- CPU usage (% per core)
- Memory usage (MB and % of limit)
- GPU utilization and memory (if available)
- Per-layer/block timing (milliseconds)
- Training progress (epochs, batches, loss, accuracy)

**Monitoring Output**:
```
[12:34:56] Starting inference | Input shape: (1, 512)
[12:34:56] Block 'encoder': (1, 512) → (1, 256) [15.3ms]
[12:34:57] Block 'decoder': (1, 256) → (1, 128) [12.8ms]

Execution Summary:
─────────────────
Total Time: 2.1s
Peak Memory: 245.3 MB
Peak CPU: 67.8%

Layer Timings (avg):
  encoder: 15.3ms
  decoder: 12.8ms
```

**Training Mode Output**:
```
[12:34:56] Starting training | Input shape: (32, 512)
[12:34:57] Epoch 1/5 started
[12:35:02] Epoch 1 | Batch 10/31 | Loss: 0.4523
[12:35:10] Epoch 1/5 complete | Train Loss: 0.3241 | Accuracy: 0.887
[12:35:15] Epoch 2/5 started
...

Execution Summary:
─────────────────
Total Time: 45.3s
Peak Memory: 1024.5 MB
Peak CPU: 87.3%

Final Metrics:
  Train Loss: 0.1234
  Train Accuracy: 0.9456
```

**Log Files**:
Monitoring data can be saved to JSON files for later analysis:
```bash
neuroscript run model.py --enable-save-logs
# Creates: neuroscript_run_20250124_123456.log
```

Log file structure:
```json
{
  "command": {"model": "model.py", "mode": "training", ...},
  "start_time": "2025-01-24T12:34:56",
  "end_time": "2025-01-24T12:35:41",
  "execution_time_seconds": 45.3,
  "monitoring_events": [...],
  "resource_samples": [...],
  "metrics": {
    "peak_memory_mb": 1024.5,
    "peak_cpu_percent": 87.3,
    "final_train_loss": 0.1234,
    "layer_timings": {"encoder": 15.3, "decoder": 12.8}
  }
}
```

**Generated Model Features**:
All compiled models include:
- Built-in monitoring instrumentation
- Command-line arguments for mode selection (`--mode inference|training`)
- Training loop with synthetic data for testing
- Per-block timing with automatic event emission
- Structured JSON event protocol for tooling integration

## Specifications Drive Implementation

The `specs/` directory contains complete architectural documentation that drives all implementation:

- `00-research.md` - Technology selection with evidence (PyTorch, Docker, YAML)
- `01-blueprint.md` - 10-component architecture diagram
- `02-requirements.md` - 10 requirements × 50 acceptance criteria
- `03-design.md` - Detailed component specifications with method signatures
- `04-tasks.md` - 21 tasks, 179 subtasks with traceability to requirements
- `05-validation.md` - 100% traceability matrix proving all criteria covered

**Key principle**: The specs are NOT aspirational—they're complete architectural blueprints that the implementation strictly follows. Each component has a corresponding spec section.

## Block Structure

Each block lives in `blocks/<name>/`:
```
blocks/linear/
├── block.yaml      # Capability specification
├── module.py       # PyTorch implementation
└── test.py         # Unit tests
```

**block.yaml format**:
```yaml
name: Linear
version: 1.0.0
capabilities:
  inputs:
    x: {shape: ["*", "in_features"], dtype: [float32, float16]}
  outputs:
    y: {shape: ["*", "out_features"], dtype: input.x.dtype}
  parameters:
    in_features: {type: int, range: [1, 100000], required: true}
    out_features: {type: int, range: [1, 100000], required: true}
    bias: {type: bool, default: true}
```

## Data Flow

```
User Architecture (architecture.yaml)
    ↓
GraphLoader → Parse → ArchitectureGraph
    ↓
BlockRegistry → Lookup → BlockCapability
    ↓
GraphValidator → Orchestrate → ValidationResult
    ├→ ShapeValidator → Shape unification
    ├→ HardwareDetector → Resource compatibility
    └→ ConstraintSolver → Valid configurations
    ↓
CompilationEngine → Generate → PyTorch code
    ↓
ContainerRuntime → Execute → Results
```

## Key Design Decisions

### 1. Validation vs Execution
- **Validation** is deterministic, instant—happens before any GPU use
- **Execution** is optional—users can validate and never compile/run
- Shape validation catches all errors at compile-time

### 2. Capability Declaration vs Implementation
- Blocks declare what they CAN do in `block.yaml` (capability spec)
- Implementation (PyTorch module) not needed for validation to work
- Separation enables validation without executing code

### 3. Hardware-Aware Filtering
- System detects CUDA capability, memory, and CPU cores before validation
- Blocks requiring CUDA >3.7 are filtered on CPU-only machines
- Prevents "CUDA not available" errors during execution

### 4. Actionable Error Messages
Validation failures include:
- What failed (source/target blocks, expected/actual shapes)
- Why it failed (constraint violation, type mismatch)
- How to fix it (add adapter, change config, use different block)

Example:
```
Error: Shape mismatch at encoder -> decoder
  encoder output: [batch, 128, 512]
  decoder input: [batch, seq, 256] where seq >= 1

Suggested fixes:
  1. Add adapter: Linear(512 -> 256)
  2. Change encoder: dim=256
  3. Change decoder: dim=512
```

## Implementation Status

| Phase | Status | Components |
|-------|--------|------------|
| Phase 1: Core Infrastructure | ✅ Complete | BlockInterface, CapabilityParser, BlockRegistry, GraphLoader |
| Phase 2: Validation Pipeline | ✅ Complete | ShapeValidator, HardwareDetector, ConstraintSolver, GraphValidator |
| Phase 3: Compilation & Execution | ✅ Complete | CompilationEngine, ContainerRuntime, Monitoring |
| Phase 4: Example Blocks | ⏳ Planned | Linear, Embedding, LayerNorm, Dropout, Sequential |
| Phase 5: Integration & Testing | ⏳ Planned | End-to-end tests, performance optimization, docs |

**Test Coverage**: 298 passing tests covering all implemented components (20 new monitoring tests)

## File Structure Highlights

```
neuroscript/
├── specs/              # Complete architectural specifications (ready)
├── src/
│   ├── core/          # 10 core components (all implemented)
│   └── cli/           # CLI with validate/compile/run commands
├── test/              # 278 comprehensive tests
│   ├── cli/
│   └── core/
├── blocks/            # Block implementations (none yet—Phase 4)
├── pyproject.toml     # Modern Python packaging config
├── pytest.ini         # Test configuration
└── README.md          # User-facing documentation

```

## Dependencies

**Core** (required):
- `pyyaml>=6.0` - YAML parsing
- `jinja2>=3.0` - Code generation templates

**Optional** (for hardware detection):
- `torch>=2.0` - CUDA detection, tensor operations
- `psutil>=5.9` - CPU/memory system queries

**Development**:
- pytest, black, mypy, ruff

## Code Quality Standards

- **Type hints**: Full type annotations throughout `src/core/`
- **Error handling**: Specific exception classes per module, no broad catches
- **Testing**: 278 tests with unit and integration coverage
- **Documentation**: Docstrings for all public functions and classes
- **Security**: Restricted eval() environment, input validation

## Working with Specifications

When implementing new features:

1. **Check specs first** → Look in `specs/04-tasks.md` for task definition
2. **Understand requirements** → Read `specs/02-requirements.md` for acceptance criteria
3. **Follow the interface** → Reference `specs/03-design.md` for method signatures
4. **Implement in src/core/** → Create or modify component
5. **Write tests** → Create/update `test/core/test_*.py`
6. **Verify traceability** → Check `specs/05-validation.md` for coverage

## Entry Points

**Programmatic**:
```python
# Registry discovery
from core.block_registry import BlockRegistry
registry = BlockRegistry()

# Graph loading and validation
from core.graph_loader import GraphLoader
from core.graph_validator import GraphValidator
from core.hardware_detector import HardwareDetector

loader = GraphLoader(registry)
graph = loader.load('architecture.yaml')
hardware = HardwareDetector()
validator = GraphValidator(graph, registry, hardware)
result = validator.validate()
```

**CLI**:
```python
# Entry point defined in pyproject.toml
# [project.scripts]
# neuroscript = "src.cli.main:main"
```

## Important Notes

- Modules in `src/core/` are loosely coupled via dataclass interfaces
- Each module has corresponding test file with 20-40 tests
- Shape patterns support wildcards (`*`) and named dimensions
- Container runtime requires Docker installation
- All tests pass on Python 3.8+
