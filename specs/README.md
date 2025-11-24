# NeuroScript v2 Architecture Specifications

Complete architectural documentation for NeuroScript v2 - a type-safe neural architecture composition system with hardware-aware validation and container-based execution.

## Document Overview

### [00-research.md](00-research.md) - Verifiable Technology Proposal
Evidence-based technology selection with citations from verified sources:
- **Plugin Architecture**: Protocol-based design for loose coupling
- **Shape Validation**: jaxtyping for runtime tensor shape checking
- **Hardware Detection**: PyTorch APIs for CUDA capability detection
- **Container Execution**: Docker with resource constraints
- **Schema Validation**: Yamale for YAML constraint validation

### [01-blueprint.md](01-blueprint.md) - Architectural Blueprint
High-level system architecture defining:
- **Core Objective**: Type-safe block composition with pre-execution validation
- **10 Core Components**: BlockInterface, BlockRegistry, CapabilityParser, ShapeValidator, ConstraintSolver, HardwareDetector, GraphLoader, GraphValidator, CompilationEngine, ContainerRuntime
- **Data Flow**: User YAML → Validation → Configuration → Compilation → Execution
- **Integration Points**: Component communication protocols and data formats

### [02-requirements.md](02-requirements.md) - Functional Requirements
Detailed requirements with testable acceptance criteria:
- **10 Requirements** covering all system functionality
- **50 Acceptance Criteria** tied to specific components
- Each criterion specifies WHEN/THE/SHALL behavior for validation

### [03-design.md](03-design.md) - Detailed Component Design
Implementation specifications for each component:
- **Interface Definitions**: Method signatures with type annotations
- **Implementation Details**: Algorithms and data structures
- **Dependencies**: Inter-component relationships
- **Example Block**: Complete Linear block implementation

### [04-tasks.md](04-tasks.md) - Implementation Plan
Actionable task breakdown:
- **21 Major Tasks** organized into 5 phases
- **179 Subtasks** with requirement traceability
- **5-Week Timeline** with parallel development paths
- **Dependency Graph**: Clear sequencing and parallelization

### [05-validation.md](05-validation.md) - Traceability Matrix
Complete validation of specification:
- **100% Coverage**: All 50 acceptance criteria traced to tasks
- **Component Coverage**: All 10 components fully specified
- **Example Blocks**: 5 test blocks (Linear, Embedding, LayerNorm, Dropout, Sequential)
- **Resource Requirements**: Runs on modest hardware (2 CPU cores, 2GB RAM)

## Key Design Decisions

### 1. Hardware-Aware Validation
System detects CUDA capability, memory, and CPU cores before validation. Blocks requiring CUDA >3.7 are filtered on MacBooks, preventing "CUDA not available" errors during execution.

### 2. Container-Based Execution
All model execution happens in Docker containers with:
- CPU limits (e.g., `--cpus="2.0"`)
- Memory limits (e.g., `--memory="2g"`)
- GPU passthrough when available (`--gpus=all`)
- Air-gapped networking (no external access)

### 3. Protocol-Based Block Interface
Blocks implement `BlockInterface` protocol for loose coupling:
```python
class BlockInterface(Protocol):
    @staticmethod
    def get_capabilities() -> BlockCapability: ...
    def __init__(self, **params): ...
    def forward(self, **inputs) -> Dict[str, Any]: ...
```

### 4. Capability-Driven Validation
Blocks declare capabilities in `block.yaml`:
```yaml
capabilities:
  inputs:
    x: {shape: ["batch", "seq", "dim"], dtype: [float32, float16]}
  outputs:
    y: {shape: ["batch", "seq", "dim"], dtype: input.x.dtype}
  constraints:
    - dim % 8 == 0
```

### 5. Actionable Error Messages
Validation failures include:
- What failed (source/target blocks, expected/actual shapes)
- Why it failed (constraint violation, type mismatch)
- How to fix it (add adapter, change config, use different block)

## Development Workflow

### Phase 1: Core Infrastructure (Week 1)
Implement foundational components (BlockInterface, CapabilityParser, BlockRegistry, GraphLoader)

### Phase 2: Validation Pipeline (Week 2)
Build validation engine (ShapeValidator, HardwareDetector, ConstraintSolver, GraphValidator)

### Phase 3: Compilation & Execution (Week 3)
Create code generation and runtime (CompilationEngine, ContainerRuntime)

### Phase 4: Example Blocks (Week 4)
Develop 5 test blocks for validation (Linear, Embedding, LayerNorm, Dropout, Sequential)

### Phase 5: Integration & Testing (Week 5)
End-to-end testing, CLI, documentation, CI/CD

## Target Feature Set

### 15 Initial Blocks (v2.0)
1. Embedding
2. Linear
3. LayerNorm
4. Dropout
5. GELU
6. Attention (MultiHead)
7. FFN
8. Residual
9. TransformerBlock
10. MambaBlock
11. LoRA
12. MixtureOfExperts
13. FlashAttention
14. ResNetBlock
15. LSTM

### 5 Example Blocks (Testing)
1. **Linear** - Basic dimension transformation
2. **Embedding** - Token to vector mapping
3. **LayerNorm** - Shape-preserving normalization
4. **Dropout** - Regularization with probability parameter
5. **Sequential** - Block composition

## Success Criteria

### Must Work End-to-End
```bash
# 1. Write architecture
$ cat > my-model.yaml << EOF
components:
  embedding: Embedding(vocab=50000, dim=512)
  encoder: TransformerBlock(layers=6, dim=512)
  head: Linear(in=512, out=50000)
topology:
  tokens -> embedding -> encoder -> head -> logits
EOF

# 2. Validate
$ neuroscript validate my-model.yaml --input-shape [32, 128]
✓ Valid - 48M params, 2.1GB memory

# 3. Compile
$ neuroscript compile my-model.yaml --input-shape [32, 128] -o model.py

# 4. Run in container
$ neuroscript run model.py --dataset dummy
Execution complete: 32 batches processed
```

## Quick Reference

### File Structure
```
neuroscript/
├── specs/              # This directory
│   ├── 00-research.md
│   ├── 01-blueprint.md
│   ├── 02-requirements.md
│   ├── 03-design.md
│   ├── 04-tasks.md
│   └── 05-validation.md
├── src/
│   ├── core/          # Validation engine
│   └── cli/           # Command-line interface
├── blocks/            # Block implementations
│   ├── linear/
│   ├── embedding/
│   └── ...
├── examples/          # Example architectures
└── test/              # Unit and integration tests
```

### Command Reference
```bash
# Validate architecture
neuroscript validate architecture.yaml [--input-shape SHAPE]

# Compile to PyTorch
neuroscript compile architecture.yaml --input-shape SHAPE --output model.py

# Execute in container
neuroscript run model.py [--gpu] [--cpu-limit N] [--memory-limit NG]

# List available blocks
neuroscript blocks list

# Show block capabilities
neuroscript blocks info BLOCK_NAME
```

## Implementation Status

- [x] Phase 0: Research and technology selection
- [x] Phase 1: Architecture blueprint
- [x] Phase 2: Requirements specification
- [x] Phase 3: Detailed design
- [x] Phase 4: Task decomposition
- [x] Phase 5: Validation and traceability
- [ ] Implementation (21 tasks, 179 subtasks)

**Status**: Specifications complete, ready for implementation.
