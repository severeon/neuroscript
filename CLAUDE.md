# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroScript v2 is a type-safe neural architecture composition system that validates block compatibility through capability-based shape inference before execution. It automatically detects hardware constraints (CUDA/CPU/memory) and executes validated architectures in resource-controlled containers.

**Current State**: Complete architectural specifications ready for implementation. See `specs/` directory for detailed design documents.

**Legacy**: The v0 parser prototype (`src/tools/parser.py`) transforms Mermaid-based `.mmd` files into IR. This is being superseded by the v2 validation-first approach.

## NeuroScript v2 Architecture (Current Focus)

### Quick Reference

**Specifications**: See `specs/README.md` for complete architectural documentation including:
- `00-research.md` - Evidence-based technology selection
- `01-blueprint.md` - System architecture with 10 core components
- `02-requirements.md` - 10 requirements with 50 acceptance criteria
- `03-design.md` - Detailed component specifications
- `04-tasks.md` - 21 implementation tasks (179 subtasks)
- `05-validation.md` - 100% traceability matrix

### Core Components

1. **BlockInterface** - Protocol defining block contract
2. **CapabilityParser** - Parses `block.yaml` capability specifications
3. **BlockRegistry** - Discovers and loads blocks from filesystem
4. **GraphLoader** - Loads `architecture.yaml` files
5. **ShapeValidator** - Validates tensor shape compatibility via unification
6. **HardwareDetector** - Detects CUDA/CPU/memory capabilities
7. **ConstraintSolver** - Enumerates valid shape configurations
8. **GraphValidator** - Orchestrates complete validation with actionable errors
9. **CompilationEngine** - Generates PyTorch code from validated graphs
10. **ContainerRuntime** - Executes models in Docker with resource limits

### Development Workflow

```bash
# Validate architecture
neuroscript validate architecture.yaml --input-shape [32, 128, 512]

# Compile to PyTorch
neuroscript compile architecture.yaml --input-shape [32, 128, 512] -o model.py

# Execute in container
neuroscript run model.py --cpu-limit 2.0 --memory-limit 2g
```

### Block Structure

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
  params:
    in_features: {type: int, range: [1, 100000], required: true}
    out_features: {type: int, range: [1, 100000], required: true}
    bias: {type: bool, default: true}
```

### Implementation Status

- [x] Research and technology selection
- [x] Complete architectural specifications (50 requirements, 179 tasks)
- [ ] Core infrastructure implementation (Phase 1: Tasks 1-4)
- [ ] Validation pipeline (Phase 2: Tasks 5-8)
- [ ] Compilation and execution (Phase 3: Tasks 9-10)
- [ ] Example blocks (Phase 4: Tasks 11-15)
- [ ] Integration testing (Phase 5: Tasks 16-21)

---

## NeuroScript v0 Parser (Legacy Prototype)

## Development Commands

### Running the Parser

Parse a NeuroScript file and output IR to JSON:

```sh
python src/tools/parser.py src/blocks/apoptotic_model.mmd > build/blocks/apoptotic_model.json

python src/tools/parser.py src/blocks/tiny_recursive_mamba.mmd > build/blocks/tiny_recursive_mamba.json
```

### Testing

Run all tests with pytest:

```sh
pytest
```

Run specific test file:

```sh
pytest test/tools/test_parser.py
```

Run specific test function:

```sh
pytest test/tools/test_parser.py::test_basic
```

## Architecture

### Core Parser (`src/tools/parser.py`)

The parser is a single-file, regex-based implementation with three main phases:

1. **Frontmatter Parsing** (`parse_frontmatter`)
   - Extracts YAML frontmatter between `---` delimiters
   - Looks for `__neuroscript` configuration block containing:
     - `conf`: list of config file paths
     - `io`: input/output type/shape definitions
     - `components`: default component configurations

2. **Node Attribute Parsing** (`parse_node_attrs`, `parse_mermaid_flow`)
   - Extracts node definitions using `@{...}` syntax
   - Pattern: `NodeName@{shape: rect, node: TypeName(param=val), label: "..."}`
   - Normalizes to IR format with fields: `id`, `shape_hint`, `type`, `params`, `label`, `meta`
   - Handles parameter parsing including:
     - Key-value pairs: `depth=3`
     - Boolean flags: `boolTrue` (parsed as `boolTrue=True`)
     - YAML-safe literals: numbers, strings, booleans

3. **Edge Parsing**
   - Captures Mermaid edges with optional guards/labels
   - Pattern: `A -->|$shape fits 64x64x64| B`
   - Stores: `source`, `target`, `kind` (arrow type), `label`/`guard`

4. **Variable Extraction**
   - Finds all variable references: `${CONFIG.data.dim_x}` or `$PROJECT_ROOT`
   - Returns sorted list for config resolution by downstream tools
   - Parser does NOT evaluate variables—defers to config loader

### IR (Intermediate Representation)

Output structure:

```yaml
id: string          # file path
meta:
  frontmatter: {}   # __neuroscript block
nodes:
  - id: string
    type: string
    params: {}
    shape_hint: string
    label: string
    meta: {}
edges:
  - source: string
    target: string
    kind: string
    label: string
variables: []       # extracted variable references
```

### Block Files (`src/blocks/`)

Example NeuroScript files demonstrating the syntax:
- `apoptotic_model.mmd`: Complex model with subgraphs, guards, and block references
- `tiny_recursive_mamba.mmd`: Recursive model with cycle guards and entry/exit nodes

### NeuroScript Syntax

Key syntax elements:

**Node attributes:**
```
NodeName@{shape: rect, node: TypeName(param=val), label: "Display Name"}
```

**Edge guards:**
```
A -->|$shape fits 64x64x64| B
A -->|$cycle < 3| B
```

**Block references:**
```
node: BlockReference($PROJECT_ROOT/src/blocks/tiny_recursive_mamba.mmd)
```

**Frontmatter:**
```yaml
---
__neuroscript:
  conf:
    - configs/base.yml
  io:
    type: Tensor
    shape: [128, 128]
---
```

### Guard Expression Semantics

Shape guards (not enforced by parser, documented for type-checker):
- `fits`: each RHS dimension ≤ corresponding LHS dimension
- `snug`: same as fits + divisibility constraint
- `eq`: exact shape match

Cycle guards:
- `$cycle < 3`: loop iteration count comparison
- `$loss > $max_loss`: runtime metric comparison

## Key Implementation Details

### Regex Patterns

- `FRONT_RE`: Matches YAML frontmatter block
- `NODE_ATTR_RE`: Matches node definitions with attributes
- `EDGE_RE`: Matches Mermaid edges with optional labels
- `KV_RE`: Parses key-value pairs in node attributes
- `NODE_TYPE_RE`: Extracts type name and parameters from `TypeName(params)`

### Parameter Parsing

Boolean flags without `=` are parsed as `True`:
```python
# Input: params: boolTrue, boolFalse=false
# Output: {'boolTrue': True, 'boolFalse': False}
```

All parameter values are parsed via `yaml.safe_load()` for type safety.

### Variable References

Variables are collected but NOT resolved. Pattern: `$\{?([A-Za-z0-9_\.\-/]+)\}?`

Examples:
- `${CONFIG.data.dim_x}` → `CONFIG.data.dim_x`
- `$PROJECT_ROOT` → `PROJECT_ROOT`

Resolution happens in downstream tools (NACE/ARIES) via config file loading.

## Project Structure

```
neuroscript/
├── src/
│   ├── blocks/          # Example .mmd files
│   └── tools/
│       └── parser.py    # Core parser implementation
├── test/
│   └── tools/
│       └── test_parser.py
├── build/               # Output directory for generated IR JSON files
├── neuroscript.spec.md  # Language specification
└── README.md            # Project documentation
```

## Ecosystem Context

NeuroScript is a foundation for:
- **NACE**: Live visual/interactive IDE (Rust)
- **ForkPoint**: Versioned architectural lineage tracker
- **ARIES**: Automated experiment generation
- **Construct**: High-level model programming language

The Python parser is a prototype. Future versions will include:
- Complete Mermaid grammar support
- Incremental parsing for NACE
- Guard expression evaluator
- Shape/type checker
- Rust backend integration
