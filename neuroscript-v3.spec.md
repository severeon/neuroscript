# NeuroScript v3 Specification: Production-Ready Neural Architecture Composition

## Core Insight

**Validation is the product. The rest is tooling.**

NeuroScript v3 delivers the full vision from v2 proposal: capability-based block composition with static shape inference, constraint solving, and actionable errors. Built on Phases 1-3 (core infra/validation/compilation complete), v3 adds production ecosystem: git-based block registry, advanced YAML syntax, 10+ reference blocks, shape inference engine, fork workflow.

Every block declares capabilities. Validator ensures compatibility chains. Inference enumerates configs. Users select dimensions. Invalid graphs rejected pre-compute.

Current status: Validator/compiler/runner operational with linear block example. 298 tests pass.

---

## Design Principles

1. **Capability-based composition** - blocks declare requirements/capabilities via YAML
2. **Static validation** - catch errors at 'compile' time
3. **Shape inference** - enumerate valid configs, user chooses (Phase 3 complete, v3 enhances)
4. **Fork-and-extend** - git-based blocks enable easy forking (v3)
5. **Explicit over implicit** - no magic, clear errors/suggestions
6. **Hardware-aware** - detect CUDA/memory, filter blocks (implemented)
7. **Containerized execution** - Docker limits prevent OOM (implemented)
8. **Monitoring** - real-time metrics/logs (implemented)

---

## Block Specification (Current v3 Syntax)

Blocks in `blocks/<name>/`:

```
blocks/linear/
├── block.yaml     # Capabilities
├── module.py      # PyTorch impl (nn.Module)
└── test.py        # Tests
```

**block.yaml** (current, v3 aligns to proposal advanced syntax):

```yaml
name: Linear
version: 1.0.0

capabilities:
  inputs:
    x:
      type: Tensor
      dtype: [float32, float16, bfloat16]
      shape: ["*", "in_features"]  # v3: [batch, in_features]

  outputs:
    y:
      type: Tensor
      dtype: input.x.dtype  # Proposal propagation (v3)
      shape: ["*", "out_features"]

  params:
    in_features:
      type: int
      range: [1, 100000]
      required: true
    out_features:
      type: int
      range: [1, 100000]
      required: true
    bias:
      type: bool
      default: true

  constraints:  # v3: advanced like dim % 8 == 0
    - in_features >= 1
    - out_features >= 1

  compute:  # v3 proposal
    params: in_features * out_features
    flops: 2 * batch * in_features * out_features
    memory: batch * out_features * 4  # bytes
```

**Shape Language (v3 targets proposal)**:
- `["batch", "seq", "dim"]` - named
- `["*", "*", 512]` - wildcards
- `input.x.shape` - references
- `dim*2` - transforms

**module.py** implements `BlockInterface`: `get_capabilities()`, `__init__(**params)`, `forward(**inputs)`.

---

## Architecture YAML (Current)

```yaml
# example_architecture.yaml
name: SimpleLinearArchitecture
version: 1.0.0

components:
  layer1:
    block: linear
    params:
      in_features: 512
      out_features: 256
  layer5:
    block: linear
    params:
      in_features: 256
      out_features: 128

topology:
  - layer1 -> layer2 -> layer3 -> layer4 -> layer5  # chains

inputs: [layer1]
outputs: [layer5]
```

v3: multi-path `"input -> [encoder, skip] -> add -> output"`, ports `encoder.y -> decoder.x`.

---

## Graph Validation (Implemented)

**ShapeValidator**: Unifies patterns across edges, binds dims (e.g., encoder.dim=512 → decoder.dim=512).

**ConstraintSolver**: Enumerates configs (basic, v3 full solver).

**GraphValidator**: Orchestrates, generates `ValidationError` with suggestions.

Example error:
```
✗ SHAPE_MISMATCH: Shape mismatch between 'layer1' output and 'layer2' input
  Location: source: layer1, target: layer2
  Output shape: [*, 256]
  Input requires: [*, 512]

Suggested fixes:
  1. Add Linear(256, 512) adapter
  2. Change layer1 out_features: 512
```

**HardwareDetector**: Filters CUDA/memory incompatible blocks.

---

## CLI Workflow (Implemented)

```bash
# 1. Validate
neuroscript validate arch.yaml --input-shape 32 128 512
✓ Valid - configs: dim=512 (params:48M, mem:8GB)

# 2. Compile
neuroscript compile arch.yaml --input-shape 32 128 512 -o model.py
✓ Compiled model.py

# 3. Run w/ monitoring
neuroscript run model.py --mode training --epochs 10 --cpu-limit 2 --memory-limit 2g --enable-save-logs
[12:34] Block 'layer1': 15ms
Epoch 1: loss=2.3

Summary:
Total: 45s, Peak Mem:1.2GB, CPU:67%

Logs: neuroscript_run_*.log
```

v3 adds `infer`, `fork`.

---

## Advanced Features (v3)

### Multi-Path Graphs
```yaml
topology:
  input -> encoder -> decoder -> output
  encoder -> skip -> output  # residual
```
Merge validated by shape match.

### Shape Inference
```bash
neuroscript infer arch.yaml --output-shape [32,128,512]
Suggested input: [32,128,256] + Linear adapter
```

### Git Registry (v3)
```bash
neuroscript blocks add github.com/neuroscript/transformer
neuroscript fork transformer ./my-wide-transformer  # edit dim=384
```

### Fork Workflow
Implemented in proposal, v3 CLI: `fork block_name path`.

---

## Implementation Phases (Status)

### Phase 1-3: Core (✅ Complete, 298 tests)
- BlockInterface, Registry, Parser, GraphLoader
- Validators (Shape, Hardware, Constraint, Graph)
- CompilationEngine, ContainerRuntime w/ monitoring

### Phase 4: Blocks (Partial)
- ✅ Linear
- ⏳ Embedding, LayerNorm, Dropout, Sequential (+5 proposal: Transformer, Mamba, etc.)

### Phase 5: Integration (Planned)
- E2E tests, perf, docs

### Phase 6: v3 Production (Next)
- Git registry/fetcher
- Advanced YAML parser (propagation, formulas)
- Full inference engine
- 10+ reference blocks
- Web docs/examples

**Deliverable**: End-to-end GPT-Nano → model.py → train.

---

## Example: MLP (Current)

Validate example_architecture.yaml → compiles linear chain → trains w/ monitoring.

**v3 Goal**: Transformer-Mamba hybrid validated/inferred/compiled in seconds.

---

## Error Messages (Implemented)

Actionable, as proposal.

## Success Criteria v3

```bash
neuroscript validate gpt-hybrid.yaml --input [32,512]
✓ 124M params, 2GB mem
neuroscript compile gpt-hybrid.yaml -o gpt.py
python gpt.py --train  # monitored
```

**Ship when git blocks + inference work.**

---