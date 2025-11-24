# NeuroScript — Parser Prototype & Language Spec (v0.1)

## 1) Quick sanity review of your example files

You provided two `.mmd` blocks: `apoptotic_model.mmd` and `tiny_recursive_mamba.mmd`. Overall they are coherent and well-formed. A few notes to tighten:

* **Frontmatter**: You use YAML frontmatter separated by `---` which is good and consistent. Make sure every file has exactly one `__neuroscript` top-level key (or none). Parser will accept optional frontmatter but require `__neuroscript` when present.

* **Node attribute syntax**: You used syntax like `TRM1@{shape: rect, node: TRMBlock(depth=CONFIG.trm.depth)}`. That is readable but ambiguous in two places:

  * `node: TRMBlock(depth=...)` mixes a node type identifier and instantiation args. In the spec below we formalize `node: TypeName(param=val,...)` as allowed, but the parser normalizes this into `{ type: "TRMBlock", params: {...} }`.
  * Use commas between attributes consistently. Trailing commas are tolerated.

* **Edge annotations**: You used `--> |$shape fits 256x128x128|` and `--> |$cycle <lt> 3|`. We'll treat edge labels as expressions/guards and parse them as strings. The parser does not (yet) evaluate them; ARIES/NACE will later evaluate via a small expression evaluator.

* **Shape assertions**: You introduced `fits`, `snug`, `eq` semantics as comments. The shape algebra will be part of the type-checker later. For now we parse `|$shape fits 64x64x64|` as an edge `guard` with `op: fits` and `rhs: [64,64,64]` if it matches a simple pattern.

* **Subgraph (`subgraph`)**: Your use of `subgraph` is valid mermaid. The parser will capture the subgraph as a nested graph block; subgraph labels become scopes.

* **Variable interpolation**: You used `${CONFIG.data.dim_x}` and `$PROJECT_ROOT`. The parser will collect these `env` tokens and return a list of referenced variables for resolution. Substitution is deferred to a config resolution stage.

---

## 2) NeuroScript language spec (summary)

This is the compact spec to include in README / language docs.

### 2.1 File Structure

A NeuroScript file is a Mermaid-flavored flowchart plus an optional treelike YAML frontmatter under `__neuroscript`. Files are `.mmd`.

```
---
# YAML frontmatter (optional)
__neuroscript:
  conf:
    - path/to/config.yml
  io:
    type: Tensor
    dtype: float32
    shape: [128, 128]
  components: {...}
---
flowchart TD
  ... mermaid content ...
```

### 2.2 Node definition

A node may be defined inline in the flowchart with extended attributes:

```
NodeName@{shape: rect, node: TypeName(param=123, flag=true), label: "Foo"}
```

Normalized semantic fields:

* `id` — NodeName (string)
* `shape` — rendering hint (brace, pill, rect...) (optional)
* `type` — TypeName (string) — maps to a module in `src/blocks/TypeName.{rs,py}`
* `params` — a dict of instantiation parameters parsed from `TypeName(...)` or `params:` block
* `label` — rendered label (optional)

If `node:` is just `TypeName` (no parens) it means default params.

Nodes can also be referenced as `BlockReference($PROJECT_ROOT/src/blocks/...)` — parser stores `ref` paths.

### 2.3 Edges

Mermaid edges are parsed. Extended edge syntax allowed:

```
A -->|$shape fits 64x64x64| B
```

Edge attributes captured:

* `source` `target`
* `type` (`->`, `-->`, `-.->`, etc)
* `guard` (string/parsed expression)

Guard expressions are not executed by the parser; they are exported to the IR for later evaluation.

### 2.4 Subgraphs, loops, and entry/exit

Subgraphs are supported and result in nested IR nodes. `ENTRY` and `EXIT` nodes are treated as ports.

### 2.5 Comments and annotations

Mermaid comments and HTML-style comments are ignored. Lines starting `%%` are treated as annotations and preserved in node `meta` for later display.

### 2.6 Imports & Composition

In `__neuroscript.conf` the `conf` array lists YAML config sources. The parser returns the ordered list so a config loader can merge in that order.

Files may reference and include other `.mmd` blocks using `BlockReference(path)` and via `subgraph` composition.

### 2.7 Shape algebra & routing semantics (overview)

<!-- I dunno if these are already named functions, if they already exist lets use them instead -->

* `fits`: each dimension of RHS <= corresponding dimension of LHS
* `snug`: same as fits + divisibility
* `eq`: exact match

These rules are enforced at the NACE type-check stage (not by this prototype parser). Parser extracts guards as structured tokens.

---

## 3) IR design (JSON/YAML serializable)

A single NeuroScript file compiles to an `IR` with structure:

```yaml
id: string  # filename-based
meta:
  title: string
  frontmatter: {...}
nodes:
  - id: "TRM1"
    type: "TRMBlock"
    params: { depth: 3 }
    shape_hint: "rect"
    ref: null
    label: "TRM1"
    meta: {...}
edges:
  - source: "TRM1"
    target: "TRM2"
    kind: "-->"
    guard: { raw: "$shape fits 64x64x64", op: "fits", rhs: [64,64,64] }
subgraphs:
  - id: "Process"
    nodes: [ ... ]
variables:
  - "CONFIG.data.dim_x"
  - "$PROJECT_ROOT"
```

This IR is intentionally minimal and designed to be easily consumed by NACE or a codegen backend.
