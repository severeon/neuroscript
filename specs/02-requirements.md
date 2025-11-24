# Requirements Document

## Introduction

This document specifies functional requirements for NeuroScript v2, a type-safe neural architecture composition system. Requirements are organized by functional area and include acceptance criteria tied to specific system components.

## Glossary

- **Block**: A reusable neural network component with capability specification
- **Capability**: A block's declaration of acceptable input/output shapes and constraints
- **Architecture**: A YAML file defining how blocks are connected
- **Graph**: Internal representation of an architecture's topology
- **Unification**: Process of matching shape patterns to find compatible configurations
- **Constraint**: A rule limiting valid shape values (e.g., `dim % 8 == 0`)

---

## Requirements

### Requirement 1: Block Capability Specification

Blocks must declare their input/output requirements and parameter constraints in a standardized YAML format that can be parsed and validated.

#### Acceptance Criteria

1.1. WHEN a block's `block.yaml` file is loaded, THE **CapabilityParser** SHALL parse the `capabilities` section into a structured object containing `inputs`, `outputs`, `params`, and `constraints` dictionaries.

1.2. WHEN an input capability declares a shape pattern (e.g., `[batch, seq, dim]`), THE **CapabilityParser** SHALL extract named dimensions as variables for constraint solving.

1.3. WHEN a constraint references another dimension (e.g., `dim == input.x.shape[2]`), THE **CapabilityParser** SHALL create a dependency relationship between dimensions.

1.4. WHEN a parameter declares a range (e.g., `layers: {range: [1, 24]}`), THE **CapabilityParser** SHALL validate that instantiated values fall within the specified range.

1.5. WHEN a block.yaml file is malformed or missing required fields, THE **CapabilityParser** SHALL raise a descriptive error identifying the specific validation failure.

### Requirement 2: Block Discovery and Registration

The system must automatically discover available blocks and load their capabilities for use in architecture validation.

#### Acceptance Criteria

2.1. WHEN the **BlockRegistry** initializes, IT SHALL scan the `blocks/` directory recursively for subdirectories containing `block.yaml` files.

2.2. WHEN a valid block is discovered, THE **BlockRegistry** SHALL register it with a unique identifier matching its directory name.

2.3. WHEN an architecture references a block by name, THE **BlockRegistry** SHALL retrieve the block's parsed capability specification.

2.4. WHEN a block directory is missing `block.yaml`, THE **BlockRegistry** SHALL log a warning and skip that directory.

2.5. WHEN multiple blocks have the same identifier, THE **BlockRegistry** SHALL raise an error indicating the naming conflict.

### Requirement 3: Architecture Graph Loading

Users must be able to define architectures in YAML files that specify components and their connections.

#### Acceptance Criteria

3.1. WHEN an architecture.yaml file is loaded, THE **GraphLoader** SHALL parse the `components` section and instantiate block references with specified parameters.

3.2. WHEN the `topology` section defines connections (e.g., `input -> encoder -> decoder`), THE **GraphLoader** SHALL create an internal graph representation with nodes and directed edges.

3.3. WHEN a component references a non-existent block, THE **GraphLoader** SHALL raise an error listing available blocks.

3.4. WHEN component parameters don't match the block's parameter schema, THE **GraphLoader** SHALL raise an error showing expected vs. provided parameters.

3.5. WHEN the topology contains cycles, THE **GraphLoader** SHALL detect them and raise an error with the cycle path.

### Requirement 4: Shape Compatibility Validation

Connected blocks must have compatible input/output shapes, verified before execution.

#### Acceptance Criteria

4.1. WHEN validating an edge A→B, THE **ShapeValidator** SHALL unify A's output shape pattern with B's input shape pattern.

4.2. WHEN unification succeeds, THE **ShapeValidator** SHALL record dimension bindings (e.g., `dim=512`) for constraint solving.

4.3. WHEN unification fails due to incompatible shapes, THE **ShapeValidator** SHALL return an error describing the mismatch with expected and actual patterns.

4.4. WHEN a shape pattern contains wildcards (`*`), THE **ShapeValidator** SHALL match any value for that dimension.

4.5. WHEN validating multiple edges sharing dimension variables, THE **ShapeValidator** SHALL enforce consistency across all bindings.

### Requirement 5: Hardware Capability Detection

The system must detect available hardware and filter blocks that exceed hardware capabilities.

#### Acceptance Criteria

5.1. WHEN the **HardwareDetector** initializes, IT SHALL detect CUDA availability via `torch.cuda.is_available()`.

5.2. WHEN CUDA is available, THE **HardwareDetector** SHALL retrieve compute capability via `torch.cuda.get_device_capability()`.

5.3. WHEN a block requires CUDA capability higher than available, THE **HardwareDetector** SHALL mark it as incompatible.

5.4. WHEN detecting available memory, THE **HardwareDetector** SHALL query system RAM and GPU VRAM if available.

5.5. WHEN a block's estimated memory exceeds available resources, THE **HardwareDetector** SHALL warn the user before validation proceeds.

### Requirement 6: Constraint Solving and Configuration Enumeration

The system must find all valid dimension configurations satisfying block constraints.

#### Acceptance Criteria

6.1. WHEN all edges are validated, THE **ConstraintSolver** SHALL enumerate possible dimension values satisfying all constraints.

6.2. WHEN multiple valid configurations exist, THE **ConstraintSolver** SHALL return all solutions with estimated parameter counts and memory usage.

6.3. WHEN no valid configuration exists, THE **ConstraintSolver** SHALL report which constraints are unsatisfiable.

6.4. WHEN the user selects a configuration, THE **ConstraintSolver** SHALL bind all dimension variables to concrete values.

6.5. WHEN constraints conflict (e.g., `dim == 256` and `dim == 512`), THE **ConstraintSolver** SHALL report the conflicting constraints with their sources.

### Requirement 7: Comprehensive Error Reporting

Validation failures must produce actionable error messages guiding users to fixes.

#### Acceptance Criteria

7.1. WHEN shape validation fails, THE **GraphValidator** SHALL report the source and target blocks, expected vs. actual shapes, and suggested fixes.

7.2. WHEN a constraint is violated, THE **GraphValidator** SHALL identify the constraint, the violating value, and valid alternatives.

7.3. WHEN hardware is insufficient, THE **GraphValidator** SHALL suggest reducing batch size, using smaller blocks, or switching to CPU-compatible blocks.

7.4. WHEN a block is missing, THE **GraphValidator** SHALL list available blocks matching similar names.

7.5. WHEN reporting errors, THE **GraphValidator** SHALL use clear, non-technical language accessible to ML practitioners.

### Requirement 8: PyTorch Code Generation

Validated architectures must be compiled to executable PyTorch code.

#### Acceptance Criteria

8.1. WHEN compilation is requested, THE **CompilationEngine** SHALL generate a Python module containing a `nn.Module` subclass.

8.2. WHEN instantiating blocks, THE **CompilationEngine** SHALL import block implementations from the `blocks/` directory.

8.3. WHEN the architecture includes sequential connections, THE **CompilationEngine** SHALL use `nn.Sequential` for efficiency.

8.4. WHEN the architecture includes parallel paths, THE **CompilationEngine** SHALL generate code to execute branches and merge results.

8.5. WHEN generating code, THE **CompilationEngine** SHALL insert shape assertions matching the validated configuration for runtime verification.

### Requirement 9: Container-Based Execution

Generated models must execute in isolated Docker containers with resource limits.

#### Acceptance Criteria

9.1. WHEN execution is requested, THE **ContainerRuntime** SHALL create a Docker container with mounted code directory.

9.2. WHEN creating the container, THE **ContainerRuntime** SHALL apply CPU and memory limits based on hardware detection.

9.3. WHEN GPU is available and requested, THE **ContainerRuntime** SHALL pass `--gpus` flag to enable GPU access.

9.4. WHEN the container exceeds memory limits, THE **ContainerRuntime** SHALL terminate it and report the resource violation.

9.5. WHEN execution completes, THE **ContainerRuntime** SHALL capture stdout/stderr and return them to the user.

### Requirement 10: Block Interface Contract

All blocks must implement a standardized interface for discovery and execution.

#### Acceptance Criteria

10.1. WHEN a block is loaded, THE **BlockInterface** protocol SHALL verify it implements `get_capabilities()` returning capability metadata.

10.2. WHEN a block is instantiated, THE **BlockInterface** SHALL verify it accepts `**params` matching its capability spec.

10.3. WHEN a block's forward pass is called, THE **BlockInterface** SHALL verify it accepts inputs matching declared input shapes.

10.4. WHEN a block is incompatible with the interface, THE **BlockRegistry** SHALL reject it with a clear error message.

10.5. WHEN the interface protocol changes, THE **BlockRegistry** SHALL version-check blocks and warn about outdated implementations.

---

## Requirements Summary

- **Total Requirements**: 10
- **Total Acceptance Criteria**: 50
- **Components Involved**: BlockInterface, BlockRegistry, CapabilityParser, ShapeValidator, ConstraintSolver, HardwareDetector, GraphLoader, GraphValidator, CompilationEngine, ContainerRuntime
