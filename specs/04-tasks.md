# Implementation Plan

## Overview

This document breaks down the NeuroScript v2 implementation into concrete, actionable tasks. Tasks are organized by component and can be developed in parallel where dependencies allow.

---

## Development Phases

### Phase 1: Core Infrastructure (Week 1)
Foundation components that other systems depend on

### Phase 2: Validation Pipeline (Week 2)
Shape validation and constraint solving

### Phase 3: Compilation & Execution (Week 3)
Code generation and container runtime

### Phase 4: Example Blocks (Week 4)
5 test blocks for validation

### Phase 5: Integration & Testing (Week 5)
End-to-end testing and documentation

---

## Task Breakdown

### Phase 1: Core Infrastructure

- [x] 1. Implement BlockInterface Protocol
  - [x] 1.1 Create `src/core/block_interface.py`
  - [x] 1.2 Define `BlockInterface` protocol class with `get_capabilities()`, `__init__()`, `forward()` methods
  - [x] 1.3 Define `BlockCapability` dataclass with inputs/outputs/params/constraints fields
  - [x] 1.4 Write unit tests for protocol validation
  - _Requirements: 10.1, 10.2, 10.3_

- [x] 2. Implement CapabilityParser
  - [x] 2.1 Create `src/core/capability_parser.py`
  - [x] 2.2 Implement `parse_file()` method to load and parse block.yaml using yamale
  - [x] 2.3 Implement `extract_dimensions()` to parse shape patterns like "[batch, seq, dim]"
  - [x] 2.4 Implement `parse_constraint()` to convert constraint strings to DimensionConstraint objects
  - [x] 2.5 Implement `validate_param_range()` to check parameter values against range constraints
  - [x] 2.6 Add error handling for malformed YAML with descriptive error messages
  - [x] 2.7 Write unit tests for each parsing method
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Implement BlockRegistry
  - [x] 3.1 Create `src/core/block_registry.py`
  - [x] 3.2 Implement `__init__()` to initialize registry and call `_discover_blocks()`
  - [x] 3.3 Implement `_discover_blocks()` to recursively scan blocks/ directory
  - [x] 3.4 Implement `get_block()` to retrieve parsed capabilities by name
  - [x] 3.5 Implement `list_blocks()` to return all registered block names
  - [x] 3.6 Implement `validate_interface()` to check BlockInterface protocol compliance
  - [x] 3.7 Add warning logs for missing block.yaml files
  - [x] 3.8 Add error handling for duplicate block names
  - [x] 3.9 Write unit tests with mock block directories
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 10.4, 10.5_

- [x] 4. Implement GraphLoader
  - [x] 4.1 Create `src/core/graph_loader.py`
  - [x] 4.2 Define `GraphNode`, `GraphEdge`, `ArchitectureGraph` dataclasses
  - [x] 4.3 Implement `load()` to parse architecture YAML files
  - [x] 4.4 Implement `parse_topology()` to convert topology strings into GraphEdge objects
  - [x] 4.5 Implement `validate_references()` to check all blocks exist in registry
  - [x] 4.6 Implement `validate_parameters()` to verify component params match block schema
  - [x] 4.7 Implement `detect_cycles()` using depth-first search
  - [x] 4.8 Write unit tests for valid and invalid architectures
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

### Phase 2: Validation Pipeline

- [x] 5. Implement ShapeValidator
  - [x] 5.1 Create `src/core/shape_validator.py`
  - [x] 5.2 Define `ShapePattern` and `UnificationResult` dataclasses
  - [x] 5.3 Implement `unify()` to match output and input shape patterns
  - [x] 5.4 Implement `match_dimension()` to handle wildcards and named dimensions
  - [x] 5.5 Implement `validate_consistency()` to check dimension bindings
  - [x] 5.6 Add error reporting for shape mismatches with expected/actual shapes
  - [x] 5.7 Write unit tests for compatible and incompatible shape pairs
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 6. Implement HardwareDetector
  - [ ] 6.1 Create `src/core/hardware_detector.py`
  - [ ] 6.2 Define `HardwareCapabilities` dataclass
  - [ ] 6.3 Implement `check_cuda_available()` using torch.cuda.is_available()
  - [ ] 6.4 Implement `get_compute_capability()` using torch.cuda.get_device_capability()
  - [ ] 6.5 Implement `_detect()` to query CPU cores and memory using psutil
  - [ ] 6.6 Implement `is_block_compatible()` to check hardware requirements
  - [ ] 6.7 Add memory estimation and warning for blocks exceeding available resources
  - [ ] 6.8 Write unit tests with mocked hardware detection
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7. Implement ConstraintSolver
  - [x] 7.1 Create `src/core/constraint_solver.py`
  - [x] 7.2 Define `Configuration` dataclass
  - [x] 7.3 Implement `solve()` to enumerate valid dimension configurations
  - [x] 7.4 Implement constraint propagation algorithm
  - [x] 7.5 Implement `check_satisfiable()` to detect unsatisfiable constraints
  - [x] 7.6 Implement `apply_configuration()` to bind dimensions to values
  - [x] 7.7 Implement `detect_conflicts()` to find conflicting constraints
  - [x] 7.8 Add parameter count and memory estimation for each configuration
  - [x] 7.9 Write unit tests for solvable and unsolvable constraint sets
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 8. Implement GraphValidator
  - [ ] 8.1 Create `src/core/graph_validator.py`
  - [ ] 8.2 Define `ValidationError` and `ValidationResult` dataclasses
  - [ ] 8.3 Implement `validate()` to orchestrate all validation steps
  - [ ] 8.4 Implement `create_shape_error()` with actionable suggestions
  - [ ] 8.5 Implement `create_constraint_error()` with valid alternatives
  - [ ] 8.6 Implement `suggest_similar_blocks()` using fuzzy string matching
  - [ ] 8.7 Add hardware insufficiency error messages with suggestions
  - [ ] 8.8 Ensure all error messages use clear, non-technical language
  - [ ] 8.9 Write integration tests combining all validators
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

### Phase 3: Compilation & Execution

- [ ] 9. Implement CompilationEngine
  - [ ] 9.1 Create `src/core/compilation_engine.py`
  - [ ] 9.2 Create code template using jinja2 for PyTorch module generation
  - [ ] 9.3 Implement `compile()` to generate Python module file
  - [ ] 9.4 Implement `generate_sequential()` for linear block chains
  - [ ] 9.5 Implement `generate_parallel()` for parallel execution paths
  - [ ] 9.6 Implement `generate_shape_assertions()` for runtime validation
  - [ ] 9.7 Add import statements for block implementations
  - [ ] 9.8 Format generated code using black
  - [ ] 9.9 Write tests that compile and execute generated modules
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 10. Implement ContainerRuntime
  - [ ] 10.1 Create `src/core/container_runtime.py`
  - [ ] 10.2 Define `ExecutionResult` dataclass
  - [ ] 10.3 Create Dockerfile for neuroscript-runtime image with PyTorch
  - [ ] 10.4 Implement `execute()` to run models in containers
  - [ ] 10.5 Implement `create_container()` with resource limits
  - [ ] 10.6 Implement `get_resource_limits()` based on hardware detection
  - [ ] 10.7 Implement `handle_oom()` for memory limit violations
  - [ ] 10.8 Add GPU passthrough via --gpus flag when available
  - [ ] 10.9 Capture and return stdout/stderr from container execution
  - [ ] 10.10 Write tests using Docker test containers
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

### Phase 4: Example Blocks

- [ ] 11. Create Linear Block
  - [ ] 11.1 Create `blocks/linear/` directory
  - [ ] 11.2 Write `blocks/linear/block.yaml` with capability specification
  - [ ] 11.3 Implement `blocks/linear/module.py` with BlockInterface
  - [ ] 11.4 Write `blocks/linear/test.py` with shape and parameter tests
  - [ ] 11.5 Test with CapabilityParser and BlockRegistry
  - _Requirements: 1.1, 1.2, 1.4, 10.1, 10.2, 10.3_

- [ ] 12. Create Embedding Block
  - [ ] 12.1 Create `blocks/embedding/` directory
  - [ ] 12.2 Write `blocks/embedding/block.yaml` with vocab/dim parameters
  - [ ] 12.3 Implement `blocks/embedding/module.py` using nn.Embedding
  - [ ] 12.4 Write unit tests for discrete input to dense output transformation
  - [ ] 12.5 Test integration with GraphLoader
  - _Requirements: 1.1, 1.2, 1.4, 10.1, 10.2, 10.3_

- [ ] 13. Create LayerNorm Block
  - [ ] 13.1 Create `blocks/layernorm/` directory
  - [ ] 13.2 Write `blocks/layernorm/block.yaml` with shape-preserving spec
  - [ ] 13.3 Implement `blocks/layernorm/module.py` using nn.LayerNorm
  - [ ] 13.4 Write tests verifying input shape == output shape
  - [ ] 13.5 Test with ShapeValidator for shape preservation
  - _Requirements: 1.1, 1.2, 4.1, 10.1, 10.2, 10.3_

- [ ] 14. Create Dropout Block
  - [ ] 14.1 Create `blocks/dropout/` directory
  - [ ] 14.2 Write `blocks/dropout/block.yaml` with dropout probability parameter
  - [ ] 14.3 Implement `blocks/dropout/module.py` using nn.Dropout
  - [ ] 14.4 Write tests for shape preservation and probability range validation
  - [ ] 14.5 Test with ConstraintSolver for parameter validation
  - _Requirements: 1.1, 1.4, 10.1, 10.2, 10.3_

- [ ] 15. Create Sequential Composition Block
  - [ ] 15.1 Create `blocks/sequential/` directory
  - [ ] 15.2 Write `blocks/sequential/block.yaml` accepting list of blocks
  - [ ] 15.3 Implement `blocks/sequential/module.py` using nn.Sequential
  - [ ] 15.4 Write tests composing Linear -> LayerNorm -> Dropout
  - [ ] 15.5 Test with GraphValidator for multi-block composition
  - _Requirements: 3.2, 4.1, 4.5, 10.1, 10.2, 10.3_

### Phase 5: Integration & Testing

- [ ] 16. Create End-to-End Example Architecture
  - [ ] 16.1 Write `examples/simple_mlp.yaml` using Linear, LayerNorm, Dropout blocks
  - [ ] 16.2 Validate architecture using GraphValidator
  - [ ] 16.3 Compile to PyTorch using CompilationEngine
  - [ ] 16.4 Execute in container using ContainerRuntime
  - [ ] 16.5 Verify output correctness
  - _Requirements: 3.1, 3.2, 4.1, 6.1, 8.1, 9.1_

- [ ] 17. Create CLI Interface
  - [ ] 17.1 Create `src/cli/main.py` with argparse
  - [ ] 17.2 Implement `neuroscript validate` command
  - [ ] 17.3 Implement `neuroscript compile` command
  - [ ] 17.4 Implement `neuroscript run` command
  - [ ] 17.5 Add `--input-shape` flag for constraint solving
  - [ ] 17.6 Add `--output` flag for compilation target
  - [ ] 17.7 Format validation errors for terminal output
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [ ] 18. Write Integration Tests
  - [ ] 18.1 Create `test/integration/test_validation_pipeline.py`
  - [ ] 18.2 Test valid architecture validation end-to-end
  - [ ] 18.3 Test shape mismatch error reporting
  - [ ] 18.4 Test hardware incompatibility detection
  - [ ] 18.5 Test constraint solving with multiple configurations
  - [ ] 18.6 Test compilation and container execution
  - _Requirements: 4.1, 4.3, 5.3, 6.1, 7.1, 8.1, 9.1_

- [ ] 19. Create Hardware Detection Tests
  - [ ] 19.1 Create `test/hardware/test_cuda_detection.py`
  - [ ] 19.2 Test CUDA availability detection
  - [ ] 19.3 Test compute capability detection
  - [ ] 19.4 Test memory detection
  - [ ] 19.5 Test block filtering based on hardware
  - [ ] 19.6 Mock hardware for CI/CD environments
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 20. Write Documentation
  - [ ] 20.1 Create `docs/block-specification.md` documenting block.yaml format
  - [ ] 20.2 Create `docs/architecture-format.md` documenting architecture YAML
  - [ ] 20.3 Create `docs/creating-blocks.md` tutorial for block development
  - [ ] 20.4 Update `README.md` with installation and usage instructions
  - [ ] 20.5 Create `examples/` directory with 3 example architectures
  - [ ] 20.6 Document error messages and troubleshooting guide
  - _Requirements: 1.1, 3.1, 7.5_

- [ ] 21. Setup CI/CD Pipeline
  - [ ] 21.1 Create `.github/workflows/test.yml` for automated testing
  - [ ] 21.2 Run pytest on all unit and integration tests
  - [ ] 21.3 Build Docker runtime image in CI
  - [ ] 21.4 Test on both CPU and CUDA environments
  - [ ] 21.5 Add code coverage reporting
  - _Requirements: All_

---

## Development Dependencies

### Phase 1 Dependencies
- Tasks 1-4 can be developed in parallel
- Task 4 depends on Tasks 1-3 (needs BlockInterface, CapabilityParser, BlockRegistry)

### Phase 2 Dependencies
- Tasks 5-7 can be developed in parallel after Phase 1
- Task 8 depends on Tasks 5-7 (orchestrates all validators)

### Phase 3 Dependencies
- Task 9 depends on Phase 1 and Phase 2
- Task 10 can be developed in parallel with Task 9

### Phase 4 Dependencies
- Tasks 11-15 can be developed in parallel after Phase 1 complete
- All blocks follow the same template pattern

### Phase 5 Dependencies
- Task 16 requires all previous phases complete
- Tasks 17-21 can proceed in parallel with each other

---

## Task Summary

- **Total Tasks**: 21 major tasks
- **Total Subtasks**: 179
- **Estimated Timeline**: 5 weeks with parallel development
- **Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 5
- **Parallel Work**: Phases 1 & 4 can overlap, Phase 5 tasks can run concurrently
