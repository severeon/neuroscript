"""CapabilityParser for parsing block.yaml capability specifications."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .block_interface import (
    BlockCapability,
    DimensionConstraint,
    ParameterSpec,
    ShapePattern,
)


class CapabilityParseError(Exception):
    """Raised when capability parsing fails."""

    pass


class CapabilityParser:
    """Parser for block.yaml capability specifications."""

    # Regex pattern for parsing shape patterns like "[batch, seq, dim]"
    SHAPE_PATTERN_RE = re.compile(r'\[([^\]]+)\]')
    # Regex pattern for parsing constraints like "dim % 8 == 0" or "dim in [256, 512]"
    CONSTRAINT_RE = re.compile(r'^[a-zA-Z0-9_\.\[\]%\s<>=!&|+\-\*\(\),]+$')

    def parse_file(self, file_path: str) -> BlockCapability:
        """
        Parse a block.yaml file into a BlockCapability object.

        Args:
            file_path: Path to block.yaml file

        Returns:
            BlockCapability: Parsed block capability specification

        Raises:
            CapabilityParseError: If YAML is malformed or missing required fields
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise CapabilityParseError(f"File not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise CapabilityParseError(f"Malformed YAML in {file_path}: {e}")
        except (OSError, IOError) as e:
            raise CapabilityParseError(f"Error reading {file_path}: {e}")

        if not isinstance(data, dict):
            raise CapabilityParseError(f"Expected YAML dict at {file_path}, got {type(data)}")

        # Validate required fields
        name = data.get('name')
        version = data.get('version')

        if not name:
            raise CapabilityParseError(
                f"Missing required field 'name' in {file_path}"
            )
        if not version:
            raise CapabilityParseError(
                f"Missing required field 'version' in {file_path}"
            )

        # Parse capabilities section
        capabilities = data.get('capabilities', {})

        if not isinstance(capabilities, dict):
            raise CapabilityParseError(
                f"'capabilities' must be a dict in {file_path}"
            )

        # Parse inputs
        inputs = self._parse_io_specs(capabilities.get('inputs', {}), 'inputs', file_path)

        # Parse outputs
        outputs = self._parse_io_specs(capabilities.get('outputs', {}), 'outputs', file_path)

        # Parse parameters
        params = self._parse_params(capabilities.get('params', {}), file_path)

        # Parse constraints
        constraints = self._parse_constraints(capabilities.get('constraints', []), file_path)

        # Parse hardware requirements
        hardware_requirements = data.get('hardware_requirements')

        return BlockCapability(
            name=str(name),
            version=str(version),
            inputs=inputs,
            outputs=outputs,
            params=params,
            constraints=constraints,
            hardware_requirements=hardware_requirements,
        )

    def _parse_io_specs(self, specs: Dict[str, Any], io_type: str, file_path: Path) -> Dict[str, ShapePattern]:
        """
        Parse input or output specifications.

        Args:
            specs: IO specifications dict
            io_type: 'inputs' or 'outputs' for error messages
            file_path: Path for error reporting

        Returns:
            Dict mapping names to ShapePattern objects
        """
        if not isinstance(specs, dict):
            raise CapabilityParseError(
                f"'{io_type}' must be a dict in {file_path}, got {type(specs)}"
            )

        result = {}

        for name, spec in specs.items():
            if not isinstance(spec, dict):
                raise CapabilityParseError(
                    f"'{io_type}.{name}' must be a dict in {file_path}"
                )

            shape = spec.get('shape')
            dtype = spec.get('dtype')

            if shape is None:
                raise CapabilityParseError(
                    f"Missing 'shape' in '{io_type}.{name}' in {file_path}"
                )

            # Extract dimensions from shape pattern
            dimensions = self.extract_dimensions(shape, file_path)

            # Parse dtype if present
            if dtype is not None and not isinstance(dtype, list):
                dtype = [dtype]

            result[name] = ShapePattern(pattern=dimensions, dtype=dtype)

        return result

    def _parse_params(self, params: Dict[str, Any], file_path: Path) -> Dict[str, ParameterSpec]:
        """
        Parse parameter specifications.

        Args:
            params: Parameters dict
            file_path: Path for error reporting

        Returns:
            Dict mapping names to ParameterSpec objects
        """
        if not isinstance(params, dict):
            raise CapabilityParseError(
                f"'params' must be a dict in {file_path}, got {type(params)}"
            )

        result = {}

        for name, spec in params.items():
            if not isinstance(spec, dict):
                raise CapabilityParseError(
                    f"'params.{name}' must be a dict in {file_path}"
                )

            param_type = spec.get('type')
            if not param_type:
                raise CapabilityParseError(
                    f"Missing 'type' in 'params.{name}' in {file_path}"
                )

            required = spec.get('required', True)
            default = spec.get('default')
            range_spec = spec.get('range')
            options = spec.get('options')

            # Parse range if present
            parsed_range = None
            if range_spec is not None:
                if not isinstance(range_spec, (list, tuple)) or len(range_spec) != 2:
                    raise CapabilityParseError(
                        f"'range' in 'params.{name}' must be [min, max] in {file_path}"
                    )
                try:
                    parsed_range = (float(range_spec[0]), float(range_spec[1]))
                except (ValueError, TypeError):
                    raise CapabilityParseError(
                        f"Range values must be numeric in 'params.{name}' in {file_path}"
                    )

            result[name] = ParameterSpec(
                name=name,
                type=str(param_type),
                required=bool(required),
                default=default,
                range=parsed_range,
                options=options,
            )

        return result

    def _parse_constraints(self, constraints: Any, file_path: Path) -> List[DimensionConstraint]:
        """
        Parse constraint specifications.

        Args:
            constraints: Constraints list or dict
            file_path: Path for error reporting

        Returns:
            List of DimensionConstraint objects
        """
        if not isinstance(constraints, (list, dict)):
            raise CapabilityParseError(
                f"'constraints' must be a list or dict in {file_path}, got {type(constraints)}"
            )

        result = []

        if isinstance(constraints, dict):
            constraints = list(constraints.values())

        for constraint in constraints:
            if isinstance(constraint, str):
                self._validate_constraint(constraint, file_path)
                result.append(DimensionConstraint(constraint))
            elif isinstance(constraint, dict) and 'constraint' in constraint:
                constraint_str = constraint['constraint']
                if not isinstance(constraint_str, str):
                    raise CapabilityParseError(
                        f"Constraint must be a string in {file_path}"
                    )
                self._validate_constraint(constraint_str, file_path)
                result.append(DimensionConstraint(constraint_str))
            else:
                raise CapabilityParseError(
                    f"Invalid constraint format in {file_path}: {constraint}"
                )

        return result

    def extract_dimensions(self, shape: Any, file_path: Optional[Path] = None) -> List[str]:
        """
        Extract dimension names from a shape pattern.

        Supports formats like:
        - "[batch, seq, dim]" → ["batch", "seq", "dim"]
        - ["batch", "seq", "dim"] → ["batch", "seq", "dim"]
        - "batch seq dim" → ["batch", "seq", "dim"]

        Args:
            shape: Shape specification (string or list)
            file_path: Path for error reporting

        Returns:
            List of dimension names

        Raises:
            CapabilityParseError: If shape format is invalid
        """
        if isinstance(shape, list):
            # Already a list
            return [str(d).strip() for d in shape]

        if not isinstance(shape, str):
            if file_path:
                raise CapabilityParseError(
                    f"Shape must be string or list in {file_path}, got {type(shape)}"
                )
            raise CapabilityParseError(
                f"Shape must be string or list, got {type(shape)}"
            )

        # Try to parse as "[dim1, dim2, ...]"
        match = self.SHAPE_PATTERN_RE.search(shape)
        if match:
            dims_str = match.group(1)
            return [d.strip() for d in dims_str.split(',')]

        # Try to parse as "dim1 dim2 ..."
        return [d.strip() for d in shape.split() if d.strip()]

    def _validate_constraint(self, constraint_str: str, file_path: Path) -> None:
        """
        Validate constraint string format.

        Args:
            constraint_str: Constraint expression
            file_path: Path for error reporting

        Raises:
            CapabilityParseError: If constraint format is invalid
        """
        if not constraint_str or not constraint_str.strip():
            raise CapabilityParseError(
                f"Empty constraint string in {file_path}"
            )

        # Basic validation - constraint should be a reasonable expression
        # This is a simple check; a full expression validator would be more thorough
        if not self.CONSTRAINT_RE.match(constraint_str):
            raise CapabilityParseError(
                f"Invalid constraint format in {file_path}: {constraint_str}"
            )

    def validate_param_range(self, param_name: str, value: Any, param_spec: ParameterSpec, file_path: Optional[Path] = None) -> None:
        """
        Validate that a parameter value falls within its specified range.

        Args:
            param_name: Name of the parameter
            value: Parameter value to validate
            param_spec: ParameterSpec definition
            file_path: Path for error reporting

        Raises:
            CapabilityParseError: If value is outside allowed range
        """
        if param_spec.range is None:
            return

        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            if file_path:
                raise CapabilityParseError(
                    f"Parameter '{param_name}' value {value} is not numeric in {file_path}"
                )
            raise CapabilityParseError(
                f"Parameter '{param_name}' value {value} is not numeric"
            )

        min_val, max_val = param_spec.range

        if numeric_value < min_val or numeric_value > max_val:
            msg = (
                f"Parameter '{param_name}' value {numeric_value} "
                f"out of range [{min_val}, {max_val}]"
            )
            if file_path:
                msg += f" in {file_path}"
            raise CapabilityParseError(msg)
