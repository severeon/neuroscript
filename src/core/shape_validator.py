"""
ShapeValidator: Validates tensor shape compatibility between connected blocks.

Implements Requirements 4.1, 4.2, 4.3, 4.4, 4.5 from specs/02-requirements.md
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class ShapePattern:
    """Represents a shape pattern with dimensions.

    Implements Req 4.1

    Examples:
        ShapePattern(["batch", "seq", "dim"], [False, False, False])
        ShapePattern(["*", "*", "512"], [True, True, False])
    """
    dimensions: List[str]  # e.g., ["batch", "seq", "dim"] or ["*", "512"]
    is_wildcard: List[bool]  # Which dimensions are wildcards

    def __post_init__(self):
        """Validate that dimensions and is_wildcard have same length."""
        if len(self.dimensions) != len(self.is_wildcard):
            raise ValueError(
                f"dimensions and is_wildcard must have same length: "
                f"{len(self.dimensions)} != {len(self.is_wildcard)}"
            )

    def __len__(self) -> int:
        """Return number of dimensions."""
        return len(self.dimensions)

    def __str__(self) -> str:
        """Return string representation of shape pattern."""
        return f"[{', '.join(self.dimensions)}]"


@dataclass
class UnificationResult:
    """Result of shape unification.

    Implements Req 4.2, 4.3
    """
    success: bool
    bindings: Dict[str, Union[int, str]] = field(default_factory=dict)  # dimension -> value or variable
    errors: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)


class ShapeValidator:
    """Validates shape compatibility via unification.

    Implements Requirements 4.1, 4.2, 4.3, 4.4, 4.5

    The ShapeValidator uses a unification algorithm to match shape patterns
    between connected blocks. It maintains a global set of dimension bindings
    to ensure consistency across the entire graph.
    """

    def __init__(self):
        """Initialize validator with empty bindings."""
        self.bindings: Dict[str, Union[int, str]] = {}  # Global dimension bindings

    def unify(
        self,
        output_pattern: ShapePattern,
        input_pattern: ShapePattern
    ) -> UnificationResult:
        """Unify two shape patterns.

        Implements Req 4.1, 4.2, 4.3

        Unification attempts to match an output shape pattern from one block
        with an input shape pattern from another block. Successful unification
        produces a set of dimension bindings.

        Args:
            output_pattern: Shape pattern from source block's output
            input_pattern: Shape pattern from target block's input

        Returns:
            UnificationResult with success status and bindings or errors

        Examples:
            >>> validator = ShapeValidator()
            >>> out_pat = ShapePattern(["batch", "seq", "512"], [False, False, False])
            >>> in_pat = ShapePattern(["batch", "seq", "dim"], [False, False, False])
            >>> result = validator.unify(out_pat, in_pat)
            >>> result.success
            True
            >>> result.bindings
            {'dim': '512'}
        """
        result = UnificationResult(success=True, bindings={})

        # Check rank compatibility
        if len(output_pattern) != len(input_pattern):
            result.success = False
            result.add_error(
                f"Shape rank mismatch: output has {len(output_pattern)} dimensions "
                f"{output_pattern}, input requires {len(input_pattern)} dimensions "
                f"{input_pattern}"
            )
            return result

        # Match each dimension pair
        for i, (out_dim, in_dim) in enumerate(
            zip(output_pattern.dimensions, input_pattern.dimensions)
        ):
            out_is_wildcard = output_pattern.is_wildcard[i]
            in_is_wildcard = input_pattern.is_wildcard[i]

            success, binding_value = self.match_dimension(
                out_dim, in_dim, out_is_wildcard, in_is_wildcard
            )

            if not success:
                result.success = False
                # Check if this is a binding inconsistency
                out_int = self._try_parse_int(out_dim)
                in_int = self._try_parse_int(in_dim)

                # Check if one is concrete and the other is a bound variable
                if out_int is not None and in_dim in self.bindings:
                    result.add_error(
                        f"Inconsistent binding for '{in_dim}': "
                        f"attempting to bind to {out_dim}, but already bound to {self.bindings[in_dim]}"
                    )
                elif in_int is not None and out_dim in self.bindings:
                    result.add_error(
                        f"Inconsistent binding for '{out_dim}': "
                        f"attempting to bind to {in_dim}, but already bound to {self.bindings[out_dim]}"
                    )
                else:
                    result.add_error(
                        f"Dimension {i} mismatch: output={out_dim}, input={in_dim}"
                    )
            elif binding_value is not None:
                # Record the binding
                result.bindings[binding_value[0]] = binding_value[1]

        # Validate all bindings are consistent
        if result.success:
            for var, value in result.bindings.items():
                if not self.validate_consistency(var, value):
                    result.success = False
                    existing = self.bindings.get(var)
                    result.add_error(
                        f"Inconsistent binding for '{var}': "
                        f"attempting to bind to {value}, but already bound to {existing}"
                    )

        # If successful, update global bindings
        if result.success:
            self.bindings.update(result.bindings)

        return result

    def match_dimension(
        self,
        dim1: str,
        dim2: str,
        is_wildcard1: bool,
        is_wildcard2: bool
    ) -> Tuple[bool, Optional[Tuple[str, Union[int, str]]]]:
        """Match two dimension specifications.

        Implements Req 4.4

        Handles various cases:
        - Wildcard matches anything
        - Concrete values must match exactly
        - Named dimensions can bind to concrete values or other names

        Args:
            dim1: First dimension (from output)
            dim2: Second dimension (from input)
            is_wildcard1: Whether dim1 is a wildcard (*)
            is_wildcard2: Whether dim2 is a wildcard (*)

        Returns:
            Tuple of (success, optional_binding)
            - success: True if dimensions are compatible
            - optional_binding: (variable_name, value) if a binding was created, None otherwise

        Examples:
            >>> validator = ShapeValidator()
            >>> validator.match_dimension("512", "512", False, False)
            (True, None)
            >>> validator.match_dimension("*", "512", True, False)
            (True, None)
            >>> validator.match_dimension("512", "dim", False, False)
            (True, ('dim', '512'))
        """
        # Wildcard always matches
        if is_wildcard1 or is_wildcard2:
            return (True, None)

        # Try to parse as integers
        dim1_int = self._try_parse_int(dim1)
        dim2_int = self._try_parse_int(dim2)

        # Both are concrete integers
        if dim1_int is not None and dim2_int is not None:
            if dim1_int == dim2_int:
                return (True, None)
            else:
                return (False, None)

        # dim1 is concrete, dim2 is variable
        if dim1_int is not None and dim2_int is None:
            # Check if dim2 is already bound
            if dim2 in self.bindings:
                existing = self.bindings[dim2]
                if str(existing) == dim1:
                    return (True, None)
                else:
                    return (False, None)
            # Bind dim2 to dim1's value
            return (True, (dim2, dim1))

        # dim1 is variable, dim2 is concrete
        if dim1_int is None and dim2_int is not None:
            # Check if dim1 is already bound
            if dim1 in self.bindings:
                existing = self.bindings[dim1]
                if str(existing) == dim2:
                    return (True, None)
                else:
                    return (False, None)
            # Bind dim1 to dim2's value
            return (True, (dim1, dim2))

        # Both are variables
        if dim1_int is None and dim2_int is None:
            # If both are the same variable name, they match
            if dim1 == dim2:
                return (True, None)

            # Check if either is already bound
            dim1_bound = dim1 in self.bindings
            dim2_bound = dim2 in self.bindings

            if dim1_bound and dim2_bound:
                # Both bound, check if they match
                if str(self.bindings[dim1]) == str(self.bindings[dim2]):
                    return (True, None)
                else:
                    return (False, None)
            elif dim1_bound:
                # Bind dim2 to dim1's value
                return (True, (dim2, self.bindings[dim1]))
            elif dim2_bound:
                # Bind dim1 to dim2's value
                return (True, (dim1, self.bindings[dim2]))
            else:
                # Neither bound, create equivalence (bind dim2 to dim1)
                return (True, (dim2, dim1))

        # Should not reach here
        return (False, None)

    def validate_consistency(
        self,
        dim: str,
        value: Union[int, str]
    ) -> bool:
        """Check if dimension binding is consistent with existing bindings.

        Implements Req 4.5

        Args:
            dim: Dimension variable name
            value: Value to bind to (can be int, str of int, or variable name)

        Returns:
            True if binding is consistent, False otherwise

        Examples:
            >>> validator = ShapeValidator()
            >>> validator.bindings['dim'] = '512'
            >>> validator.validate_consistency('dim', '512')
            True
            >>> validator.validate_consistency('dim', '256')
            False
        """
        if dim not in self.bindings:
            return True

        existing = self.bindings[dim]

        # Normalize values for comparison
        existing_str = str(existing)
        value_str = str(value)

        # If one is a variable reference and the other is too, check transitively
        if self._is_variable(existing_str) and self._is_variable(value_str):
            # Both are variables, resolve them
            existing_resolved = self._resolve_variable(existing_str)
            value_resolved = self._resolve_variable(value_str)
            return existing_resolved == value_resolved

        return existing_str == value_str

    def reset_bindings(self) -> None:
        """Reset all dimension bindings.

        Useful when validating multiple independent graphs.
        """
        self.bindings = {}

    def get_bindings(self) -> Dict[str, Union[int, str]]:
        """Get all current dimension bindings.

        Returns:
            Copy of current bindings dictionary
        """
        return self.bindings.copy()

    def _try_parse_int(self, s: str) -> Optional[int]:
        """Try to parse string as integer.

        Args:
            s: String to parse

        Returns:
            Integer value if parseable, None otherwise
        """
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    def _is_variable(self, s: str) -> bool:
        """Check if string represents a variable name (not a concrete value).

        Args:
            s: String to check

        Returns:
            True if s is a variable name, False if concrete value
        """
        return self._try_parse_int(s) is None

    def _resolve_variable(self, var: str) -> str:
        """Recursively resolve a variable to its final value.

        Args:
            var: Variable name to resolve

        Returns:
            Final resolved value (may still be a variable if not bound to concrete value)
        """
        if var not in self.bindings:
            return var

        value = str(self.bindings[var])
        if self._is_variable(value):
            # Recursively resolve
            return self._resolve_variable(value)
        else:
            return value


def parse_shape_pattern(shape_list: List[Union[str, int]]) -> ShapePattern:
    """Parse a shape specification into a ShapePattern.

    Helper function to convert shape lists from YAML into ShapePattern objects.

    Args:
        shape_list: List of dimension specs, e.g., ["batch", "seq", 512] or ["*", "*", "dim"]

    Returns:
        ShapePattern object

    Examples:
        >>> parse_shape_pattern(["batch", "seq", "dim"])
        ShapePattern(dimensions=['batch', 'seq', 'dim'], is_wildcard=[False, False, False])
        >>> parse_shape_pattern(["*", "*", 512])
        ShapePattern(dimensions=['*', '*', '512'], is_wildcard=[True, True, False])
    """
    dimensions = []
    is_wildcard = []

    for dim in shape_list:
        dim_str = str(dim)
        dimensions.append(dim_str)
        is_wildcard.append(dim_str == "*")

    return ShapePattern(dimensions=dimensions, is_wildcard=is_wildcard)
