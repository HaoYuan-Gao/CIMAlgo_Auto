"""Shared checks for mappings, strings and integers."""

import operator
from collections.abc import Mapping
from typing import Any


def require_mapping(
    value: Any,
    field: str,
    *,
    error_type: type[Exception] = ValueError,
) -> Mapping[str, Any]:
    """Return a mapping or raise a field-aware validation error."""
    if not isinstance(value, Mapping):
        raise error_type(f"{field} must be a mapping")
    return value


def require_string(
    value: Any,
    field: str,
    *,
    error_type: type[Exception] = ValueError,
) -> str:
    """Return a stripped, non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{field} must be a non-empty string")
    return value.strip()


def require_integer(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    error_type: type[Exception] = ValueError,
) -> int:
    """Return a strict integer, optionally constrained by a minimum value."""
    if isinstance(value, bool):
        raise error_type(f"{field} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise error_type(f"{field} must be an integer") from error
    if minimum is not None and value < minimum:
        raise error_type(f"{field} must be at least {minimum}")
    return value
