"""A small, dependency-free JSON-Schema validator (quiz Q6).

The real ``jsonschema`` package is the production choice, but implementing a
focused subset from scratch keeps this project zero-dependency *and* makes the
mechanics visible: a schema is just data describing required fields, types, and
constraints, and validation is a recursive walk over that data.

Supported keywords:
    type        -> "object" | "array" | "string" | "number" | "integer"
                   | "boolean" | "null"  (a list of these is also allowed)
    required    -> list of property names that must be present (objects)
    properties  -> per-property sub-schemas (objects)
    items       -> sub-schema applied to every array element
    enum        -> value must be one of the listed options
    minimum     -> numeric lower bound (inclusive)
    minLength   -> minimum string length

``validate`` returns a list of error strings (empty == valid) so callers can
collect every problem at once; ``validate_or_raise`` is the strict wrapper.
"""

from __future__ import annotations

from typing import Any

from .errors import SchemaValidationError

# Map JSON-Schema type names to Python types. ``bool`` is checked before
# ``int`` everywhere because in Python ``bool`` is a subclass of ``int``.
_PY_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _matches_type(value: Any, type_name: str) -> bool:
    """True if ``value`` matches a single JSON-Schema type name."""
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name in ("number", "integer"):
        # Exclude bool, which would otherwise sneak through as an int.
        if isinstance(value, bool):
            return False
        return isinstance(value, _PY_TYPES[type_name])
    return isinstance(value, _PY_TYPES[type_name])


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate ``instance`` against ``schema``; return a list of error strings."""
    errors: list[str] = []

    # --- type -------------------------------------------------------------
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(instance, n) for n in names):
            got = type(instance).__name__
            errors.append(f"{path}: expected type {expected!r}, got {got}")
            # If the base type is wrong, deeper checks are noise; stop here.
            return errors

    # --- enum -------------------------------------------------------------
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']!r}")

    # --- numbers ----------------------------------------------------------
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is below minimum {schema['minimum']}")

    # --- strings ----------------------------------------------------------
    if isinstance(instance, str) and "minLength" in schema:
        if len(instance) < schema["minLength"]:
            errors.append(
                f"{path}: string shorter than minLength {schema['minLength']}"
            )

    # --- objects ----------------------------------------------------------
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        for key, subschema in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(validate(instance[key], subschema, f"{path}.{key}"))

    # --- arrays -----------------------------------------------------------
    if isinstance(instance, list) and "items" in schema:
        for i, element in enumerate(instance):
            errors.extend(validate(element, schema["items"], f"{path}[{i}]"))

    return errors


def is_valid(instance: Any, schema: dict[str, Any]) -> bool:
    """Convenience boolean wrapper around :func:`validate`."""
    return not validate(instance, schema)


def validate_or_raise(instance: Any, schema: dict[str, Any], *, what: str) -> None:
    """Validate and raise :class:`SchemaValidationError` if anything is wrong."""
    errors = validate(instance, schema)
    if errors:
        raise SchemaValidationError(
            f"{what} failed schema validation: " + "; ".join(errors),
            errors=errors,
        )
