"""JSON Schema validator — built directly from schema files, not hand-transcribed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


class SchemaValidationError(Exception):
    def __init__(self, errors: List[str]) -> None:
        super().__init__("Schema validation failed")
        self.errors = errors


class SchemaValidator:
    """Validates a dict against a JSON Schema (draft-agnostic subset)."""

    def __init__(self, schema_path: str) -> None:
        with Path(schema_path).open("r", encoding="utf-8") as fh:
            self.schema = json.load(fh)

    def validate(self, payload: Any) -> None:
        errors: List[str] = []
        self._check(self.schema, payload, [], errors)
        if errors:
            raise SchemaValidationError(errors)

    def _check(self, schema: Dict, value: Any, path: List, errors: List) -> None:
        # type check
        expected = schema.get("type")
        if expected is not None and not self._type_ok(expected, value):
            errors.append(f"{self._path(path)}: expected {expected}, got {type(value).__name__}")
            return

        if "const" in schema and value != schema["const"]:
            errors.append(f"{self._path(path)}: expected const {schema['const']!r}, got {value!r}")
            return

        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{self._path(path)}: expected one of {schema['enum']!r}, got {value!r}")
            return

        if "minimum" in schema and isinstance(value, (int, float)):
            if value < schema["minimum"]:
                errors.append(f"{self._path(path)}: value {value} < minimum {schema['minimum']}")

        if "maximum" in schema and isinstance(value, (int, float)):
            if value > schema["maximum"]:
                errors.append(f"{self._path(path)}: value {value} > maximum {schema['maximum']}")

        if isinstance(value, dict):
            self._check_object(schema, value, path, errors)
        elif isinstance(value, list):
            self._check_array(schema, value, path, errors)

    def _check_object(self, schema: Dict, value: Dict, path: List, errors: List) -> None:
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{self._path(path + [key])}: required field missing")
        for key, item in value.items():
            if key in props:
                self._check(props[key], item, path + [key], errors)

    def _check_array(self, schema: Dict, value: List, path: List, errors: List) -> None:
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(value):
                self._check(items_schema, item, path + [i], errors)
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{self._path(path)}: need ≥{schema['minItems']} items, got {len(value)}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{self._path(path)}: need ≤{schema['maxItems']} items, got {len(value)}")
        # contains: at least one item must match
        if "contains" in schema:
            contains = schema["contains"]
            matched = any(not self._sub_errors(contains, item) for item in value)
            if not matched:
                errors.append(f"{self._path(path)}: no item satisfies 'contains' constraint")

    def _sub_errors(self, schema: Dict, value: Any) -> List[str]:
        errs: List[str] = []
        self._check(schema, value, [], errs)
        return errs

    def _type_ok(self, expected: Union[str, List], value: Any) -> bool:
        if isinstance(expected, list):
            return any(self._type_ok(t, value) for t in expected)
        if expected == "null":
            return value is None
        if expected == "string":
            return isinstance(value, str)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        return True

    def _path(self, path: List) -> str:
        return ".".join(str(p) for p in path) or "<root>"
