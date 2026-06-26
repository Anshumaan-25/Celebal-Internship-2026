"""Tests for the from-scratch JSON-schema validator (Q6)."""

import unittest

from agentic_pipeline.errors import SchemaValidationError
from agentic_pipeline.schemas import is_valid, validate, validate_or_raise


class TestSchemas(unittest.TestCase):
    def test_type_checks(self):
        self.assertTrue(is_valid("hi", {"type": "string"}))
        self.assertTrue(is_valid(3, {"type": "integer"}))
        self.assertTrue(is_valid(3.5, {"type": "number"}))
        self.assertFalse(is_valid("hi", {"type": "integer"}))

    def test_bool_is_not_int(self):
        # bool is a subclass of int in Python; the validator must not conflate.
        self.assertFalse(is_valid(True, {"type": "integer"}))
        self.assertTrue(is_valid(True, {"type": "boolean"}))

    def test_required_and_properties(self):
        schema = {
            "type": "object",
            "required": ["expression"],
            "properties": {"expression": {"type": "string"}},
        }
        self.assertEqual(validate({"expression": "1+1"}, schema), [])
        errs = validate({}, schema)
        self.assertTrue(any("missing required" in e for e in errs))

    def test_nested_and_array(self):
        schema = {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
            },
        }
        self.assertEqual(validate({"keywords": ["a", "b"]}, schema), [])
        self.assertTrue(validate({"keywords": ["a", 2]}, schema))

    def test_enum_minimum_minlength(self):
        self.assertFalse(is_valid("x", {"enum": ["a", "b"]}))
        self.assertTrue(is_valid("a", {"enum": ["a", "b"]}))
        self.assertFalse(is_valid(0, {"type": "integer", "minimum": 1}))
        self.assertFalse(is_valid("", {"type": "string", "minLength": 1}))

    def test_validate_or_raise(self):
        with self.assertRaises(SchemaValidationError) as ctx:
            validate_or_raise({}, {"type": "object", "required": ["x"]}, what="thing")
        self.assertTrue(ctx.exception.errors)


if __name__ == "__main__":
    unittest.main()
