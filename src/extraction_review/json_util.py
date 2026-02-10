"""Utilities for working with JSON schemas."""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

from json_schema_to_pydantic import create_model
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _hash_schema(json_schema: dict[str, Any]) -> str:
    """Create a stable hash of a JSON schema for caching."""
    schema_str = json.dumps(json_schema, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()


@lru_cache(maxsize=16)
def _get_cached_model(schema_hash: str, schema_json: str) -> type[BaseModel]:
    """Get or create a Pydantic model from a JSON schema, cached by hash."""
    schema = json.loads(schema_json)
    return create_model(schema)


def get_extraction_schema(
    json_schema: dict[str, Any],
    discriminator_field: str | None = None,
    discriminator_value: str | None = None,
) -> type[BaseModel]:
    """Convert a JSON schema to a Pydantic model for validating extraction results.

    Use this to create a schema class for `ExtractedData.from_extraction_result()`.
    Results are cached by schema hash for efficiency.

    Args:
        json_schema: A JSON Schema object from config (e.g., `extract_config.json_schema`).
        discriminator_field: For multi-schema workflows, the field name to identify
            document type (e.g., "document_type"). When set, adds this field with
            a default value so extraction results validate correctly.
        discriminator_value: The value for the discriminator field (e.g., "invoice").
            Required if discriminator_field is set.

    Returns:
        A Pydantic model class for validation.

    Example:
        ```python
        # Single-schema workflow
        schema_class = get_extraction_schema(extract_config.json_schema)

        # Multi-schema workflow (adds discriminator with default value)
        schema_class = get_extraction_schema(
            extract_config.json_schema,
            discriminator_field="document_type",
            discriminator_value="invoice",
        )
        data = ExtractedData.from_extraction_result(result=extract_run, schema=schema_class, ...)
        # data.data.document_type == "invoice"
        ```
    """
    schema = json_schema

    # Add discriminator field if specified
    if discriminator_field is not None:
        if discriminator_value is None:
            raise ValueError(
                "discriminator_value is required when discriminator_field is set"
            )
        schema = _add_discriminator_to_schema(
            schema, discriminator_field, discriminator_value
        )

    schema_hash = _hash_schema(schema)
    schema_json = json.dumps(schema, sort_keys=True)
    return _get_cached_model(schema_hash, schema_json)


def _add_discriminator_to_schema(
    schema: dict[str, Any],
    discriminator_field: str,
    discriminator_value: str,
) -> dict[str, Any]:
    """Add a discriminator field with a default value to a schema."""
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    new_properties = {
        discriminator_field: {
            "type": "string",
            "default": discriminator_value,
            "description": "Type of document that was extracted",
        },
        **properties,
    }

    new_required = list(required)
    if discriminator_field not in new_required:
        new_required = [discriminator_field] + new_required

    return {
        **schema,
        "properties": new_properties,
        "required": new_required,
    }


def _schemas_are_equal(schema1: dict[str, Any], schema2: dict[str, Any]) -> bool:
    """Check if two JSON schemas are structurally equal."""
    return json.dumps(schema1, sort_keys=True) == json.dumps(schema2, sort_keys=True)


def _merge_property_schemas(
    existing: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Merge two property schemas, creating anyOf for conflicting types."""
    if _schemas_are_equal(existing, new):
        return existing

    if "anyOf" in existing:
        for variant in existing["anyOf"]:
            if _schemas_are_equal(variant, new):
                return existing
        return {"anyOf": existing["anyOf"] + [new]}

    return {"anyOf": [existing, new]}


def create_union_schema(
    schemas: dict[str, dict[str, Any]],
    discriminator_field: str = "document_type",
) -> dict[str, Any]:
    """Create a union JSON schema from multiple extraction schemas.

    Merges all properties from input schemas into a single flat schema,
    adding a discriminator field. When the same field appears with different
    types across schemas, creates an anyOf to accommodate both.

    Args:
        schemas: Map of document type names to their JSON schemas.
        discriminator_field: Name of the field to add for document type.

    Returns:
        A JSON schema with all fields from all schemas plus a discriminator.
    """
    schemas_with_discriminator = [
        name
        for name, schema in schemas.items()
        if discriminator_field in schema.get("properties", {})
    ]
    if schemas_with_discriminator:
        logger.warning(
            f"Discriminator field '{discriminator_field}' found in schemas: "
            f"{', '.join(schemas_with_discriminator)}. "
            f"It will be replaced with the union discriminator."
        )

    all_properties: dict[str, Any] = {}
    for schema in schemas.values():
        for prop_name, prop_def in schema.get("properties", {}).items():
            if prop_name == discriminator_field:
                continue
            if prop_name not in all_properties:
                all_properties[prop_name] = prop_def
            else:
                all_properties[prop_name] = _merge_property_schemas(
                    all_properties[prop_name], prop_def
                )

    schema_list = list(schemas.values())
    common_required: set[str] = set()
    if schema_list:
        common_required = set(schema_list[0].get("required", []))
        for schema in schema_list[1:]:
            common_required &= set(schema.get("required", []))

    common_required.discard(discriminator_field)

    required_fields = [discriminator_field] + sorted(common_required)
    return {
        "type": "object",
        "properties": {
            discriminator_field: {
                "type": "string",
                "enum": list(schemas.keys()),
                "description": "Type of document that was extracted",
            },
            **all_properties,
        },
        "required": required_fields,
    }
