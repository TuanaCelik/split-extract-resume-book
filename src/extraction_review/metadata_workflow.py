from typing import Annotated, Any

import jsonref
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.resource import ResourceConfig

from .config import EXTRACTED_DATA_COLLECTION, JsonSchema


class MetadataResponse(StopEvent):
    json_schema: dict[str, Any]
    extracted_data_collection: str
    # For multi-schema workflows: individual schemas by document type
    schemas: dict[str, dict[str, Any]] | None = None
    discriminator_field: str | None = None


class MetadataWorkflow(Workflow):
    """Provide extraction schema and configuration to the workflow editor."""

    @step
    async def get_metadata(
        self,
        _: StartEvent,
        extraction_schema: Annotated[
            JsonSchema,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract.json_schema",
                label="Extraction Schema",
                description="JSON Schema defining the fields to extract from documents",
            ),
        ],
    ) -> MetadataResponse:
        """Return the data schema and storage settings for the review interface."""
        schema_dict = extraction_schema.to_dict()
        json_schema = jsonref.replace_refs(schema_dict, proxies=False)
        return MetadataResponse(
            json_schema=json_schema,
            extracted_data_collection=EXTRACTED_DATA_COLLECTION,
        )


workflow = MetadataWorkflow(timeout=None)
