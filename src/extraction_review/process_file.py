import asyncio
import json
import logging
from typing import Annotated, Literal

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.beta.extracted_data import ExtractedData, InvalidExtractionData
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.resource import Resource, ResourceConfig

from .clients import agent_name, get_llama_cloud_client, project_id
from .config import (
    EXTRACTED_DATA_COLLECTION,
    ExtractConfig,
    SplitConfig,
    get_extraction_schema,
)

logger = logging.getLogger(__name__)


class FileEvent(StartEvent):
    file_id: str
    file_hash: str | None = None


class Status(Event):
    level: Literal["info", "warning", "error"]
    message: str


class SplitJobStartedEvent(Event):
    pass


class SplitCompletedEvent(Event):
    pass


class ExtractedEvent(Event):
    data: ExtractedData


class ExtractedInvalidEvent(Event):
    """Event for extraction results that failed validation."""

    data: ExtractedData[dict]


class ResumeSegment(BaseModel):
    """A resume segment identified by split."""

    pages: list[int]
    extract_job_id: str | None = None


class ExtractionState(BaseModel):
    file_id: str | None = None
    filename: str | None = None
    file_hash: str | None = None
    split_job_id: str | None = None
    resume_segments: list[ResumeSegment] = []
    extracted_count: int = 0


class ProcessFileWorkflow(Workflow):
    """Split a resume book and extract candidate information from each resume."""

    @step()
    async def start_split(
        self,
        event: FileEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        split_config: Annotated[
            SplitConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="split",
                label="Split Settings",
                description="Categories for splitting resume book into sections",
            ),
        ],
    ) -> SplitJobStartedEvent:
        """Split the resume book into individual sections."""
        file_id = event.file_id
        logger.info(f"Processing file {file_id}")

        # Get file metadata
        files_response = await llama_cloud_client.files.list(file_ids=[file_id])
        file_metadata = files_response.items[0]
        filename = file_metadata.name

        ctx.write_event_to_stream(
            Status(level="info", message=f"Splitting {filename} into sections")
        )

        # Start split job
        categories = [
            {"name": cat.name, "description": cat.description}
            for cat in split_config.categories
        ]
        split_job = await llama_cloud_client.beta.split.create(
            categories=categories,
            document_input={"type": "file_id", "value": file_id},
            splitting_strategy=split_config.settings.splitting_strategy.model_dump(),
            project_id=project_id,
        )

        file_hash = event.file_hash or file_metadata.external_file_id

        async with ctx.store.edit_state() as state:
            state.file_id = file_id
            state.filename = filename
            state.file_hash = file_hash
            state.split_job_id = split_job.id

        return SplitJobStartedEvent()

    @step()
    async def complete_split_and_start_extraction(
        self,
        event: SplitJobStartedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Resume Extraction",
                description="Fields to extract from each resume",
            ),
        ],
    ) -> SplitCompletedEvent:
        """Wait for split, then start extraction jobs for each resume segment."""
        state = await ctx.store.get_state()

        # Wait for split to complete
        split_result = await llama_cloud_client.beta.split.wait_for_completion(
            state.split_job_id
        )

        # Filter for resume segments only
        resume_segments: list[ResumeSegment] = []
        for segment in split_result.result.segments:
            if segment.category == "resume":
                resume_segments.append(ResumeSegment(pages=segment.pages))

        if not resume_segments:
            ctx.write_event_to_stream(
                Status(level="warning", message="No resumes found in the document")
            )
            async with ctx.store.edit_state() as s:
                s.resume_segments = []
            return SplitCompletedEvent()

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Found {len(resume_segments)} resume(s), extracting candidate data",
            )
        )

        # Start extraction jobs for each resume segment
        for segment in resume_segments:
            page_range = ",".join(str(p) for p in segment.pages)
            extract_job = await llama_cloud_client.extraction.run(
                config={
                    **extract_config.settings.model_dump(),
                    "page_range": page_range,
                },
                data_schema=extract_config.json_schema,
                file_id=state.file_id,
                project_id=project_id,
            )
            segment.extract_job_id = extract_job.id

        async with ctx.store.edit_state() as s:
            s.resume_segments = resume_segments

        return SplitCompletedEvent()

    @step()
    async def complete_extractions(
        self,
        event: SplitCompletedEvent,
        ctx: Context[ExtractionState],
        llama_cloud_client: Annotated[
            AsyncLlamaCloud, Resource(get_llama_cloud_client)
        ],
        extract_config: Annotated[
            ExtractConfig,
            ResourceConfig(
                config_file="configs/config.json",
                path_selector="extract",
                label="Resume Extraction",
                description="Fields to extract from each resume",
            ),
        ],
    ) -> StopEvent:
        """Wait for all extractions and save results."""
        state = await ctx.store.get_state()

        if not state.resume_segments:
            return StopEvent(result=[])

        # Remove old data for this file
        if state.file_hash is not None:
            delete_result = await llama_cloud_client.beta.agent_data.delete_by_query(
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
                filter={"file_hash": {"eq": state.file_hash}},
            )
            if delete_result.deleted_count > 0:
                logger.info(
                    f"Removed {delete_result.deleted_count} existing record(s) for file {state.filename}"
                )

        # Process all extraction results
        saved_ids: list[str] = []
        schema_class = get_extraction_schema(extract_config.json_schema)

        for i, segment in enumerate(state.resume_segments):
            # Wait for extraction to complete
            await llama_cloud_client.extraction.jobs.wait_for_completion(
                segment.extract_job_id
            )

            extracted_result = await llama_cloud_client.extraction.jobs.get_result(
                segment.extract_job_id
            )
            extract_run = await llama_cloud_client.extraction.runs.get(
                run_id=extracted_result.run_id
            )

            # Validate and create ExtractedData
            extracted_event: ExtractedEvent | ExtractedInvalidEvent
            try:
                logger.info(
                    f"Extracted resume {i + 1}: {json.dumps(extracted_result.model_dump(), indent=2)}"
                )
                data = ExtractedData.from_extraction_result(
                    result=extract_run,
                    schema=schema_class,
                    file_name=state.filename,
                    file_id=state.file_id,
                    file_hash=state.file_hash,
                )
                extracted_event = ExtractedEvent(data=data)
            except InvalidExtractionData as e:
                logger.error(f"Error validating extracted data: {e}", exc_info=True)
                extracted_event = ExtractedInvalidEvent(data=e.invalid_item)

            ctx.write_event_to_stream(extracted_event)

            # Save to Agent Data
            extracted_data = extracted_event.data
            item = await llama_cloud_client.beta.agent_data.agent_data(
                data=extracted_data.model_dump(),
                deployment_name=agent_name or "_public",
                collection=EXTRACTED_DATA_COLLECTION,
            )
            saved_ids.append(item.id)

            async with ctx.store.edit_state() as s:
                s.extracted_count = i + 1

        ctx.write_event_to_stream(
            Status(
                level="info",
                message=f"Extracted {len(saved_ids)} resume(s) from {state.filename}",
            )
        )

        return StopEvent(result=saved_ids)


workflow = ProcessFileWorkflow(timeout=None)

if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    async def main():
        file = await get_llama_cloud_client().files.create(
            file=Path("test.pdf").open("rb"),
            purpose="split",
        )
        await workflow.run(start_event=FileEvent(file_id=file.id))

    asyncio.run(main())
