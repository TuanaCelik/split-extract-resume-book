"""
<system>
This is the default test for the extract-basic template.
Adapt the test to your use case following the guidelines below.
</system>
<guidelines>
You can read and modify this file.
Here are your editing permissions, which you **MUST ALWAYS** follow:

- Lines and blocks tagged with `<edit></edit>` should be ALWAYS modified to something different, based on the use case.
- Lines and blocks tagged with `<adapt></adapt>` should be adapted to the specific use case you are dealing with, but only if needed.
- If something does not have tags, it **MUST NOT** be modified.
</guidelines>
"""

import json
import warnings
from pathlib import Path

import pytest
from extraction_review.clients import fake

# <edit>
from extraction_review.config import EXTRACTED_DATA_COLLECTION
from extraction_review.metadata_workflow import MetadataResponse
from extraction_review.metadata_workflow import workflow as metadata_workflow
from extraction_review.process_file import FileEvent
from extraction_review.process_file import workflow as process_file_workflow
from workflows.events import StartEvent

# </edit>


def get_extraction_schema() -> dict:
    """Load the extraction schema from the unified config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text())
    return config["extract"]["json_schema"]


@pytest.mark.asyncio
# <adapt>
async def test_process_file_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    # load a file to the mock LlamaCloud server and retrieve its file id (modify if you don't have any files to load as input)
    if fake is not None:
        file_id = fake.files.preload(path="tests/files/test.pdf")
    else:
        warnings.warn(
            "Skipping test because it cannot be mocked. Set `FAKE_LLAMA_CLOUD=true` in your environment to enable this test..."
        )
        return
    try:
        result = await process_file_workflow.run(start_event=FileEvent(file_id=file_id))
    except Exception:
        result = None
    assert result is not None
    # all generated agent data IDs are alphanumeric strings with 7 characters
    # the following assert statements ensure that that is the case
    assert isinstance(result, str)
    assert len(result) == 7


# </adapt>


# <adapt>
@pytest.mark.asyncio
async def test_metadata_workflow() -> None:
    result = await metadata_workflow.run(start_event=StartEvent())
    assert isinstance(result, MetadataResponse)
    assert result.extracted_data_collection == EXTRACTED_DATA_COLLECTION
    assert result.json_schema == get_extraction_schema()


# </adapt>
