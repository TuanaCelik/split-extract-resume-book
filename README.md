# Data Extraction and Ingestion

This is a starter for LlamaAgents. See the [LlamaAgents (llamactl) getting started guide](https://developers.llamaindex.ai/python/llamaagents/llamactl/getting-started/) for context on local development and deployment.

To run the application, install [`uv`](https://docs.astral.sh/uv/) and run `uvx llamactl serve`.

## Simple customizations

For some basic customizations, you can modify `src/extraction_review/config.py`

- **`EXTRACTION_AGENT_NAME`**: Logical name for your Extraction Agent. When `USE_REMOTE_EXTRACTION_SCHEMA` is `False`, this name is used to upsert the agent with your local schema; when `True`, it is used to fetch an existing agent.
- **`EXTRACTED_DATA_COLLECTION`**: The Agent Data collection name used to store extractions (namespaced by agent name and environment).
- **`ExtractionSchema`**: When using a local schema, edit this Pydantic model to match the fields you want extracted. Prefer optional types where possible to allow for partial extractions. Note that the extraction process requires all values! so you must explicitly set values to be optional if they are not required. (pydantic default factories will not work, as pydantic only uses default values for missing fields).

The UI fetches the JSON Schema and collection name from the backend metadata workflow at runtime, and dynamically
generates an editing UI based on the schema. If you customize this application to have a different extraction schema from
the presentation schema rendered in the UI, for example if you customize the extraction process to add additional fields or otherwise
transforma it, then you must return the presentation schema from the metadata workflow.

## Complex customizations

For more complex customizations, you can edit the rest of the application. For example, you could
- Modify the existing file processing workflow to provide additional context for the extraction process
- Take further action based on the extracted data.
- Add additional workflows to submit data upon approval.

## Linting and type checking

Python and javascript packages contain helpful scripts to lint, format, and type check the code.

To check and fix python code:

```bash
uv run hatch run lint
uv run hatch run typecheck
uv run hatch run test
# run all at once
uv run hatch run all-fix
```

To check and fix javascript code, within the `ui` directory:

```bash
pnpm run lint
pnpm run typecheck
pnpm run test
# run all at once
pnpm run all-fix
```