from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

import httpx
from llama_cloud.resources.extraction.runs import AsyncPaginatedExtractRuns
from llama_cloud.types import (
    File as CloudFile,
)
from llama_cloud.types.extraction.extract_agent import ExtractAgent
from llama_cloud.types.extraction.extract_config import ExtractConfig
from llama_cloud.types.extraction.extract_job import ExtractJob
from llama_cloud.types.extraction.extract_run import ExtractRun
from llama_cloud.types.extraction.extraction_agent_list_response import (
    ExtractionAgentListResponse,
)
from llama_cloud.types.extraction.job_get_result_response import JobGetResultResponse
from llama_cloud.types.status_enum import StatusEnum

from ._deterministic import (
    combined_seed,
    fingerprint_file,
    generate_data_from_schema,
    hash_schema,
    utcnow,
)
from .files import FakeFilesNamespace, StoredFile
from .matchers import RequestContext, RequestMatcher

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


@dataclass
class ExtractRunStub:
    matcher: Optional[RequestMatcher]
    data: Optional[Any]
    status: Optional[str]
    metadata: Optional[Dict[str, Any]]
    error: Optional[str]
    job_status: Optional[str]
    once: bool


@dataclass
class AgentRunStub:
    agent_id: str
    matcher: Optional[RequestMatcher]
    job_status: Optional[str]
    run_status: Optional[str]
    error: Optional[str]
    once: bool


@dataclass
class StoredRun:
    job: ExtractJob
    run: ExtractRun


class FakeExtractNamespace:
    def __init__(
        self,
        *,
        server: "FakeLlamaCloudServer",
        files: FakeFilesNamespace,
    ) -> None:
        self._server = server
        self._files = files
        self._jobs: Dict[str, StoredRun] = {}
        self._runs: Dict[str, ExtractRun] = {}
        self._agents: Dict[str, ExtractAgent] = {}
        self._agents_by_name: Dict[str, str] = {}
        self._run_stubs: List[ExtractRunStub] = []
        self._agent_run_stubs: List[AgentRunStub] = []
        self.routes: Dict[str, Any] = {}

    # Public APIs ----------------------------------------------------
    def stub_run(
        self,
        matcher: Optional[RequestMatcher],
        *,
        data: Optional[Any] = None,
        status: Optional[str] = None,
        job_status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        once: bool = True,
    ) -> None:
        self._run_stubs.append(
            ExtractRunStub(
                matcher=matcher,
                data=data,
                status=status,
                metadata=metadata,
                error=error,
                job_status=job_status,
                once=once,
            )
        )

    def stub_agent_run(
        self,
        *,
        agent_id: str,
        matcher: Optional[RequestMatcher],
        job_status: Optional[str] = None,
        run_status: Optional[str] = None,
        error: Optional[str] = None,
        once: bool = True,
    ) -> None:
        self._agent_run_stubs.append(
            AgentRunStub(
                agent_id=agent_id,
                matcher=matcher,
                job_status=job_status,
                run_status=run_status,
                error=error,
                once=once,
            )
        )

    # Route registration ---------------------------------------------
    def register(self) -> None:
        server = self._server
        route = server.add_route(
            "POST",
            "/api/v1/extraction/run",
            self._handle_stateless_run,
            namespace="extract",
            alias="extract_run",
        )
        self.routes["stateless_run"] = route
        self.stateless_run = route
        server.add_route(
            "POST",
            "/api/v1/extraction/extraction-agents",
            self._handle_create_agent,
            namespace="extract",
        )
        server.add_route(
            "PATCH",
            "/api/v1/extraction/extraction-agents/{agent_id}",
            self._handle_update_agent,
            namespace="extract",
        )
        server.add_route(
            "PUT",
            "/api/v1/extraction/extraction-agents/{agent_id}",
            self._handle_update_agent,
            namespace="extract",
        )
        server.add_route(
            "GET",
            "/api/v1/extraction/extraction-agents/{agent_id}",
            self._handle_get_agent,
            namespace="extract",
        )
        server.add_route(
            "GET",
            "/api/v1/extraction/extraction-agents",
            self._handle_list_agents,
            namespace="extract",
        )
        server.add_route(
            "DELETE",
            "/api/v1/extraction/extraction-agents/{agent_id}",
            self._handle_delete_agent,
            namespace="extract",
        )
        server.add_route(
            "POST",
            "/api/v1/extraction/extraction-agents/schema/validation",
            self._handle_validate_schema,
            namespace="extract",
        )
        agent_job_route = server.add_route(
            "POST",
            "/api/v1/extraction/jobs",
            self._handle_agent_job,
            namespace="extract",
            alias="agent_job",
        )
        self.routes["agent_job"] = agent_job_route
        self.agent_job = agent_job_route
        server.add_route(
            "GET",
            "/api/v1/extraction/jobs",
            self._handle_list_jobs,
            namespace="extract",
        )
        server.add_route(
            "GET",
            "/api/v1/extraction/jobs/{job_id}",
            self._handle_get_job,
            namespace="extract",
        )
        server.add_route(
            "GET",
            "/api/v1/extraction/jobs/{job_id}/result",
            self._handle_get_job_result,
            namespace="extract",
        )
        agent_run_route = server.add_route(
            "GET",
            "/api/v1/extraction/runs/by-job/{job_id}",
            self._handle_get_run_by_job,
            namespace="extract",
            alias="agent_run",
        )
        self.routes["agent_run"] = agent_run_route
        self.agent_run = agent_run_route
        server.add_route(
            "GET",
            "/api/v1/extraction/runs/{run_id}",
            self._handle_get_run,
            namespace="extract",
        )
        server.add_route(
            "DELETE",
            "/api/v1/extraction/runs/{run_id}",
            self._handle_delete_run,
            namespace="extract",
        )
        server.add_route(
            "GET",
            "/api/v1/extraction/runs",
            self._handle_list_runs,
            namespace="extract",
        )

    # Handlers -------------------------------------------------------
    def _handle_stateless_run(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        config = ExtractConfig.model_validate(payload["config"])
        data_schema = payload["data_schema"]
        schema_hash = hash_schema(data_schema)

        file_info = self._extract_file_info(payload, request)
        agent = self._build_ephemeral_agent(
            config, data_schema, file_info.file.project_id
        )

        context = RequestContext(
            request=request,
            json=payload,
            file_id=file_info.file.id,
            filename=file_info.file.name,
            file_sha256=file_info.sha256,
            schema_hash=schema_hash,
            project_id=file_info.file.project_id,
            organization_id=self._server.default_organization_id,
        )

        stub = self._pop_stub(self._run_stubs, context)
        job_status: StatusEnum = "SUCCESS"
        run_status = "SUCCESS"
        metadata = {"deterministic": {"value": True}}
        error = None
        run_data = self._generate_run_data(data_schema, file_info.sha256)

        if stub:
            if stub.job_status:
                job_status = cast(StatusEnum, stub.job_status)
            if stub.status:
                run_status = stub.status
            if stub.metadata:
                metadata = stub.metadata
            if stub.error:
                error = stub.error
            if stub.data is not None:
                if callable(stub.data):
                    run_data = stub.data(payload)  # type: ignore[assignment]
                else:
                    run_data = stub.data

        stored = self._create_job_and_run(
            agent=agent,
            config=config,
            data_schema=data_schema,
            file_info=file_info,
            job_status=job_status,
            run_status=cast(StatusEnum, run_status),
            metadata=metadata,
            data=run_data,
            error=error,
            project_id=file_info.file.project_id,
        )
        return self._server.json_response(stored.job.model_dump())

    def _handle_create_agent(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        name = payload["name"]
        config = ExtractConfig.model_validate(payload["config"])
        data_schema = payload["data_schema"]
        agent_id = self._server.new_id("agent")
        agent = ExtractAgent(
            id=agent_id,
            name=name,
            config=config,
            data_schema=data_schema,
            project_id=request.url.params.get(
                "project_id", self._server.default_project_id
            ),
            created_at=utcnow(),
            updated_at=utcnow(),
            custom_configuration=None,
        )
        self._agents[agent_id] = agent
        self._agents_by_name[name] = agent_id
        return self._server.json_response(agent.model_dump())

    def _handle_update_agent(self, request: httpx.Request) -> httpx.Response:
        agent_id = request.url.path.split("/")[-1]
        if agent_id not in self._agents:
            return self._server.json_response(
                {"detail": "Agent not found"}, status_code=404
            )
        payload = self._server.json(request)
        agent = self._agents[agent_id]
        config = payload.get("config", agent.config)
        data_schema = payload.get("data_schema", agent.data_schema)
        updated = agent.model_copy(
            update={
                "config": ExtractConfig.model_validate(config)
                if isinstance(config, dict)
                else config,
                "data_schema": data_schema,
                "updated_at": utcnow(),
            }
        )
        self._agents[agent_id] = updated
        return self._server.json_response(updated.model_dump())

    def _handle_get_agent(self, request: httpx.Request) -> httpx.Response:
        agent_id = request.url.path.split("/")[-1]
        agent = self._agents.get(agent_id)
        if not agent:
            return self._server.json_response(
                {"detail": "Agent not found"}, status_code=404
            )
        return self._server.json_response(agent.model_dump())

    def _handle_list_agents(self, request: httpx.Request) -> httpx.Response:
        include_default = (
            request.url.params.get("include_default", "false").lower() == "true"
        )
        agents: ExtractionAgentListResponse = list(self._agents.values())
        if include_default and not agents:
            default_agent = self._build_ephemeral_agent(
                ExtractConfig(),
                {"type": "object", "properties": {}},
                self._server.default_project_id,
            )
            agents.append(default_agent)
        return self._server.json_response([agent.model_dump() for agent in agents])

    def _handle_delete_agent(self, request: httpx.Request) -> httpx.Response:
        agent_id = request.url.path.split("/")[-1]
        agent = self._agents.pop(agent_id, None)
        if agent:
            self._agents_by_name.pop(agent.name, None)
        return self._server.json_response({}, status_code=200)

    def _handle_validate_schema(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        return self._server.json_response({"data_schema": payload["data_schema"]})

    def _handle_agent_job(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        agent_id = payload["extraction_agent_id"]
        agent = self._agents.get(agent_id)
        if not agent:
            return self._server.json_response(
                {"detail": "Agent not found"}, status_code=404
            )

        file_id = payload["file_id"]
        stored_file = self._files.get(file_id)
        if not stored_file:
            return self._server.json_response(
                {"detail": "File not found"}, status_code=404
            )

        schema = payload.get("data_schema_override", agent.data_schema)
        config_payload = payload.get("config_override", agent.config)
        config = (
            ExtractConfig.model_validate(config_payload)
            if isinstance(config_payload, dict)
            else config_payload
        )

        stub = self._pop_agent_stub(
            agent_id, RequestContext(request=request, json=payload)
        )
        job_status: StatusEnum = "SUCCESS"
        run_status = "SUCCESS"
        error = None
        if stub:
            if stub.job_status:
                job_status = cast(StatusEnum, stub.job_status)
            if stub.run_status:
                run_status = stub.run_status
            if stub.error:
                error = stub.error

        stored = self._create_job_and_run(
            agent=agent,
            config=config,
            data_schema=schema,
            file_info=stored_file,
            job_status=job_status,
            run_status=cast(StatusEnum, run_status),
            metadata={"agent": {"value": agent.id}},
            data=self._generate_run_data(schema, stored_file.sha256),
            error=error,
            project_id=agent.project_id,
        )
        return self._server.json_response(stored.job.model_dump())

    def _handle_get_job_result(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.split("/")[-2]
        if job_id not in self._jobs:
            return self._server.json_response(
                {"detail": f"Job {job_id} not found"}, status_code=404
            )
        job = self._jobs[job_id]
        response = JobGetResultResponse(
            data=job.run.data,
            extraction_agent_id=job.run.extraction_agent_id,
            extraction_metadata=job.run.extraction_metadata or {},
            run_id=job.run.id,
        )
        return self._server.json_response(response.model_dump())

    def _handle_list_jobs(self, request: httpx.Request) -> httpx.Response:
        agent_id = request.url.params.get("extraction_agent_id")
        items = []
        for stored in self._jobs.values():
            if agent_id and stored.job.extraction_agent.id != agent_id:
                continue
            items.append(stored.job.model_dump())
        return self._server.json_response(items)

    def _handle_get_job(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.split("/")[-1]
        stored = self._jobs.get(job_id)
        if not stored:
            return self._server.json_response(
                {"detail": "Job not found"}, status_code=404
            )
        return self._server.json_response(stored.job.model_dump())

    def _handle_get_run_by_job(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.split("/")[-1]
        stored = self._jobs.get(job_id)
        if not stored:
            return self._server.json_response(
                {"detail": "Run not found"}, status_code=404
            )
        return self._server.json_response(stored.run.model_dump())

    def _handle_get_run(self, request: httpx.Request) -> httpx.Response:
        run_id = request.url.path.split("/")[-1]
        run = self._runs.get(run_id)
        if not run:
            return self._server.json_response(
                {"detail": "Run not found"}, status_code=404
            )
        return self._server.json_response(run.model_dump())

    def _handle_delete_run(self, request: httpx.Request) -> httpx.Response:
        run_id = request.url.path.split("/")[-1]
        self._runs.pop(run_id, None)
        to_delete = [
            job_id for job_id, stored in self._jobs.items() if stored.run.id == run_id
        ]
        for job_id in to_delete:
            self._jobs.pop(job_id, None)
        return self._server.json_response({}, status_code=200)

    def _handle_list_runs(self, request: httpx.Request) -> httpx.Response:
        agent_id = request.url.params.get("extraction_agent_id")
        skip = int(request.url.params.get("skip", "0"))
        filtered = [
            stored.run
            for stored in self._jobs.values()
            if not agent_id or stored.job.extraction_agent.id == agent_id
        ]
        page = filtered[skip:]
        response = AsyncPaginatedExtractRuns[ExtractRun](
            items=page,
            skip=skip,
            total=len(filtered),
        )
        return self._server.json_response(response.model_dump())

    # Internal helpers -----------------------------------------------
    def _extract_file_info(
        self, payload: Dict[str, Any], request: httpx.Request
    ) -> StoredFile:
        if "file_id" in payload:
            file_id = payload["file_id"]
            stored = self._files.get(file_id)
            if not stored:
                raise ValueError("file_id not found in fake store")
            return stored
        if "file" in payload:
            content, filename = self._files.decode_file_data(payload)
            file_id = self._server.new_id("file")
            stored = StoredFile(
                file=CloudFile(
                    id=file_id,
                    name=filename or f"inline-{file_id}",
                    project_id=request.url.params.get(
                        "project_id", self._server.default_project_id
                    ),
                    external_file_id=None,
                    file_size=len(content),
                    file_type=None,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                    data_source_id=None,
                    permission_info=None,
                    resource_info=None,
                    last_modified_at=utcnow(),
                ),
                content=content,
                sha256=fingerprint_file(content, filename),
            )
            return stored
        if "text" in payload:
            text_bytes = payload["text"].encode("utf-8")
            file_id = self._server.new_id("file")
            stored = StoredFile(
                file=CloudFile(
                    id=file_id,
                    name=f"text-{file_id}.txt",
                    project_id=self._server.default_project_id,
                    external_file_id=None,
                    file_size=len(text_bytes),
                    file_type="text/plain",
                    created_at=utcnow(),
                    updated_at=utcnow(),
                    data_source_id=None,
                    permission_info=None,
                    resource_info=None,
                    last_modified_at=utcnow(),
                ),
                content=text_bytes,
                sha256=fingerprint_file(text_bytes, None),
            )
            return stored
        raise ValueError("file payload missing")

    def _build_ephemeral_agent(
        self,
        config: ExtractConfig,
        data_schema: Dict[str, Any],
        project_id: str,
    ) -> ExtractAgent:
        return ExtractAgent(
            id=self._server.new_id("agent"),
            name="stateless-agent",
            config=config,
            data_schema=data_schema,
            project_id=project_id,
            created_at=utcnow(),
            updated_at=utcnow(),
            custom_configuration=None,
        )

    def _generate_run_data(self, schema: Dict[str, Any], file_hash: str) -> Any:
        seed = combined_seed(file_hash, hash_schema(schema))
        return generate_data_from_schema(schema, seed)

    def _create_job_and_run(
        self,
        *,
        agent: ExtractAgent,
        config: ExtractConfig,
        data_schema: Dict[str, Any],
        file_info: StoredFile,
        job_status: StatusEnum,
        run_status: StatusEnum,
        metadata: Dict[str, Any],
        data: Any,
        error: Optional[str],
        project_id: str,
    ) -> StoredRun:
        job_id = self._server.new_id("job")
        run_id = self._server.new_id("run")
        now = utcnow()

        job = ExtractJob(
            id=job_id,
            file=file_info.file,
            extraction_agent=agent,
            status=job_status,
            error=error,
        )
        run = ExtractRun(
            id=run_id,
            job_id=job_id,
            file=file_info.file,
            extraction_agent_id=agent.id,
            status=cast(Literal["CREATED", "PENDING", "SUCCESS", "ERROR"], run_status),
            config=config,
            data_schema=data_schema,
            data=data,
            extraction_metadata=metadata,
            created_at=now,
            updated_at=now,
            from_ui=False,
            error=error,
            project_id=project_id,
        )
        stored = StoredRun(job=job, run=run)
        self._jobs[job_id] = stored
        self._runs[run_id] = run
        return stored

    def _pop_stub(
        self,
        stubs: List[ExtractRunStub],
        context: RequestContext,
    ) -> Optional[ExtractRunStub]:
        for index, stub in enumerate(list(stubs)):
            if context.matches(stub.matcher):
                if stub.once:
                    stubs.pop(index)
                return stub
        return None

    def _pop_agent_stub(
        self,
        agent_id: str,
        context: RequestContext,
    ) -> Optional[AgentRunStub]:
        for index, stub in enumerate(list(self._agent_run_stubs)):
            if stub.agent_id != agent_id:
                continue
            if context.matches(stub.matcher):
                if stub.once:
                    self._agent_run_stubs.pop(index)
                return stub
        return None
