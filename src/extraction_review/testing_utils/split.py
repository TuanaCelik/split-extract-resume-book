from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from llama_cloud.types.beta.split_category import SplitCategory
from llama_cloud.types.beta.split_category_param import SplitCategoryParam
from llama_cloud.types.beta.split_create_response import SplitCreateResponse
from llama_cloud.types.beta.split_document_input import SplitDocumentInput
from llama_cloud.types.beta.split_get_response import SplitGetResponse
from llama_cloud.types.beta.split_result_response import SplitResultResponse
from llama_cloud.types.beta.split_segment_response import SplitSegmentResponse
from pydantic.dataclasses import dataclass

from ._deterministic import categorize_pages, utcnow
from .files import StoredFile

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


@dataclass
class SplitRequest:
    categories: list[SplitCategoryParam]
    file_id: str
    stored_file: StoredFile


class FakeSplitNamespace:
    def __init__(self, *, server: "FakeLlamaCloudServer") -> None:
        self._server = server
        self._jobs: dict[str, SplitGetResponse] = {}
        self.routes: dict[str, Any] = {}
        self._allowed_input_types = ("file_id",)
        self._page_size = 50

    def _validate_split_request(
        self, request: httpx.Request
    ) -> httpx.Response | SplitRequest:
        payload = self._server.json(request)
        document_input = payload.get("document_input")
        if not document_input:
            response = {"detail": "the document_input field should be non-null"}
            return self._server.json_response(response, status_code=400)
        input_type = document_input.get("type", "file_id")
        if input_type not in self._allowed_input_types:
            response = {
                "detail": f"document_input.type {input_type} is invalid. Allowed input types: {', '.join(self._allowed_input_types)}"
            }
            return self._server.json_response(response, status_code=400)
        input_value = document_input.get("value")
        if input_value is None:
            response = {"detail": "Missing document_input.value field"}
            return self._server.json_response(response, status_code=400)
        categories = payload.get("categories", [])
        if not categories:
            response = {"detail": "categories field should be non-null and non-empty"}
            return self._server.json_response(response, status_code=400)
        stored_file = self._server.files.get(input_value)
        if stored_file is None:
            response = {"detail": f"file with ID {input_value} not found"}
            return self._server.json_response(response, status_code=404)
        return SplitRequest(
            categories=categories, file_id=input_value, stored_file=stored_file
        )

    def _create_split_job(self, request: httpx.Request) -> httpx.Response:
        validated = self._validate_split_request(request)
        if isinstance(validated, httpx.Response):
            return validated
        categorized = categorize_pages(
            validated.stored_file.content,
            [category["name"] for category in validated.categories],
            0,
        )
        result = SplitResultResponse(segments=[])
        for c in categorized:
            result.segments.append(
                SplitSegmentResponse(
                    category=c, confidence_category="high", pages=categorized[c]
                )
            )
        job_id = self._server.new_id("split-")
        job = SplitGetResponse(
            id=job_id,
            categories=[
                SplitCategory(name=c["name"], description=c.get("description"))
                for c in validated.categories
            ],
            document_input=SplitDocumentInput(type="file_id", value=validated.file_id),
            project_id=self._server.default_project_id,
            user_id=self._server.default_user_id,
            status="completed",
            result=result,
            created_at=utcnow(),
            updated_at=utcnow(),
            error_message=None,
        )
        self._jobs[job_id] = job
        response = SplitCreateResponse(
            id=job_id,
            categories=[
                SplitCategory(name=c["name"], description=c.get("description"))
                for c in validated.categories
            ],
            document_input=SplitDocumentInput(type="file_id", value=validated.file_id),
            project_id=self._server.default_project_id,
            user_id=self._server.default_user_id,
            status="pending",
            error_message=None,
        )
        return self._server.json_response(response.model_dump(), status_code=200)

    def _get_split_job_result(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.split("/")[-1]
        job = self._jobs.get(job_id)
        if job is not None:
            return self._server.json_response(job.model_dump())
        return self._server.json_response(
            {"detail": f"job with ID {job_id} does not exist"}, status_code=404
        )

    def register(self) -> None:
        server = self._server
        create_route = server.add_route(
            "POST",
            "/api/v1/beta/split/jobs",
            self._create_split_job,
            namespace="split",
            alias="create",
        )
        self.routes["create"] = create_route
        get_route = server.add_route(
            "GET",
            "/api/v1/beta/split/jobs/{split_job_id}",
            self._get_split_job_result,
            namespace="split",
            alias="get",
        )
        self.routes["get"] = get_route
