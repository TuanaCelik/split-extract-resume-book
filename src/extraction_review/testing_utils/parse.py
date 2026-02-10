from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import httpx
from llama_cloud.types.parsing_create_response import ParsingCreateResponse
from llama_cloud.types.parsing_get_response import (
    Items,
    ItemsPage,
    ItemsPageStructuredResultPage,
    TextItem,
    Job,
    Markdown,
    MarkdownPage,
    MarkdownPageMarkdownResultPage,
    ParsingGetResponse,
    Text,
    TextPage,
)

from ._deterministic import generate_text_blob, hash_schema, utcnow

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


@dataclass
class ParseJobRecord:
    job_id: str
    file_name: str
    status: str
    result: Dict[str, Any]
    content: bytes


class FakeParseNamespace:
    def __init__(self, *, server: "FakeLlamaCloudServer") -> None:
        self._server = server
        self._jobs: Dict[str, ParsingGetResponse] = {}
        self.routes: Dict[str, Any] = {}
        self.allowed_expands = ("text", "markdown", "items")

    def register(self) -> None:
        server = self._server
        server.add_route(
            "POST",
            "/api/v2/parse/upload",
            self._handle_upload,
            namespace="parse",
        )
        server.add_route(
            "GET",
            "/api/v2/parse/{job_id}",
            self._handle_job_result,
            namespace="parse",
        )
        server.add_route(
            "POST",
            "/api/v2/parse",
            self._handle_file_id_source_url,
            namespace="parse",
        )

    def _handle_upload(self, request: httpx.Request) -> httpx.Response:
        _, filename, form_data = self._split_multipart(request)
        job_id = self._server.new_id("parse-job")
        seed_hash = hash_schema({"filename": filename, "form": form_data})
        seed = int(seed_hash[:16], 16)
        page_text = generate_text_blob(seed, sentences=3)
        item_pages: list[ItemsPage] = [
            ItemsPageStructuredResultPage(
                items=[TextItem(md=page_text, value=page_text, bBox=None, type="text")],
                page_height=1,
                page_number=1,
                page_width=1,
                success=True,
            )
        ]
        md_pages: list[MarkdownPage] = [
            MarkdownPageMarkdownResultPage(
                markdown=page_text,
                page_number=1,
                success=True,
            )
        ]
        txt_pages: list[TextPage] = [
            TextPage(
                text=page_text,
                page_number=1,
            )
        ]
        record = ParsingGetResponse(
            job=Job(
                id=job_id,
                status="COMPLETED",
                project_id=self._server.default_project_id,
                created_at=utcnow(),
                updated_at=utcnow(),
                error_message=None,
            ),
            items=Items(pages=item_pages),
            markdown=Markdown(pages=md_pages),
            text=Text(pages=txt_pages),
        )
        self._jobs[job_id] = record
        response = ParsingCreateResponse(
            id=job_id,
            project_id=self._server.default_project_id,
            status="COMPLETED",
            created_at=utcnow(),
            updated_at=utcnow(),
            error_message=None,
        )
        return self._server.json_response(response.model_dump())

    def _handle_file_id_source_url(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        file_id = payload.get("file_id")
        source_url = payload.get("source_url")
        if file_id is not None:
            file = self._server.files.get(file_id)
            if file is None:
                return self._server.json_response(
                    {"details": f"File {file_id} not found"},
                    status_code=404,
                )
            else:
                seed_hash = file.sha256
        elif source_url is not None:
            response = self._get_file_from_source_url(source_url)
            if isinstance(response, int):
                return self._server.json_response(
                    {"details": f"Could not find file associated with {source_url}"},
                    status_code=response,
                )
            file_content, filename = response
            file_id = self._server.files.preload_from_source(filename, file_content)
            seed_hash = self._server.files._files[file_id].sha256
        else:
            return self._server.json_response(
                {
                    "details": "At least one between file_id and source_url should be not-null",
                },
                status_code=400,
            )
        job_id = self._server.new_id("parse-job")
        seed = int(seed_hash[:16], 16)
        page_text = generate_text_blob(seed, sentences=3)
        item_pages: list[ItemsPage] = [
            ItemsPageStructuredResultPage(
                items=[TextItem(md=page_text, value=page_text, bBox=None, type="text")],
                page_height=1,
                page_number=1,
                page_width=1,
                success=True,
            )
        ]
        md_pages: list[MarkdownPage] = [
            MarkdownPageMarkdownResultPage(
                markdown=page_text,
                page_number=1,
                success=True,
            )
        ]
        txt_pages: list[TextPage] = [
            TextPage(
                text=page_text,
                page_number=1,
            )
        ]
        record = ParsingGetResponse(
            job=Job(
                id=job_id,
                status="COMPLETED",
                project_id=self._server.default_project_id,
                created_at=utcnow(),
                updated_at=utcnow(),
                error_message=None,
            ),
            items=Items(pages=item_pages),
            markdown=Markdown(pages=md_pages),
            text=Text(pages=txt_pages),
        )
        self._jobs[job_id] = record
        response = ParsingCreateResponse(
            id=job_id,
            project_id=self._server.default_project_id,
            status="COMPLETED",
            created_at=utcnow(),
            updated_at=utcnow(),
            error_message=None,
        )
        return self._server.json_response(response.model_dump())

    def _handle_job_result(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.split("/")[-1]
        expandees = request.url.params.get_list("expand")
        expandees = (
            [e for e in expandees if e in self.allowed_expands]
            if len(expandees) > 0
            else ["items"]
        )
        job_response = self._jobs.get(job_id)
        if not job_response:
            return self._server.json_response(
                {"detail": "Result not found"}, status_code=404
            )
        jb_resp_copy = deepcopy(job_response)
        if "markdown" not in expandees:
            jb_resp_copy.markdown = None
        if "text" not in expandees:
            jb_resp_copy.text = None
        if "items" not in expandees:
            jb_resp_copy.items = None
        return self._server.json_response(jb_resp_copy.model_dump())

    def _split_multipart(
        self, request: httpx.Request
    ) -> tuple[bytes, str, Dict[str, str]]:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart form data for parse upload")
        boundary = content_type.split("boundary=")[-1]
        delimiter = f"--{boundary}".encode()
        closing = f"--{boundary}--".encode()
        parts = []
        body = request.content
        for chunk in body.split(delimiter):
            chunk = chunk.strip()
            if not chunk or chunk == closing:
                continue
            parts.append(chunk)

        file_bytes = b""
        filename = "upload.pdf"
        form_data: Dict[str, str] = {}
        for part in parts:
            header_blob, _, payload = part.partition(b"\r\n\r\n")
            payload = payload.rstrip(b"\r\n")
            header_text = header_blob.decode("utf-8", errors="ignore")
            if "filename=" in header_text:
                # Extract filename from Content-Disposition header, handling quotes
                # and avoiding capturing subsequent headers or parameters
                match = re.search(r'filename="([^"]+)"', header_text)
                if not match:
                    match = re.search(r"filename='([^']+)'", header_text)
                if not match:
                    match = re.search(r"filename=([^\s;\r\n]+)", header_text)
                if match:
                    filename = match.group(1)
                file_bytes = payload
            else:
                name = header_text.split('name="')[-1].split('"')[0].strip()
                form_data[name] = payload.decode("utf-8", errors="ignore")
        if not file_bytes:
            raise ValueError("File part missing from multipart payload")
        return file_bytes, filename, form_data

    def _get_file_from_source_url(self, source_url: str) -> tuple[bytes, str] | int:
        name = source_url.split("/")[-1]
        with httpx.Client() as client:
            response = client.get(source_url, follow_redirects=True)
        if response.status_code >= 400:
            return response.status_code
        content = response.content
        return content, name
