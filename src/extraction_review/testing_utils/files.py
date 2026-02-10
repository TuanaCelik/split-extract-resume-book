from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
import respx
from llama_cloud.types import File as CloudFile
from llama_cloud.types.file_list_response import FileListResponse
from llama_cloud.types.file_query_response import FileQueryResponse, Item
from llama_cloud.types.presigned_url import PresignedURL

from ._deterministic import (
    fingerprint_file,
    utcnow,
)
from .matchers import RequestMatcher

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


@dataclass
class StoredFile:
    file: CloudFile
    content: bytes
    sha256: str


@dataclass
class PendingUpload:
    file_id: str
    filename: str
    project_id: str
    organization_id: str
    external_file_id: Optional[str]
    expected_size: Optional[int]


class FakeFilesNamespace:
    def __init__(
        self,
        *,
        server: "FakeLlamaCloudServer",
        upload_base_url: str,
        download_base_url: str,
    ) -> None:
        self._server = server
        self._upload_base_url = upload_base_url.rstrip("/")
        self._download_base_url = download_base_url.rstrip("/")
        self._files: Dict[str, StoredFile] = {}
        self._pending: Dict[str, PendingUpload] = {}
        self._upload_stubs: List[
            tuple[RequestMatcher | None, int, Dict[str, Any], bool]
        ] = []
        self.routes: Dict[str, respx.Route] = {}

    # Public helpers -------------------------------------------------
    def preload(self, *, path: str | Path, filename: Optional[str] = None) -> str:
        path = Path(path)
        content = path.read_bytes()
        file_id = self._server.new_id("file")
        name = filename or path.name
        stored = self._build_file(
            file_id=file_id,
            name=name,
            project_id=self._server.default_project_id,
            organization_id=self._server.default_organization_id,
            content=content,
            external_file_id=None,
        )
        self._files[file_id] = stored
        return file_id

    def read(self, file_id: str) -> bytes:
        return self._files[file_id].content

    def get(self, file_id: str) -> Optional[StoredFile]:
        return self._files.get(file_id)

    def preload_from_source(self, filename: str, content: bytes) -> str:
        file_id = self._server.new_id("file")
        name = filename
        stored = self._build_file(
            file_id=file_id,
            name=name,
            project_id=self._server.default_project_id,
            organization_id=self._server.default_organization_id,
            content=content,
            external_file_id=None,
        )
        self._files[file_id] = stored
        return file_id

    def stub_upload(
        self,
        matcher: Optional[RequestMatcher],
        *,
        status_code: int = 413,
        json_body: Optional[Dict[str, Any]] = None,
        once: bool = True,
    ) -> None:
        body = json_body or {"detail": "upload rejected by fake server"}
        self._upload_stubs.append((matcher, status_code, body, once))

    def all_files(self) -> Dict[str, StoredFile]:
        return dict(self._files)

    # Route registration ---------------------------------------------
    def register(self) -> None:
        server = self._server
        upload_route = server.add_route(
            "POST",
            "/api/v1/beta/files",
            self._handle_direct_upload,
            namespace="files",
            alias="upload",
        )
        self.routes["upload"] = upload_route
        list_route = server.add_route(
            "GET",
            "/api/v1/beta/files",
            self._handle_list,
            namespace="files",
            alias="list_files",
        )
        self.routes["list"] = list_route
        get_route = server.add_route(
            "GET",
            "/api/v1/beta/files/{file_id}/content",
            self._handle_read_content,
            namespace="files",
            alias="get",
        )
        self.routes["get"] = get_route
        server.add_route(
            "DELETE",
            "/api/v1/beta/files/{file_id}",
            self._handle_delete,
            namespace="files",
        )
        server.add_route(
            "POST",
            "/api/v1/beta/files/query",
            self._handle_query,
            namespace="files",
            alias="query",
        )
        server.add_route(
            "GET",
            "/files/{file_id}",
            self._handle_presigned_download,
            namespace="files",
            base_urls=[self._download_base_url],
            alias="download",
        )

    # Handlers -------------------------------------------------------
    def _handle_direct_upload(self, request: httpx.Request) -> httpx.Response:
        file_bytes, filename = self._extract_multipart_file(request)
        file_id = self._server.new_id("file")
        stored = self._build_file(
            file_id=file_id,
            name=filename or f"upload-{file_id}.bin",
            project_id=request.url.params.get(
                "project_id", self._server.default_project_id
            ),
            organization_id=request.url.params.get(
                "organization_id", self._server.default_organization_id
            ),
            content=file_bytes,
            external_file_id=request.url.params.get("external_file_id"),
        )
        self._files[file_id] = stored
        return self._server.json_response(stored.file.model_dump())

    def _handle_list(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        file_ids_raw = params.multi_items()
        file_ids_filter = [v for k, v in file_ids_raw if k == "file_ids"]
        file_name = params.get("file_name")
        external_file_id = params.get("external_file_id")
        page_size = int(params.get("page_size", "50"))

        files = list(self._files.values())
        if file_ids_filter:
            files = [f for f in files if f.file.id in file_ids_filter]
        if file_name:
            files = [f for f in files if f.file.name == file_name]
        if external_file_id:
            files = [f for f in files if f.file.external_file_id == external_file_id]

        files = files[:page_size]
        items = [
            FileListResponse(
                id=f.file.id,
                name=f.file.name,
                project_id=f.file.project_id,
                expires_at=f.file.expires_at,
                external_file_id=f.file.external_file_id,
                file_type=f.file.file_type,
                last_modified_at=f.file.last_modified_at,
                purpose=f.file.purpose,
            )
            for f in files
        ]
        return self._server.json_response(
            {
                "items": [item.model_dump() for item in items],
                "next_page_token": None,
            }
        )

    def _handle_delete(self, request: httpx.Request) -> httpx.Response:
        file_id = request.url.path.split("/")[-1]
        self._files.pop(file_id, None)
        self._pending.pop(file_id, None)
        return self._server.json_response({}, status_code=200)

    def _handle_read_content(self, request: httpx.Request) -> httpx.Response:
        file_id = request.url.path.split("/")[-2]
        if file_id not in self._files:
            return self._server.json_response(
                {"detail": "File not found"}, status_code=404
            )
        presigned = PresignedURL(
            url=f"{self._download_base_url}/files/{file_id}?{urlencode({'token': 'fake'})}",
            expires_at=utcnow(),
            form_fields=None,
        )
        return self._server.json_response(presigned.model_dump())

    def _handle_presigned_download(self, request: httpx.Request) -> httpx.Response:
        file_id = request.url.path.split("/")[-1]
        stored = self._files.get(file_id)
        if not stored:
            return httpx.Response(404, json={"detail": "File not found"})
        return httpx.Response(200, content=stored.content)

    def _handle_query(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        files: list[StoredFile] = []
        items: list[Item] = []
        if payload.get("filter") is not None:
            file_ids = payload["filter"].get("file_ids", [])
            for file_id in self._files:
                if file_id in file_ids:
                    files.append(self._files[file_id])
        else:
            files = list(self._files.values())
        for f in files:
            item = Item(
                id=f.file.id,
                name=f.file.name,
                project_id=self._server.default_project_id,
                expires_at=utcnow(),
                external_file_id=f.file.external_file_id,
                purpose=f.file.purpose,
                last_modified_at=utcnow(),
                file_type=f.file.file_type,
            )
            items.append(item)
        response = FileQueryResponse(
            items=items, next_page_token=None, total_size=len(items)
        )
        return self._server.json_response(response.model_dump())

    # Internal helpers -----------------------------------------------
    def _build_file(
        self,
        *,
        file_id: str,
        name: str,
        project_id: str,
        organization_id: str,
        content: bytes,
        external_file_id: Optional[str],
    ) -> StoredFile:
        sha256 = fingerprint_file(content, name)
        now = utcnow()
        cloud_file = CloudFile(
            id=file_id,
            name=name,
            project_id=project_id,
            external_file_id=external_file_id,
            file_size=len(content),
            file_type=Path(name).suffix or "application/octet-stream",
            created_at=now,
            updated_at=now,
            data_source_id=None,
            permission_info=None,
            resource_info=None,
            last_modified_at=now,
        )
        return StoredFile(file=cloud_file, content=content, sha256=sha256)

    def _extract_multipart_file(
        self, request: httpx.Request
    ) -> tuple[bytes, Optional[str]]:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart upload")

        boundary = content_type.split("boundary=")[-1]
        boundary_bytes = boundary.encode("utf-8")
        body = request.content
        delimiter = b"--" + boundary_bytes
        parts = [
            part
            for part in body.split(delimiter)
            if part.strip(b"\r\n") and part.strip(b"\r\n") != b"--"
        ]
        for part in parts:
            headers, _, payload = part.partition(b"\r\n\r\n")
            header_text = headers.decode("utf-8", errors="ignore")
            if 'name="upload_file"' in header_text or 'name="file"' in header_text:
                filename = None
                if "filename=" in header_text:
                    filename = (
                        header_text.split("filename=")[-1].strip().strip('"').strip("'")
                    )
                return payload.rstrip(b"\r\n"), filename
        raise ValueError("upload file part not found")

    def decode_file_data(self, data: Dict[str, Any]) -> tuple[bytes, Optional[str]]:
        if "file" not in data:
            raise ValueError("file payload missing")
        file_payload = data["file"]
        encoded = file_payload["data"]
        content = base64.b64decode(encoded)
        filename = file_payload.get("filename")
        return content, filename
