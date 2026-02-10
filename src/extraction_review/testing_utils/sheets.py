from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from llama_cloud.types.beta.sheets_job import Region, SheetsJob, WorksheetMetadata
from llama_cloud.types.beta.sheets_parsing_config import SheetsParsingConfig
from llama_cloud.types.presigned_url import PresignedURL

from ._deterministic import combined_seed, generate_text_blob, utcnow
from .files import FakeFilesNamespace, StoredFile

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


@dataclass
class SheetsJobRecord:
    job: SheetsJob


class FakeSheetsNamespace:
    def __init__(
        self,
        *,
        server: "FakeLlamaCloudServer",
        files: FakeFilesNamespace,
        download_base_url: str,
    ) -> None:
        self._server = server
        self._files = files
        self._download_base_url = download_base_url.rstrip("/")
        self._jobs: Dict[str, SheetsJobRecord] = {}
        self._region_content: Dict[str, bytes] = {}

    def register(self) -> None:
        server = self._server
        server.add_route(
            "POST",
            "/api/v1/beta/sheets/jobs",
            self._handle_create,
            namespace="sheets",
        )
        server.add_route(
            "GET",
            "/api/v1/beta/sheets/jobs",
            self._handle_list,
            namespace="sheets",
        )
        server.add_route(
            "GET",
            "/api/v1/beta/sheets/jobs/{spreadsheet_job_id}",
            self._handle_get,
            namespace="sheets",
        )
        server.add_route(
            "DELETE",
            "/api/v1/beta/sheets/jobs/{spreadsheet_job_id}",
            self._handle_delete,
            namespace="sheets",
        )
        server.add_route(
            "GET",
            "/api/v1/beta/sheets/jobs/{spreadsheet_job_id}/regions/{region_id}/result/{region_type}",
            self._handle_get_result_table,
            namespace="sheets",
        )
        server.add_route(
            "GET",
            "/sheets/{job_id}/{region_id}/{region_type}",
            self._handle_presigned_download,
            namespace="sheets",
            base_urls=[self._download_base_url],
        )

    def _handle_create(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        file_id = payload.get("file_id", "")
        config_raw = payload.get("config") or {}

        stored_file = self._files.get(file_id) if file_id else None
        if file_id and not stored_file:
            return self._server.json_response(
                {"detail": f"File {file_id} not found"}, status_code=404
            )

        job_id = self._server.new_id("sheets-job")
        now = utcnow()

        config = SheetsParsingConfig.model_validate(config_raw)
        regions, worksheet_metadata = self._build_results(job_id, config, stored_file)

        job = SheetsJob(
            id=job_id,
            config=config,
            created_at=now.isoformat(),
            file_id=file_id,
            project_id=request.url.params.get(
                "project_id", self._server.default_project_id
            ),
            status="SUCCESS",
            updated_at=now.isoformat(),
            user_id="fake-user",
            errors=None,
            file=None,
            regions=regions,
            success=True,
            worksheet_metadata=worksheet_metadata,
        )
        self._jobs[job_id] = SheetsJobRecord(job=job)
        return self._server.json_response(job.model_dump())

    def _handle_list(self, request: httpx.Request) -> httpx.Response:
        items = [r.job.model_dump() for r in self._jobs.values()]
        return self._server.json_response({"items": items, "next_page_token": None})

    def _handle_get(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.split("/")[-1]
        record = self._jobs.get(job_id)
        if not record:
            return self._server.json_response(
                {"detail": "Sheets job not found"}, status_code=404
            )
        return self._server.json_response(record.job.model_dump())

    def _handle_delete(self, request: httpx.Request) -> httpx.Response:
        job_id = request.url.path.split("/")[-1]
        self._jobs.pop(job_id, None)
        return self._server.json_response({}, status_code=200)

    def _handle_get_result_table(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        # .../jobs/{job_id}/regions/{region_id}/result/{region_type}
        job_id = parts[-5]
        region_id = parts[-3]

        record = self._jobs.get(job_id)
        if not record:
            return self._server.json_response(
                {"detail": "Sheets job not found"}, status_code=404
            )

        if record.job.regions:
            found = any(r.region_id == region_id for r in record.job.regions)
            if not found:
                return self._server.json_response(
                    {"detail": f"Region {region_id} not found"}, status_code=404
                )

        region_type = parts[-1]
        presigned = PresignedURL(
            url=(
                f"{self._download_base_url}/sheets/{job_id}/{region_id}/{region_type}"
                f"?{urlencode({'token': 'fake'})}"
            ),
            expires_at=utcnow(),
            form_fields=None,
        )
        return self._server.json_response(presigned.model_dump())

    def _handle_presigned_download(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        # /sheets/{job_id}/{region_id}/{region_type}
        region_id = parts[-2]
        content_key = region_id
        content = self._region_content.get(content_key)
        if content is None:
            return httpx.Response(404, json={"detail": "Region content not found"})
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": "application/octet-stream"},
        )

    def _build_results(
        self,
        job_id: str,
        config: SheetsParsingConfig,
        stored_file: Optional[StoredFile],
    ) -> tuple[List[Region], List[WorksheetMetadata]]:
        file_hash = stored_file.sha256 if stored_file else "no-file"
        seed = combined_seed(file_hash, job_id)
        rng = random.Random(seed)

        sheet_names_config = config.sheet_names if config else None
        if sheet_names_config:
            sheet_names = list(sheet_names_config)
        else:
            num_sheets = rng.randint(1, 3)
            sheet_names = [f"Sheet{i + 1}" for i in range(num_sheets)]

        worksheet_metadata: List[WorksheetMetadata] = []
        for name in sheet_names:
            worksheet_metadata.append(
                WorksheetMetadata(
                    sheet_name=name,
                    title=f"Title for {name}",
                    description=f"Description for {name}",
                )
            )

        regions: List[Region] = []
        region_types = ["table", "extra"]
        for sheet_name in sheet_names:
            num_regions = rng.randint(1, 3)
            for j in range(num_regions):
                region_id = self._server.new_id("region")
                rtype = region_types[rng.randint(0, len(region_types) - 1)]
                row_start = rng.randint(1, 10)
                col_start = chr(ord("A") + rng.randint(0, 5))
                row_end = row_start + rng.randint(3, 20)
                col_end = chr(ord(col_start) + rng.randint(1, 5))
                location = f"{col_start}{row_start}:{col_end}{row_end}"
                regions.append(
                    Region(
                        region_id=region_id,
                        region_type=rtype,
                        sheet_name=sheet_name,
                        location=location,
                        title=f"Region {j + 1} in {sheet_name}",
                        description=f"Deterministic region from {sheet_name}",
                    )
                )
                content_seed = combined_seed(file_hash, job_id, region_id)
                self._region_content[region_id] = _build_fake_parquet(
                    content_seed, sheet_name, location
                )

        return regions, worksheet_metadata


def _build_fake_parquet(seed: int, sheet_name: str, location: str) -> bytes:
    """Build a minimal Apache Parquet file with deterministic tabular data.

    The file uses the PAR1 magic bytes and contains a simplified but
    structurally valid Parquet layout so that downstream consumers can
    at minimum verify the magic header.
    """
    rng = random.Random(seed)
    num_cols = rng.randint(2, 5)
    num_rows = rng.randint(3, 10)

    headers = [f"col_{i}" for i in range(num_cols)]
    rows: List[List[str]] = []
    for _ in range(num_rows):
        row = [
            generate_text_blob(rng.randint(0, 1_000_000), sentences=1)[:30]
            for _ in headers
        ]
        rows.append(row)

    # Encode as minimal Parquet-like binary with correct magic.
    # Real Parquet parsers need Thrift metadata; for the mock server we
    # embed the data as JSON in the page payload between the PAR1 markers
    # so tests can verify content determinism if needed.
    import json as _json

    payload = _json.dumps(
        {
            "sheet_name": sheet_name,
            "location": location,
            "headers": headers,
            "rows": rows,
        }
    ).encode("utf-8")

    magic = b"PAR1"
    # Parquet footer: 4-byte LE footer length + magic
    footer_len = struct.pack("<I", len(payload))
    return magic + payload + footer_len + magic
