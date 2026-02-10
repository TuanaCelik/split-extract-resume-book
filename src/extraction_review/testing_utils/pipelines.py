from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from llama_cloud.types.managed_ingestion_status_response import (
    ManagedIngestionStatusResponse,
)
from llama_cloud.types.pipeline import Pipeline
from llama_cloud.types.pipeline_retrieve_response import (
    PipelineRetrieveResponse,
    RetrievalNode,
)
from llama_cloud.types.pipelines.cloud_document import CloudDocument
from llama_cloud.types.pipelines.pipeline_file import PipelineFile
from llama_cloud.types.pipelines.text_node import TextNode

from ._deterministic import combined_seed, generate_text_blob, utcnow

if TYPE_CHECKING:
    from .server import FakeLlamaCloudServer


class FakePipelinesNamespace:
    def __init__(self, *, server: "FakeLlamaCloudServer") -> None:
        self._server = server
        self._pipelines: Dict[str, Pipeline] = {}
        # Per-pipeline storage for ingested documents and files
        self._documents: Dict[str, Dict[str, CloudDocument]] = {}
        self._files: Dict[str, Dict[str, PipelineFile]] = {}
        self.routes: Dict[str, Any] = {}

    def register(self) -> None:
        server = self._server
        server.add_route(
            "POST",
            "/api/v1/pipelines",
            self._handle_create,
            namespace="pipelines",
        )
        server.add_route(
            "GET",
            "/api/v1/pipelines",
            self._handle_list,
            namespace="pipelines",
        )
        server.add_route(
            "GET",
            "/api/v1/pipelines/{pipeline_id}",
            self._handle_get,
            namespace="pipelines",
        )
        server.add_route(
            "PUT",
            "/api/v1/pipelines/{pipeline_id}",
            self._handle_update,
            namespace="pipelines",
        )
        server.add_route(
            "DELETE",
            "/api/v1/pipelines/{pipeline_id}",
            self._handle_delete,
            namespace="pipelines",
        )
        server.add_route(
            "GET",
            "/api/v1/pipelines/{pipeline_id}/status",
            self._handle_get_status,
            namespace="pipelines",
        )
        server.add_route(
            "POST",
            "/api/v1/pipelines/{pipeline_id}/retrieve",
            self._handle_retrieve,
            namespace="pipelines",
        )
        # Document ingestion
        server.add_route(
            "POST",
            "/api/v1/pipelines/{pipeline_id}/documents",
            self._handle_create_documents,
            namespace="pipelines",
        )
        server.add_route(
            "PUT",
            "/api/v1/pipelines/{pipeline_id}/documents",
            self._handle_upsert_documents,
            namespace="pipelines",
        )
        # File ingestion
        server.add_route(
            "PUT",
            "/api/v1/pipelines/{pipeline_id}/files",
            self._handle_upsert_files,
            namespace="pipelines",
        )

    # Handlers -------------------------------------------------------

    def _handle_create(self, request: httpx.Request) -> httpx.Response:
        payload = self._server.json(request)
        name = payload.get("name")
        if not name:
            return self._server.json_response(
                {"detail": "name is required"}, status_code=400
            )

        pipeline_id = self._server.new_id("pipeline")
        project_id = request.url.params.get(
            "project_id", self._server.default_project_id
        )
        now = utcnow()

        embedding_config = payload.get("embedding_config") or {
            "type": "MANAGED_OPENAI_EMBEDDING",
            "component": {},
        }

        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            project_id=project_id,
            embedding_config=embedding_config,
            created_at=now,
            updated_at=now,
            pipeline_type=payload.get("pipeline_type", "MANAGED"),
            status="CREATED",
        )
        self._pipelines[pipeline_id] = pipeline
        self._documents[pipeline_id] = {}
        self._files[pipeline_id] = {}
        return self._server.json_response(pipeline.model_dump(), status_code=200)

    def _handle_list(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        project_id = params.get("project_id")
        pipeline_name = params.get("pipeline_name")

        pipelines = list(self._pipelines.values())
        if project_id:
            pipelines = [p for p in pipelines if p.project_id == project_id]
        if pipeline_name:
            pipelines = [p for p in pipelines if p.name == pipeline_name]

        return self._server.json_response([p.model_dump() for p in pipelines])

    def _handle_get(self, request: httpx.Request) -> httpx.Response:
        pipeline_id = request.url.path.split("/")[-1]
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return self._server.json_response(
                {"detail": f"Pipeline {pipeline_id} not found"}, status_code=404
            )
        return self._server.json_response(pipeline.model_dump())

    def _handle_update(self, request: httpx.Request) -> httpx.Response:
        pipeline_id = request.url.path.split("/")[-1]
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return self._server.json_response(
                {"detail": f"Pipeline {pipeline_id} not found"}, status_code=404
            )

        payload = self._server.json(request)
        data = pipeline.model_dump()
        data.update({k: v for k, v in payload.items() if v is not None})
        data["updated_at"] = utcnow()
        updated = Pipeline.model_validate(data)
        self._pipelines[pipeline_id] = updated
        return self._server.json_response(updated.model_dump())

    def _handle_delete(self, request: httpx.Request) -> httpx.Response:
        pipeline_id = request.url.path.split("/")[-1]
        self._pipelines.pop(pipeline_id, None)
        self._documents.pop(pipeline_id, None)
        self._files.pop(pipeline_id, None)
        return self._server.json_response({}, status_code=200)

    def _handle_get_status(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        pipeline_id = parts[-2]
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return self._server.json_response(
                {"detail": f"Pipeline {pipeline_id} not found"}, status_code=404
            )
        status = ManagedIngestionStatusResponse(status="SUCCESS")
        return self._server.json_response(status.model_dump())

    # Helpers --------------------------------------------------------

    @staticmethod
    def _extract_list(payload: Any, key: str) -> List[Dict[str, Any]]:
        """Extract a list from payload that may be a bare array or a dict."""
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get(key, [])
        return []

    # Document ingestion ---------------------------------------------

    def _ingest_documents(
        self, pipeline_id: str, documents: List[Dict[str, Any]]
    ) -> List[CloudDocument]:
        store = self._documents.setdefault(pipeline_id, {})
        results: List[CloudDocument] = []
        for doc_payload in documents:
            doc_id = doc_payload.get("id") or self._server.new_id("doc")
            doc = CloudDocument(
                id=doc_id,
                text=doc_payload.get("text", ""),
                metadata=doc_payload.get("metadata", {}),
                excluded_embed_metadata_keys=doc_payload.get(
                    "excluded_embed_metadata_keys"
                ),
                excluded_llm_metadata_keys=doc_payload.get(
                    "excluded_llm_metadata_keys"
                ),
            )
            store[doc_id] = doc
            results.append(doc)
        return results

    def _handle_create_documents(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        pipeline_id = parts[-2]
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return self._server.json_response(
                {"detail": f"Pipeline {pipeline_id} not found"}, status_code=404
            )

        payload = self._server.json(request)
        documents = self._extract_list(payload, "documents")
        results = self._ingest_documents(pipeline_id, documents)
        return self._server.json_response(
            [d.model_dump() for d in results], status_code=200
        )

    def _handle_upsert_documents(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        pipeline_id = parts[-2]
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return self._server.json_response(
                {"detail": f"Pipeline {pipeline_id} not found"}, status_code=404
            )

        payload = self._server.json(request)
        documents = self._extract_list(payload, "documents")
        results = self._ingest_documents(pipeline_id, documents)
        return self._server.json_response(
            [d.model_dump() for d in results], status_code=200
        )

    # File ingestion -------------------------------------------------

    def _handle_upsert_files(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        pipeline_id = parts[-2]
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return self._server.json_response(
                {"detail": f"Pipeline {pipeline_id} not found"}, status_code=404
            )

        payload = self._server.json(request)
        file_ids = self._extract_list(payload, "files")
        now = utcnow()
        store = self._files.setdefault(pipeline_id, {})
        results: List[PipelineFile] = []
        for entry in file_ids:
            # Each entry can be a dict with file_id + optional metadata, or a plain string
            if isinstance(entry, dict):
                file_id = entry.get("file_id", self._server.new_id("file"))
                custom_metadata = entry.get("custom_metadata")
                name = entry.get("name")
            else:
                file_id = str(entry)
                custom_metadata = None
                name = None

            pf_id = self._server.new_id("pf")
            pf = PipelineFile(
                id=pf_id,
                pipeline_id=pipeline_id,
                file_id=file_id,
                name=name or f"file-{file_id}",
                status="SUCCESS",
                project_id=pipeline.project_id,
                custom_metadata=custom_metadata,
                created_at=now,
                updated_at=now,
            )
            store[pf_id] = pf
            results.append(pf)
        return self._server.json_response(
            [pf.model_dump() for pf in results], status_code=200
        )

    # Retrieval ------------------------------------------------------

    def _handle_retrieve(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")
        pipeline_id = parts[-2]
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return self._server.json_response(
                {"detail": f"Pipeline {pipeline_id} not found"}, status_code=404
            )

        payload = self._server.json(request)
        query = payload.get("query", "")
        top_k = payload.get("dense_similarity_top_k") or 3

        nodes = self._build_retrieval_nodes(pipeline_id, query, top_k)
        response = PipelineRetrieveResponse(
            pipeline_id=pipeline_id,
            retrieval_nodes=nodes,
        )
        return self._server.json_response(response.model_dump())

    def _build_retrieval_nodes(
        self, pipeline_id: str, query: str, top_k: int
    ) -> List[RetrievalNode]:
        """Build retrieval nodes from ingested documents and files."""
        chunks: List[_Chunk] = []

        # Collect chunks from ingested documents
        for doc_id, doc in self._documents.get(pipeline_id, {}).items():
            text = doc.text or ""
            # Split the document text into paragraph-sized chunks
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text] if text else []
            for i, para in enumerate(paragraphs):
                chunks.append(
                    _Chunk(
                        text=para,
                        source_id=doc_id,
                        chunk_index=i,
                        metadata=dict(doc.metadata) if doc.metadata else {},
                    )
                )

        # Collect chunks from ingested files (generate deterministic text)
        for pf_id, pf in self._files.get(pipeline_id, {}).items():
            seed = combined_seed(pipeline_id, pf.file_id or pf_id)
            file_text = generate_text_blob(seed, sentences=6)
            # Split generated text into sentence-pair chunks
            sentences = file_text.split(". ")
            for i in range(0, len(sentences), 2):
                chunk_text = ". ".join(sentences[i : i + 2])
                if not chunk_text.endswith("."):
                    chunk_text += "."
                metadata: Dict[str, Any] = {"file_name": pf.name or ""}
                if pf.file_id:
                    metadata["file_id"] = pf.file_id
                chunks.append(
                    _Chunk(
                        text=chunk_text,
                        source_id=pf.file_id or pf_id,
                        chunk_index=i // 2,
                        metadata=metadata,
                    )
                )

        if not chunks:
            return []

        # Score chunks deterministically based on query+chunk content
        scored: List[tuple[float, _Chunk]] = []
        for chunk in chunks:
            seed = combined_seed(query, chunk.text)
            # Generate a score between 0.5 and 1.0
            score = 0.5 + (seed % 5000) / 10000.0
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:top_k]

        nodes: List[RetrievalNode] = []
        for score, chunk in selected:
            node_id = self._server.new_id("node")
            text_node = TextNode(
                id=node_id,
                text=chunk.text,
                extra_info=chunk.metadata or None,
                start_char_idx=0,
                end_char_idx=len(chunk.text),
            )
            nodes.append(
                RetrievalNode(
                    node=text_node,
                    score=round(score, 4),
                )
            )
        return nodes


class _Chunk:
    """Internal helper to represent a text chunk for retrieval."""

    __slots__ = ("text", "source_id", "chunk_index", "metadata")

    def __init__(
        self,
        *,
        text: str,
        source_id: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.source_id = source_id
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
