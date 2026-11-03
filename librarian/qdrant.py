"""Qdrant REST storage. Credentials stay in the environment."""
from __future__ import annotations

import os
import uuid
from urllib.parse import quote, urlsplit

import requests

from .config import settings


class QdrantStore:
    def __init__(self, collection: str | None = None):
        url = settings.qdrant_url.rstrip("/")
        if not url:
            raise ValueError("Set LIB_QDRANT_URL (or QDRANT_URL) to your Qdrant cluster endpoint")
        parsed = urlsplit(url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
        ):
            raise ValueError("Use an HTTPS Qdrant URL (HTTP is allowed only on localhost)")
        self.collection = collection or settings.qdrant_collection
        self.base = f"{url}/collections/{quote(self.collection, safe='')}"
        self.session = requests.Session()
        key = os.environ.get("CLUSTER_KEY", "")
        if key:
            self.session.headers["api-key"] = key

    def request(self, method, suffix="", **kwargs):
        response = self.session.request(method, self.base + suffix, timeout=120, **kwargs)
        response.raise_for_status()
        return response.json().get("result")

    @staticmethod
    def hnsw_configuration():
        if (settings.qdrant_hnsw_m <= 0 or settings.qdrant_hnsw_ef_construct < 4
                or settings.qdrant_hnsw_ef <= 0
                or settings.qdrant_full_scan_threshold < 10
                or settings.qdrant_indexing_threshold <= 0):
            raise ValueError("HNSW requires positive m/ef/indexing threshold, ef_construct >= 4, full_scan_threshold >= 10")
        return {
            "hnsw_config": {
                "m": settings.qdrant_hnsw_m,
                "ef_construct": settings.qdrant_hnsw_ef_construct,
                "full_scan_threshold": settings.qdrant_full_scan_threshold,
            },
            "optimizers_config": {"indexing_threshold": settings.qdrant_indexing_threshold},
        }

    def configure_hnsw(self):
        """Update an existing collection in place; Qdrant indexes in the background."""
        return self.request("PATCH", json=self.hnsw_configuration())

    def upload(self, chunks, vectors, generation: str):
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Qdrant upload requires nonempty matching chunks and vectors")
        dims = int(vectors.shape[1])
        response = self.session.get(self.base, timeout=30)
        if response.status_code == 404:
            self.request("PUT", json={
                "vectors": {"size": dims, "distance": "Cosine"},
                **self.hnsw_configuration(),
            })
        else:
            response.raise_for_status()
            config = response.json()["result"]["config"]["params"]["vectors"]
            if config.get("size") != dims or config.get("distance") != "Cosine":
                raise ValueError("Qdrant collection vector configuration differs; use a new collection")
            self.configure_hnsw()
        if settings.qdrant_batch <= 0:
            raise ValueError("LIB_QDRANT_BATCH must be positive")
        for field in ("generation", "id"):
            self.request("PUT", "/index", params={"wait": "true"}, json={
                "field_name": field, "field_schema": "keyword",
            })
        for start in range(0, len(chunks), settings.qdrant_batch):
            points = []
            for i in range(start, min(start + settings.qdrant_batch, len(chunks))):
                chunk = chunks[i]
                payload = chunk.to_dict()
                payload["generation"] = generation
                points.append({
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, generation + ":" + chunk.id)),
                    "vector": vectors[i].tolist(), "payload": payload,
                })
            self.request("PUT", "/points", params={"wait": "true"}, json={"points": points})
            if start // settings.qdrant_batch % 10 == 0 or start + len(points) == len(chunks):
                print(f"[qdrant] uploaded {start + len(points)}/{len(chunks)} chunks", flush=True)

    def search(self, vector, n: int, generation: str, allowed_ids=None, *, exact=False):
        if n <= 0:
            return []
        if settings.qdrant_hnsw_ef <= 0:
            raise ValueError("LIB_QDRANT_HNSW_EF must be positive")
        conditions = [{"key": "generation", "match": {"value": generation}}]
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            conditions.append({"key": "id", "match": {"any": allowed_ids}})
        result = self.request("POST", "/points/query", json={
            "query": vector.tolist(), "limit": n,
            "filter": {"must": conditions}, "with_payload": ["id"],
            "params": {"hnsw_ef": max(n, settings.qdrant_hnsw_ef), "exact": exact},
        })
        return [(point["payload"]["id"], float(point["score"])) for point in result["points"]]
