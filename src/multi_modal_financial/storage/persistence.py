"""Persistence subsystem for saving, loading, and serializing HybridIndex state."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from multi_modal_financial.indexing.hybrid import HybridIndex
from multi_modal_financial.types import Chunk, Document


@dataclass
class IndexManifest:
    """Metadata describing a persisted hybrid index archive."""

    version: str = "0.1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_documents: int = 0
    total_chunks: int = 0
    vector_dimension: int = 128
    bm25_params: dict[str, float] = field(default_factory=lambda: {"k1": 1.5, "b": 0.75})
    tickers: list[str] = field(default_factory=list)
    checksum: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> IndexManifest:
        return cls(**data)


class IndexPersistence:
    """Serializes and restores HybridIndex state to/from disk and archives."""

    @classmethod
    def save(
        cls,
        index: HybridIndex,
        output_path: str | Path,
        compress: bool = False,
    ) -> Path:
        """Save full hybrid index state to a directory or zip bundle."""
        target = Path(output_path)

        # Temporary work directory for staging
        work_dir = target if not compress else target.parent / f"{target.stem}_tmp_persist"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save chunks
            chunks_data = [chunk.to_dict() for chunk in index.chunks.values()]
            chunks_file = work_dir / "chunks.json"
            chunks_file.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

            # 2. Save documents
            docs_data = [doc.to_dict() for doc in index.documents.values()]
            docs_file = work_dir / "documents.json"
            docs_file.write_text(json.dumps(docs_data, indent=2), encoding="utf-8")

            # 3. Save BM25 state
            bm25_state = {
                "k1": index.bm25.k1,
                "b": index.bm25.b,
                "corpus_size": index.bm25.corpus_size,
                "avgdl": index.bm25.avgdl,
                "doc_freqs": dict(index.bm25.doc_freqs),
                "idf": index.bm25.idf,
                "doc_len": index.bm25.doc_len,
                "doc_tokens": index.bm25.doc_tokens,
                "doc_ids": index.bm25.doc_ids,
            }
            bm25_file = work_dir / "bm25.json"
            bm25_file.write_text(json.dumps(bm25_state, indent=2), encoding="utf-8")

            # 4. Save Dense Vectors
            vectors = index.vector.vectors if index.vector.vectors is not None else np.empty((0, index.vector.dimension), dtype=np.float32)
            vec_file = work_dir / "vectors.npz"
            np.savez_compressed(
                vec_file,
                vectors=vectors,
                doc_ids=np.array(index.vector.doc_ids, dtype=object),
                dimension=np.array([index.vector.dimension]),
            )

            # 5. Compute manifest and checksum
            tickers = sorted(
                {
                    doc.metadata.ticker.upper()
                    for doc in index.documents.values()
                    if doc.metadata.ticker
                }
            )

            hasher = hashlib.sha256()
            for p in [chunks_file, docs_file, bm25_file, vec_file]:
                hasher.update(p.read_bytes())
            checksum = hasher.hexdigest()

            manifest = IndexManifest(
                total_documents=len(index.documents),
                total_chunks=len(index.chunks),
                vector_dimension=index.vector.dimension,
                bm25_params={"k1": index.bm25.k1, "b": index.bm25.b},
                tickers=tickers,
                checksum=checksum,
            )
            manifest_file = work_dir / "manifest.json"
            manifest_file.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

            if compress:
                # Create zip archive
                archive_path = target if target.suffix.lower() == ".zip" else target.with_suffix(".zip")
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in [manifest_file, chunks_file, docs_file, bm25_file, vec_file]:
                        zf.write(f, arcname=f.name)
                shutil.rmtree(work_dir, ignore_errors=True)
                return archive_path

            return work_dir
        except Exception:
            if compress and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            raise

    @classmethod
    def load(cls, source_path: str | Path) -> HybridIndex:
        """Restore HybridIndex from persisted directory or zip archive."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Persisted index not found: {source}")

        cleanup_temp = False
        if source.is_file() and source.suffix.lower() == ".zip":
            temp_extract = source.parent / f"{source.stem}_unpacked"
            temp_extract.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(temp_extract)
            read_dir = temp_extract
            cleanup_temp = True
        else:
            read_dir = source

        try:
            # 1. Manifest
            manifest_path = read_dir / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("Corrupt archive: missing manifest.json")
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = IndexManifest.from_dict(manifest_data)

            # 2. Instantiate index
            hybrid = HybridIndex(
                dimension=manifest.vector_dimension,
                bm25_k1=manifest.bm25_params.get("k1", 1.5),
                bm25_b=manifest.bm25_params.get("b", 0.75),
            )

            # 3. Load documents
            docs_path = read_dir / "documents.json"
            if docs_path.exists():
                docs_raw = json.loads(docs_path.read_text(encoding="utf-8"))
                for d_dict in docs_raw:
                    doc = Document.from_dict(d_dict)
                    hybrid.documents[doc.doc_id] = doc

            # 4. Load chunks
            chunks_path = read_dir / "chunks.json"
            if chunks_path.exists():
                chunks_raw = json.loads(chunks_path.read_text(encoding="utf-8"))
                for c_dict in chunks_raw:
                    chunk = Chunk.from_dict(c_dict)
                    hybrid.chunks[chunk.chunk_id] = chunk

            # 5. Restore BM25
            bm25_path = read_dir / "bm25.json"
            if bm25_path.exists():
                bm25_state = json.loads(bm25_path.read_text(encoding="utf-8"))
                hybrid.bm25.k1 = bm25_state.get("k1", 1.5)
                hybrid.bm25.b = bm25_state.get("b", 0.75)
                hybrid.bm25.corpus_size = bm25_state.get("corpus_size", 0)
                hybrid.bm25.avgdl = bm25_state.get("avgdl", 0.0)
                hybrid.bm25.doc_freqs = bm25_state.get("doc_freqs", {})
                hybrid.bm25.idf = bm25_state.get("idf", {})
                hybrid.bm25.doc_len = bm25_state.get("doc_len", [])
                hybrid.bm25.doc_tokens = bm25_state.get("doc_tokens", [])
                hybrid.bm25.doc_ids = bm25_state.get("doc_ids", [])
                hybrid.bm25.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(hybrid.bm25.doc_ids)}

            # 6. Restore Dense Vectors
            vec_path = read_dir / "vectors.npz"
            if vec_path.exists():
                with np.load(vec_path, allow_pickle=True) as npz:
                    hybrid.vector.vectors = npz["vectors"].astype(np.float32)
                    doc_ids_arr = list(npz["doc_ids"])
                    hybrid.vector.doc_ids = doc_ids_arr
                    hybrid.vector.doc_id_to_idx = {d_id: i for i, d_id in enumerate(doc_ids_arr)}
                    hybrid.vector.dimension = int(npz["dimension"][0])

            return hybrid
        finally:
            if cleanup_temp and read_dir.exists():
                shutil.rmtree(read_dir, ignore_errors=True)

    @classmethod
    def export_corpus(cls, index: HybridIndex, output_file: str | Path) -> None:
        """Export all chunks and documents in an index as a single readable JSON bundle."""
        out = Path(output_file)
        payload = {
            "version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "total_documents": len(index.documents),
            "total_chunks": len(index.chunks),
            "documents": [doc.to_dict() for doc in index.documents.values()],
            "chunks": [chunk.to_dict() for chunk in index.chunks.values()],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def import_corpus(cls, input_file: str | Path) -> HybridIndex:
        """Import chunks and documents from a JSON bundle and rebuild indices."""
        inp = Path(input_file)
        data = json.loads(inp.read_text(encoding="utf-8"))

        hybrid = HybridIndex()
        for d_data in data.get("documents", []):
            doc = Document.from_dict(d_data)
            hybrid.documents[doc.doc_id] = doc

        for c_data in data.get("chunks", []):
            chunk = Chunk.from_dict(c_data)
            hybrid.chunks[chunk.chunk_id] = chunk

        # Re-index
        hybrid._rebuild_indices()
        return hybrid
