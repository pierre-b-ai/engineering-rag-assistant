from __future__ import annotations

import hashlib

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.config import VECTOR_DB_DIR, COLLECTION_NAME
from src.embeddings.embedder import get_embedding_model
from src.ingestion.document_manifest import delete_manifest


def _chunk_storage_id(chunk: dict) -> str:
    metadata = chunk["metadata"]
    raw_id = "|".join(
        [
            str(metadata.get("file_hash", "")),
            str(metadata.get("page", "")),
            str(metadata.get("chunk_id", "")),
        ]
    )
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def load_vector_store() -> Chroma:
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )


def add_chunks_to_vector_store(chunks: list[dict]) -> int:
    """Ajoute des chunks sans reconstruire toute la collection."""
    if not chunks:
        return 0

    documents = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]
    ids = [_chunk_storage_id(chunk) for chunk in chunks]
    vector_store = load_vector_store()
    vector_store.add_documents(documents=documents, ids=ids)
    return len(documents)


def build_vector_store(chunks: list[dict]) -> Chroma:
    """Compatibilité avec l'ancien pipeline : ajoute les chunks fournis."""
    add_chunks_to_vector_store(chunks)
    return load_vector_store()


def delete_document_from_vector_store(source: str) -> int:
    """Supprime tous les chunks associés à un document source."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return 0

    try:
        existing = collection.get(where={"source": source}, include=[])
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0



def delete_file_version_from_vector_store(source: str, file_hash: str) -> int:
    """Supprime uniquement une version précise d'un document."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return 0

    try:
        existing = collection.get(
            where={"$and": [{"source": source}, {"file_hash": file_hash}]},
            include=[],
        )
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def get_vector_store_count() -> int:
    """Retourne le nombre de chunks actuellement stockés dans Chroma."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return int(collection.count())
    except Exception:
        return 0


def clear_vector_store(*, clear_manifest: bool = True) -> None:
    """Supprime la collection Chroma et, par défaut, son manifeste."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    if clear_manifest:
        delete_manifest()


def get_indexed_chunks(
    *,
    source: str | None = None,
    page: int | None = None,
    quality_status: str | None = None,
) -> list[dict]:
    """Retourne les chunks réellement stockés dans Chroma pour inspection."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    where_parts: list[dict] = []
    if source:
        where_parts.append({"source": source})
    if page is not None:
        where_parts.append({"page": int(page)})
    if quality_status and quality_status != "all":
        where_parts.append({"chunk_quality_status": quality_status})

    where = None
    if len(where_parts) == 1:
        where = where_parts[0]
    elif len(where_parts) > 1:
        where = {"$and": where_parts}

    kwargs = {"include": ["documents", "metadatas"]}
    if where is not None:
        kwargs["where"] = where

    try:
        result = collection.get(**kwargs)
    except Exception:
        return []

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    chunks = [
        {
            "id": chunk_id,
            "text": document or "",
            "metadata": metadata or {},
        }
        for chunk_id, document, metadata in zip(ids, documents, metadatas)
    ]

    return sorted(
        chunks,
        key=lambda item: (
            str(item["metadata"].get("source", "")).lower(),
            int(item["metadata"].get("page", 0) or 0),
            int(item["metadata"].get("chunk_id", 0) or 0),
        ),
    )


def get_indexed_sources() -> list[str]:
    """Liste les documents présents dans l'index."""
    return sorted(
        {
            str(chunk["metadata"].get("source"))
            for chunk in get_indexed_chunks()
            if chunk["metadata"].get("source")
        },
        key=str.lower,
    )
