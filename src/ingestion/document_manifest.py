"""Manifeste local permettant l'indexation incrémentale des PDF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_DIR,
)

MANIFEST_PATH = Path(VECTOR_DB_DIR) / "index_manifest.json"
MANIFEST_VERSION = 1


def compute_file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_index_signature() -> dict:
    """Paramètres qui rendent les embeddings/chunks existants incompatibles."""
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def empty_manifest() -> dict:
    return {
        "version": MANIFEST_VERSION,
        "index_signature": current_index_signature(),
        "documents": {},
    }


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return empty_manifest()

    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_manifest()

    if not isinstance(data, dict) or not isinstance(data.get("documents"), dict):
        return empty_manifest()
    return data


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MANIFEST_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(MANIFEST_PATH)


def delete_manifest() -> None:
    try:
        MANIFEST_PATH.unlink()
    except FileNotFoundError:
        pass


def manifest_matches_current_config(manifest: dict) -> bool:
    return manifest.get("index_signature") == current_index_signature()
