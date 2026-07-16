"""Orchestration de l'indexation documentaire.

Ce module centralise le scan du corpus, le contrôle du manifeste, l'ingestion,
le chunking et la mise à jour incrémentale de Chroma. L'interface Streamlit ne
fait qu'appeler cette couche applicative.
"""

from __future__ import annotations

from pathlib import Path

from src.config import RAW_DATA_DIR
from src.chunking.splitter import split_pages_into_chunks
from src.ingestion.document_manifest import (
    compute_file_hash,
    current_index_signature,
    empty_manifest,
    load_manifest,
    manifest_matches_current_config,
    save_manifest,
)
from src.ingestion.ingestion_report import IngestionReport
from src.ingestion.pdf_loader import load_pdf_text
from src.vectorstore.chroma_store import (
    add_chunks_to_vector_store,
    clear_vector_store,
    delete_document_from_vector_store,
    delete_file_version_from_vector_store,
    get_vector_store_count,
)


def scan_pdf_corpus(raw_data_dir: str | Path = RAW_DATA_DIR) -> dict[str, Path]:
    """Retourne les PDF du corpus, indexés par nom de fichier."""
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return {
        path.name: path
        for path in sorted(raw_dir.glob("*.pdf"), key=lambda item: item.name.lower())
    }


def _prepare_document(
    pdf_path: Path,
    *,
    file_hash: str,
    report: IngestionReport,
) -> tuple[list[dict], dict]:
    """Extrait et découpe un document sans modifier le vector store."""
    pages_before = report.pages_analyzed
    indexed_before = report.pages_indexed
    rejected_before = report.pages_rejected

    pages = load_pdf_text(pdf_path, file_hash=file_hash, report=report)
    chunks = split_pages_into_chunks(pages)

    manifest_entry = {
        "file_hash": file_hash,
        "file_size": pdf_path.stat().st_size,
        "chunk_count": len(chunks),
        "pages_analyzed": report.pages_analyzed - pages_before,
        "pages_indexed": report.pages_indexed - indexed_before,
        "pages_rejected": report.pages_rejected - rejected_before,
    }
    return chunks, manifest_entry


def update_document_index(
    source: str,
    pdf_path: Path,
    *,
    current_hash: str,
    previous_entry: dict | None,
    report: IngestionReport,
) -> dict:
    """Prépare puis remplace atomiquement au mieux une version de document.

    La nouvelle version est ajoutée avant la suppression de l'ancienne. Grâce
    au hash stocké dans les métadonnées, l'ancienne version peut ensuite être
    retirée sans supprimer les nouveaux chunks. Une erreur d'extraction ne
    détruit donc pas l'index valide précédent.
    """
    chunks, manifest_entry = _prepare_document(
        pdf_path,
        file_hash=current_hash,
        report=report,
    )

    added_count = add_chunks_to_vector_store(chunks)
    report.chunks_added += added_count
    manifest_entry["chunk_count"] = added_count

    if previous_entry:
        previous_hash = previous_entry.get("file_hash")
        if previous_hash and previous_hash != current_hash:
            report.chunks_deleted += delete_file_version_from_vector_store(
                source=source,
                file_hash=previous_hash,
            )

    return manifest_entry


def index_raw_pdfs(*, force_rebuild: bool = False) -> IngestionReport:
    """Met à jour l'index pour les PDF nouveaux, modifiés ou supprimés."""
    report = IngestionReport(full_rebuild=force_rebuild)
    current_files = scan_pdf_corpus()
    report.documents_detected = len(current_files)

    manifest = load_manifest()

    if (
        not force_rebuild
        and not manifest.get("documents")
        and get_vector_store_count() > 0
    ):
        report.rebuild_required = True
        report.message = (
            "Un index existant a été créé sans manifeste incrémental. "
            "Effectuez une reconstruction complète une seule fois."
        )
        return report

    if not force_rebuild and not manifest_matches_current_config(manifest):
        report.rebuild_required = True
        report.message = (
            "La configuration d'indexation a changé. "
            "Effectuez une reconstruction complète."
        )
        return report

    if force_rebuild:
        clear_vector_store(clear_manifest=True)
        manifest = empty_manifest()

    old_documents = dict(manifest.get("documents", {}))
    new_manifest_documents = dict(old_documents)

    current_names = set(current_files)
    old_names = set(old_documents)

    for source in sorted(old_names - current_names):
        report.chunks_deleted += delete_document_from_vector_store(source)
        new_manifest_documents.pop(source, None)
        report.documents_deleted += 1

    for source, pdf_path in current_files.items():
        try:
            current_hash = compute_file_hash(pdf_path)
        except OSError:
            report.documents_failed += 1
            report.failed_documents.append(source)
            continue

        previous = old_documents.get(source)
        if previous and previous.get("file_hash") == current_hash:
            report.documents_unchanged += 1
            continue

        try:
            manifest_entry = update_document_index(
                source,
                pdf_path,
                current_hash=current_hash,
                previous_entry=previous,
                report=report,
            )
        except Exception as error:
            report.documents_failed += 1
            report.failed_documents.append(source)
            report.document_errors[source] = str(error)
            # On conserve l'entrée précédente : l'ancienne version reste valide.
            continue

        new_manifest_documents[source] = manifest_entry
        if previous:
            report.documents_modified += 1
        else:
            report.documents_new += 1

    save_manifest(
        {
            "version": 1,
            "index_signature": current_index_signature(),
            "documents": new_manifest_documents,
        }
    )
    return report
