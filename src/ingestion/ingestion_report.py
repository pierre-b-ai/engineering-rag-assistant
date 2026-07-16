"""Structures de données utilisées pour le bilan d'indexation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RejectedPage:
    source: str
    page: int
    reason: str


@dataclass
class IngestionReport:
    documents_detected: int = 0
    documents_new: int = 0
    documents_modified: int = 0
    documents_unchanged: int = 0
    documents_deleted: int = 0
    documents_failed: int = 0
    pages_analyzed: int = 0
    pages_indexed: int = 0
    pages_rejected: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    full_rebuild: bool = False
    rebuild_required: bool = False
    message: str = ""
    rejected_pages: list[RejectedPage] = field(default_factory=list)
    failed_documents: list[str] = field(default_factory=list)
    document_errors: dict[str, str] = field(default_factory=dict)

    def add_rejected_page(self, source: str, page: int, reason: str) -> None:
        self.pages_rejected += 1
        self.rejected_pages.append(
            RejectedPage(source=source, page=page, reason=reason)
        )

    def to_dict(self) -> dict:
        return asdict(self)
