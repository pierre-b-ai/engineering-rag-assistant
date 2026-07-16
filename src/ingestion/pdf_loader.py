from pathlib import Path

from pypdf import PdfReader

from src.ingestion.ingestion_report import IngestionReport
from src.ingestion.text_quality import assess_text_quality


def load_pdf_text(
    pdf_path: str | Path,
    *,
    file_hash: str,
    report: IngestionReport,
) -> list[dict]:
    """Extrait un PDF page par page et exclut les pages inutilisables."""

    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        report.pages_analyzed += 1

        try:
            text = page.extract_text()
        except Exception:
            report.add_rejected_page(
                source=pdf_path.name,
                page=page_number,
                reason="pdf_text_extraction_error",
            )
            continue

        quality = assess_text_quality(text)
        if not quality.is_valid:
            report.add_rejected_page(
                source=pdf_path.name,
                page=page_number,
                reason=quality.reason,
            )
            continue

        report.pages_indexed += 1
        pages.append(
            {
                "text": text.strip(),
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                    "file_hash": file_hash,
                    "extraction_method": "native",
                    "text_quality_score": quality.score,
                    "text_quality_status": quality.status,
                    "text_quality_reason": quality.reason,
                },
            }
        )

    return pages
