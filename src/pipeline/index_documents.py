from pathlib import Path

from app.config import RAW_DATA_DIR
from src.ingestion.pdf_loader import load_pdf_text
from src.chunking.splitter import split_pages_into_chunks
from src.vectorstore.chroma_store import build_vector_store


def index_raw_pdfs() -> int:
    """
    Indexe tous les PDF présents dans data/raw.
    """

    # Liste PDF
    pdf_files = list(Path(RAW_DATA_DIR).glob("*.pdf"))

    # Stockage chunks
    all_chunks = []

    # Parcours PDF
    for pdf_file in pdf_files:

        # Lecture PDF
        pages = load_pdf_text(pdf_file)

        # Découpage chunks
        chunks = split_pages_into_chunks(pages)

        # Ajout
        all_chunks.extend(chunks)

    # Si aucun chunk
    if not all_chunks:
        return 0

    # Stockage Chroma
    build_vector_store(all_chunks)

    # Nombre chunks
    return len(all_chunks)