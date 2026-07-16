from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH
from src.ingestion.text_quality import assess_text_quality


def split_pages_into_chunks(pages: list[dict]) -> list[dict]:
    """Découpe les pages en chunks et conserve leurs métadonnées qualité.

    Un chunk partiellement dégradé reste indexé : son statut est informatif et
    n'est pas utilisé pour pénaliser son score de retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[dict] = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        split_texts = splitter.split_text(text)

        kept_chunk_index = 0
        for chunk_text in split_texts:
            cleaned = chunk_text.strip()
            if len(cleaned) < MIN_CHUNK_LENGTH:
                continue

            chunk_quality = assess_text_quality(cleaned)
            page_status = metadata.get("text_quality_status", "clean")
            chunk_status = chunk_quality.status

            # Une page valide ne produit jamais un chunk supprimé ici : un
            # résultat "rejected" au niveau chunk devient "degraded" afin de
            # conserver l'information sémantique éventuellement exploitable.
            if chunk_status == "rejected":
                chunk_status = "degraded"

            final_status = (
                "degraded"
                if "degraded" in {page_status, chunk_status}
                else "clean"
            )

            chunks.append(
                {
                    "text": cleaned,
                    "metadata": {
                        **metadata,
                        "chunk_id": kept_chunk_index,
                        "chunk_length": len(cleaned),
                        "chunk_quality_status": final_status,
                        "chunk_quality_score": chunk_quality.score,
                        "chunk_quality_reason": chunk_quality.reason,
                    },
                }
            )
            kept_chunk_index += 1

    return chunks
