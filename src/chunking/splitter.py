from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_pages_into_chunks(pages: list[dict]) -> list[dict]:
    """
    Découpe les pages PDF en chunks avec metadata.
    """

    # Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Stockage chunks
    chunks = []

    # Parcours pages
    for page in pages:

        # Texte page
        text = page["text"]

        # Metadata page
        metadata = page["metadata"]

        # Découpe texte
        split_texts = splitter.split_text(text)

        # Parcours chunks
        for chunk_index, chunk_text in enumerate(split_texts):

            # Ajout chunk
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_index,
                    },
                }
            )

    # Résultat
    return chunks