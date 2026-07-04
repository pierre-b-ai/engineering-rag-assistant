import chromadb

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import VECTOR_DB_DIR, COLLECTION_NAME
from src.embeddings.embedder import get_embedding_model


def build_vector_store(chunks: list[dict]) -> Chroma:
    """
    Crée une base Chroma à partir des chunks.
    """

    # Documents LangChain
    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        for chunk in chunks
    ]

    # Modèle embeddings
    embedding_model = get_embedding_model()

    # Base vectorielle persistante
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME,
    )

    # Résultat
    return vector_store


def load_vector_store() -> Chroma:
    """
    Recharge une base Chroma existante.
    """

    # Modèle embeddings
    embedding_model = get_embedding_model()

    # Base existante
    vector_store = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    # Résultat
    return vector_store


def clear_vector_store() -> None:
    """
    Supprime la collection Chroma sans supprimer les fichiers Windows.
    """

    # Client Chroma persistant
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # Suppression collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass