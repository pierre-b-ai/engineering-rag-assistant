from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL_NAME


def get_embedding_model():
    """
    Charge le modèle d'embeddings.
    """

    # Modèle local
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Résultat
    return embedding_model