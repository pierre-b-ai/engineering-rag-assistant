"""Configuration centrale du projet.

Les valeurs peuvent être personnalisées dans un fichier ``.env`` situé à la
racine du dépôt. Les valeurs par défaut permettent toutefois de lancer le
projet sans créer de fichier ``.env`` pour le mode retrieval-only.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# Racine du projet : dossier parent de ``src``.
BASE_DIR = Path(__file__).resolve().parents[1]

# Charge le fichier local .env s'il existe.
#
# ``override=False`` conserve la priorité des variables déjà définies dans le
# terminal ou par le système d'exploitation.
load_dotenv(BASE_DIR / ".env", override=False)


def _env(name: str, default: str) -> str:
    """Lit une variable d'environnement avec un repli sûr.

    Une variable absente ou laissée vide utilise la valeur par défaut.
    """

    value = os.getenv(name, "").strip()
    return value or default


# ---------------------------------------------------------------------------
# Dossiers
# ---------------------------------------------------------------------------

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


# ---------------------------------------------------------------------------
# Découpage du texte
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
MIN_CHUNK_LENGTH = 300


# ---------------------------------------------------------------------------
# Embeddings et base vectorielle
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
COLLECTION_NAME = "engineering_docs"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

# URL complète de l'endpoint de génération Ollama.
OLLAMA_URL = _env(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

# Modèle utilisé pour générer la réponse finale.
OLLAMA_MODEL = _env(
    "OLLAMA_MODEL",
    "gemma4:12b",
)

# Modèle utilisé pour reformuler les requêtes.
#
# Par défaut, il reprend OLLAMA_MODEL afin qu'un utilisateur puisse ne
# configurer qu'un seul modèle. Il peut toutefois être remplacé séparément
# dans le fichier .env.
QUERY_REWRITE_MODEL = _env(
    "QUERY_REWRITE_MODEL",
    OLLAMA_MODEL,
)
