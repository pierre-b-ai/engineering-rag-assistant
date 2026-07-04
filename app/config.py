from pathlib import Path

# Racine projet
BASE_DIR = Path(__file__).resolve().parents[1]

# Dossiers
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Découpage texte
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250
MIN_CHUNK_LENGTH = 300

# Modèle embeddings
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Collection Chroma
COLLECTION_NAME = "engineering_docs"