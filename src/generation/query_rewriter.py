import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "dolphin-mixtral:latest"


def rewrite_query_for_retrieval(query: str) -> str:
    """
    Reformule une question utilisateur pour améliorer la recherche documentaire.

    Objectif :
    - ajouter des synonymes utiles ;
    - expliciter l'intention ;
    - ne pas répondre à la question ;
    - rester court.
    """

    prompt = f"""
Tu reformules une question utilisateur pour une recherche documentaire RAG.

Règles strictes :
- Réponds uniquement en français.
- Ne réponds pas à la question.
- Ne donne aucune explication.
- Ne produis pas de syntaxe booléenne : pas de AND, OR, guillemets ou parenthèses.
- Garde les mots importants de la question originale.
- Ajoute quelques synonymes techniques ou formulations proches susceptibles d'apparaître dans une notice technique.
- N'invente aucune valeur chiffrée.
- Retourne une seule ligne courte.


Question utilisateur :
{query}

Requête enrichie :
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 80,
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    rewritten_query = response.json()["response"].strip()

    return rewritten_query