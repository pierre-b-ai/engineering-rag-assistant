"""Réécriture déterministe des requêtes avec un modèle local Ollama."""

from __future__ import annotations

import requests

from src.config import OLLAMA_URL, QUERY_REWRITE_MODEL

# Une seed fixe rend les évaluations reproductibles. Elle n'interdit pas toute
# variation entre versions d'Ollama ou du modèle, mais stabilise un environnement
# donné pour un prompt identique.
QUERY_REWRITE_SEED = 42


def rewrite_query_for_retrieval(query: str) -> str:
    """Produit une reformulation courte adaptée à la recherche sémantique."""

    prompt = f"""
Reformule la question suivante en une phrase courte et naturelle en français,
optimisée pour une recherche sémantique dans des notices techniques.

Règles strictes :
- conserve exactement les noms de modèles, références, codes, nombres et unités ;
- conserve le sens et les relations exprimées dans la question ;
- ajoute au maximum deux ou trois synonymes techniques utiles ;
- n'invente aucune information ni valeur chiffrée ;
- ne réponds pas à la question ;
- réponds avec une seule phrase, sans explication, liste ni guillemets.

Question originale :
{query}

Requête reformulée :
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": QUERY_REWRITE_MODEL,
            "prompt": prompt,
            "stream": False,
            # Gemma consommait auparavant toute la limite de génération dans sa
            # phase de raisonnement. Le rewrite ne nécessite pas cette phase.
            "think": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "num_predict": 120,
                "seed": QUERY_REWRITE_SEED,
            },
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    rewritten_query = data.get("response", "").strip()

    # Repli sûr : une panne ou une sortie vide ne doit jamais empêcher le
    # retrieval. La question originale reste alors la requête de recherche.
    return rewritten_query or query
