import requests


from src.config import OLLAMA_URL, OLLAMA_MODEL


def build_context(chunks: list[dict]) -> str:
    """
    Construit le contexte fourni au LLM à partir des chunks retrouvés.
    """

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})

        source = metadata.get("source", "unknown source")
        page = metadata.get("page", "unknown page")
        chunk_id = metadata.get("chunk_id", "unknown chunk")
        text = chunk.get("text", "")

        context_parts.append(
            f"[Source {i}: {source} | page {page} | chunk {chunk_id}]\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer_with_ollama(query: str, chunks: list[dict]) -> str:
    """
    Génère une réponse avec le LLM local Ollama à partir des chunks récupérés.
    """

    context = build_context(chunks)

    prompt = f"""
Tu es un assistant RAG spécialisé dans les documents techniques.

Règles strictes :
- Réponds uniquement à partir du contexte fourni.
- Le contexte peut contenir des passages proches du sujet mais qui ne répondent pas directement à la question.
- Ne choisis pas automatiquement le premier passage : compare les sources disponibles.
- Utilise en priorité les passages qui répondent explicitement à la question posée.
- Ignore les passages seulement connexes s'ils ne donnent pas directement l'information demandée.
- Si plusieurs valeurs existent, explique clairement à quoi correspond chaque valeur.
- Si le contexte ne contient pas la réponse, dis clairement que tu ne sais pas.
- Ne fais pas d'hypothèse non supportée par les sources.
- Donne une réponse claire et concise.
- Cite les sources utilisées avec le nom du document et la page.


Question utilisateur :
{query}

Contexte :
{context}

Réponse :
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()["response"]