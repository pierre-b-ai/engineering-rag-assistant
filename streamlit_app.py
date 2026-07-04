import streamlit as st

from src.retrieval.retriever import retrieve_relevant_chunks
from src.generation.ollama_generator import generate_answer_with_ollama
from src.generation.query_rewriter import rewrite_query_for_retrieval

from src.vectorstore.chroma_store import clear_vector_store
from src.pipeline.index_documents import index_raw_pdfs


# Configuration page
st.set_page_config(
    page_title="Engineering RAG Assistant",
    page_icon="📄",
    layout="wide",
)


# Titre
st.title("📄 Engineering RAG Assistant")

# Description
st.write(
    "Ask a question about indexed technical documents. "
    "The app retrieves the most relevant chunks with sources."
)

st.divider()

# Index documentaire
st.subheader("Document index")

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear vector database"):
        clear_vector_store()
        st.success("Vector database cleared.")

with col2:
    if st.button("Reindex PDFs from data/raw"):
        with st.spinner("Indexing PDFs..."):
            chunk_count = index_raw_pdfs()
        st.success(f"Indexing complete: {chunk_count} chunks stored.")

st.divider()

# Mode de réponse
answer_mode = st.selectbox(
    "Answer mode",
    options=[
        "Retrieval only",
        "Local LLM",
        "API LLM (coming soon)",
    ],
)

# Option expérimentale : query rewriting
use_query_rewriting = st.checkbox(
    "Use query rewriting",
    value=False,
    help="Experimental: rewrites the user question before retrieval. It can improve or degrade results.",
)

# Question utilisateur
query = st.text_input(
    "Question",
    placeholder="Example: Comment poser les panneaux isolants ?",
)

# Nombre de chunks
k = st.slider(
    "Number of chunks to retrieve",
    min_value=1,
    max_value=5,
    value=3,
)

# Bouton recherche
if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:
        # Mode API pas encore implémenté
        if answer_mode == "API LLM (coming soon)":
            st.info("API LLM answer generation will be added in a future version.")

        # Par défaut, on cherche avec la question originale
        search_query = query

        # Query rewriting optionnel
        if use_query_rewriting:
            with st.spinner("Rewriting query for retrieval..."):
                rewritten_query = rewrite_query_for_retrieval(query)

            # On garde la question originale + la reformulation
            # pour éviter qu'un mauvais rewrite remplace totalement la question initiale.
            search_query = f"{query} {rewritten_query}"

        st.caption(f"Search query used: {search_query}")

        # Recherche des chunks pertinents
        with st.spinner("Searching relevant chunks..."):
            chunks = retrieve_relevant_chunks(query=search_query, k=k)

        # Génération de réponse avec LLM local
        if answer_mode == "Local LLM" and chunks:
            with st.spinner("Generating answer with local LLM..."):
                answer = generate_answer_with_ollama(query=query, chunks=chunks)

            st.subheader("Answer")
            st.write(answer)

            st.divider()

        # Affichage des résultats
        st.subheader("Retrieved chunks")

        if not chunks:
            st.info("No relevant chunks found.")

        for i, chunk in enumerate(chunks, start=1):

            metadata = chunk["metadata"]

            distance = chunk.get("score")
            semantic_score = chunk.get("semantic_score")

            source = metadata.get("source", "unknown source")
            page = metadata.get("page", "unknown page")
            chunk_id = metadata.get("chunk_id", "unknown chunk")

            distance_text = f"{distance:.4f}" if distance is not None else "N/A"
            semantic_text = f"{semantic_score:.4f}" if semantic_score is not None else "N/A"

            expander_title = (
                f"Chunk {i} — {source} | page {page} | chunk {chunk_id} "
                f"| distance {distance_text} | semantic score {semantic_text}"
            )

            with st.expander(expander_title, expanded=True):
                st.write(chunk["text"])