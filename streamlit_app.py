import random

import streamlit as st

from src.generation.ollama_generator import generate_answer_with_ollama
from src.pipeline.indexing_pipeline import index_raw_pdfs
from src.retrieval.bm25_index import clear_bm25_cache
from src.retrieval.retriever import (
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_hybrid,
    retrieve_relevant_chunks_hybrid_with_rewrite,
    retrieve_relevant_chunks_with_rewrite,
)
from src.vectorstore.chroma_store import (
    clear_vector_store,
    get_indexed_chunks,
    get_indexed_sources,
)


st.set_page_config(
    page_title="Engineering RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Engineering RAG Assistant")
st.write(
    "Ask a question about indexed technical documents, inspect the exact "
    "chunks stored in Chroma, and manage the document index."
)


def display_ingestion_report(report) -> None:
    if report.rebuild_required:
        st.warning(report.message)
        return

    if report.documents_failed:
        st.warning(
            f"{report.documents_failed} document(s) could not be indexed: "
            + ", ".join(report.failed_documents)
        )
        if report.document_errors:
            with st.expander("Indexing errors", expanded=False):
                for source, error in report.document_errors.items():
                    st.write(f"- {source}: {error}")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("New documents", report.documents_new)
    col_b.metric("Modified documents", report.documents_modified)
    col_c.metric("Unchanged documents", report.documents_unchanged)
    col_d.metric("Deleted documents", report.documents_deleted)

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Pages analyzed", report.pages_analyzed)
    col_f.metric("Pages indexed", report.pages_indexed)
    col_g.metric("Pages excluded", report.pages_rejected)
    col_h.metric("Chunks added", report.chunks_added)

    if report.rejected_pages:
        with st.expander("Pages not indexed", expanded=False):
            for rejected in report.rejected_pages:
                st.write(
                    f"- {rejected.source} — page {rejected.page}: "
                    f"{rejected.reason}"
                )


def display_chunk(chunk: dict, *, expanded: bool = False) -> None:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "unknown source")
    page = metadata.get("page", "unknown page")
    chunk_number = metadata.get("chunk_id", "unknown")
    status = metadata.get("chunk_quality_status", "unknown")
    quality_score = metadata.get("chunk_quality_score")
    length = metadata.get("chunk_length", len(chunk.get("text", "")))

    score_text = (
        f"{float(quality_score):.3f}" if quality_score is not None else "N/A"
    )
    title = (
        f"{source} | page {page} | chunk {chunk_number} | "
        f"quality {status} ({score_text}) | {length} chars"
    )

    with st.expander(title, expanded=expanded):
        st.caption(
            f"Extraction: {metadata.get('extraction_method', 'unknown')} · "
            f"Reason: {metadata.get('chunk_quality_reason', 'unknown')} · "
            f"Chroma ID: {chunk.get('id', 'not displayed')}"
        )
        st.write(chunk.get("text", ""))


def choose_retriever(*, retrieval_method: str, use_rewrite: bool):
    """Sélectionne une stratégie sans dupliquer la logique de recherche."""

    if retrieval_method == "Hybrid (BGE-M3 + BM25/RRF)":
        return (
            retrieve_relevant_chunks_hybrid_with_rewrite
            if use_rewrite
            else retrieve_relevant_chunks_hybrid
        )

    return (
        retrieve_relevant_chunks_with_rewrite
        if use_rewrite
        else retrieve_relevant_chunks
    )


def build_retrieval_caption(chunk: dict) -> str:
    """Construit un résumé des scores adapté au mode dense ou hybride."""

    retrieval_mode = chunk.get("retrieval_mode", "dense")

    if retrieval_mode.startswith("hybrid"):
        rrf_score = chunk.get("rrf_score")
        dense_rank = chunk.get("dense_rank")
        bm25_rank = chunk.get("bm25_rank")
        bm25_score = chunk.get("bm25_score")

        parts = [
            f"mode {retrieval_mode}",
            f"RRF {rrf_score:.5f}" if rrf_score is not None else "RRF N/A",
            f"dense rank {dense_rank if dense_rank is not None else '—'}",
            f"BM25 rank {bm25_rank if bm25_rank is not None else '—'}",
        ]
        if bm25_score is not None:
            parts.append(f"BM25 score {bm25_score:.3f}")

        return " | ".join(parts)

    distance = chunk.get("score")
    semantic_score = chunk.get("semantic_score")
    distance_text = f"{distance:.4f}" if distance is not None else "N/A"
    semantic_text = (
        f"{semantic_score:.4f}" if semantic_score is not None else "N/A"
    )
    return (
        f"mode {retrieval_mode} | distance {distance_text} | "
        f"semantic score {semantic_text}"
    )


search_tab, explorer_tab, index_tab = st.tabs(
    ["Search and answer", "Chunk explorer", "Index management"]
)

with search_tab:
    answer_mode = st.selectbox(
        "Answer mode",
        options=["Retrieval only", "Local LLM", "API LLM (coming soon)"],
    )

    retrieval_method = st.selectbox(
        "Retrieval method",
        options=[
            "Dense (BGE-M3)",
            "Hybrid (BGE-M3 + BM25/RRF)",
        ],
        help=(
            "Hybrid retrieval combines semantic search and exact lexical "
            "matching, then merges their rankings with RRF."
        ),
    )

    use_query_rewriting = st.checkbox(
        "Use query rewriting",
        value=False,
        help=(
            "Experimental: adds a deterministic Gemma reformulation to the "
            "original question before retrieval."
        ),
    )

    query = st.text_input(
        "Question",
        placeholder="Example: Quels réglages en fonction de la dureté de l'eau ?",
    )

    k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=5,
        value=3,
    )

    if st.button("Search", key="search_button"):
        if not query.strip():
            st.warning("Please enter a question.")
        elif answer_mode == "API LLM (coming soon)":
            st.info("API LLM answer generation will be added in a future version.")
        else:
            retriever = choose_retriever(
                retrieval_method=retrieval_method,
                use_rewrite=use_query_rewriting,
            )

            with st.spinner("Searching relevant chunks..."):
                chunks = retriever(query=query, k=k)

            if chunks:
                first_chunk = chunks[0]
                rewritten_query = first_chunk.get("rewritten_query")
                search_query = first_chunk.get("search_query", query)

                if rewritten_query:
                    st.caption(f"Rewritten query: {rewritten_query}")
                st.caption(f"Search query used: {search_query}")

            if answer_mode == "Local LLM" and chunks:
                with st.spinner("Generating answer with local LLM..."):
                    answer = generate_answer_with_ollama(query=query, chunks=chunks)
                st.subheader("Answer")
                st.write(answer)
                st.divider()

            st.subheader("Retrieved chunks")
            if not chunks:
                st.info("No relevant chunks found.")

            for i, chunk in enumerate(chunks, start=1):
                metadata = chunk["metadata"]
                source = metadata.get("source", "unknown source")
                page = metadata.get("page", "unknown page")
                chunk_id = metadata.get("chunk_id", "unknown chunk")
                status = metadata.get("chunk_quality_status", "unknown")
                score_caption = build_retrieval_caption(chunk)

                expander_title = (
                    f"Chunk {i} — {source} | page {page} | chunk {chunk_id} "
                    f"| quality {status}"
                )

                with st.expander(expander_title, expanded=True):
                    st.caption(score_caption)
                    st.write(chunk["text"])

with explorer_tab:
    st.subheader("Chunk explorer")
    st.caption(
        "This view reads Chroma directly, so it shows exactly what is indexed. "
        "Quality is informative only and does not change retrieval scores."
    )

    def clear_explorer_sample() -> None:
        """Forget the previous manual/random selection after a filter change."""
        st.session_state.pop("explorer_chunk_ids", None)

    def on_source_change() -> None:
        """Reset the dependent page filter when the selected document changes."""
        st.session_state["explorer_page"] = "All pages"
        clear_explorer_sample()

    sources = get_indexed_sources()
    source_options = ["All documents", *sources]

    if st.session_state.get("explorer_source") not in source_options:
        st.session_state["explorer_source"] = "All documents"

    selected_source = st.selectbox(
        "Document",
        source_options,
        key="explorer_source",
        on_change=on_source_change,
    )

    source_filter = None if selected_source == "All documents" else selected_source
    source_chunks = get_indexed_chunks(source=source_filter)

    available_pages = sorted(
        {
            int(chunk["metadata"].get("page"))
            for chunk in source_chunks
            if chunk["metadata"].get("page") is not None
        }
    )
    page_options = ["All pages", *available_pages]

    if st.session_state.get("explorer_page") not in page_options:
        st.session_state["explorer_page"] = "All pages"

    selected_page = st.selectbox(
        "Page",
        page_options,
        key="explorer_page",
        on_change=clear_explorer_sample,
    )
    page_filter = None if selected_page == "All pages" else int(selected_page)

    selected_quality = st.selectbox(
        "Extraction quality",
        ["all", "clean", "degraded"],
        key="explorer_quality",
        format_func=lambda value: value.capitalize(),
        on_change=clear_explorer_sample,
    )

    filtered_chunks = get_indexed_chunks(
        source=source_filter,
        page=page_filter,
        quality_status=selected_quality,
    )

    clean_count = sum(
        chunk["metadata"].get("chunk_quality_status") == "clean"
        for chunk in filtered_chunks
    )
    degraded_count = sum(
        chunk["metadata"].get("chunk_quality_status") == "degraded"
        for chunk in filtered_chunks
    )

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Matching chunks", len(filtered_chunks))
    metric_b.metric("Clean", clean_count)
    metric_c.metric("Degraded", degraded_count)

    chunk_count = len(filtered_chunks)
    max_sample_size = min(100, chunk_count)

    if chunk_count == 0:
        sample_size = 0
    elif chunk_count == 1:
        sample_size = 1
        st.caption("1 chunk matches the selected filters.")
    else:
        sample_size = st.slider(
            "Number of chunks to inspect",
            min_value=1,
            max_value=max_sample_size,
            value=min(20, max_sample_size),
        )

    controls_a, controls_b = st.columns(2)
    with controls_a:
        if st.button("Show first chunks", use_container_width=True):
            st.session_state["explorer_chunk_ids"] = [
                chunk["id"] for chunk in filtered_chunks[:sample_size]
            ]
    with controls_b:
        if st.button("Draw random sample", use_container_width=True):
            sample = (
                random.sample(
                    filtered_chunks,
                    k=min(sample_size, len(filtered_chunks)),
                )
                if filtered_chunks
                else []
            )
            st.session_state["explorer_chunk_ids"] = [
                chunk["id"] for chunk in sample
            ]

    selected_ids = st.session_state.get("explorer_chunk_ids")
    filtered_by_id = {chunk["id"]: chunk for chunk in filtered_chunks}
    if selected_ids:
        chunks_to_display = [
            filtered_by_id[chunk_id]
            for chunk_id in selected_ids
            if chunk_id in filtered_by_id
        ]
    else:
        chunks_to_display = filtered_chunks[:sample_size]

    if not filtered_chunks:
        st.info("No chunks match the selected filters.")
    else:
        st.caption(
            "Suggested manual check: inspect 30–50 chunks across documents, "
            "including all degraded chunks and several pages containing tables."
        )
        for chunk in chunks_to_display:
            display_chunk(chunk)

with index_tab:
    st.subheader("Document index")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Update index", use_container_width=True):
            with st.spinner("Checking new and modified PDFs..."):
                report = index_raw_pdfs(force_rebuild=False)
            # Le corpus lexical doit être reconstruit après une modification de
            # Chroma. L'opération est différée jusqu'à la prochaine recherche.
            clear_bm25_cache()
            st.session_state["last_ingestion_report"] = report
            st.session_state.pop("explorer_chunk_ids", None)
            if not report.rebuild_required:
                st.success("Incremental index update complete.")

    with col2:
        if st.button("Rebuild full index", use_container_width=True):
            with st.spinner("Rebuilding the full index..."):
                report = index_raw_pdfs(force_rebuild=True)
            clear_bm25_cache()
            st.session_state["last_ingestion_report"] = report
            st.session_state.pop("explorer_chunk_ids", None)
            st.success("Full index rebuild complete.")

    with col3:
        if st.button("Clear vector database", use_container_width=True):
            clear_vector_store(clear_manifest=True)
            clear_bm25_cache()
            st.session_state.pop("last_ingestion_report", None)
            st.session_state.pop("explorer_chunk_ids", None)
            st.success("Vector database, BM25 cache and manifest cleared.")

    if "last_ingestion_report" in st.session_state:
        display_ingestion_report(st.session_state["last_ingestion_report"])
