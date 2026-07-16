"""Évaluation document + page des différentes stratégies de retrieval."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

SOURCE_KEYS = ("source", "file_name", "filename", "document", "doc_name")
PAGE_KEYS = ("page", "page_number", "page_index", "pdf_page")


@dataclass
class RetrievedChunk:
    """Vue normalisée d'un chunk, indépendante du retriever utilisé."""

    rank: int
    source: str | None
    page: int | None
    score: float | None
    semantic_score: float | None
    bm25_score: float | None
    rrf_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    retrieval_mode: str | None
    preview: str


@dataclass
class EvaluationResult:
    """Résultat complet d'une question du jeu d'évaluation."""

    id: str
    question: str
    rewritten_query: str | None
    search_query: str
    retrieval_mode: str | None
    expected_document: str
    expected_pages: list[int]
    first_relevant_rank: int | None
    hits: dict[str, bool]
    retrieved_chunks: list[RetrievedChunk]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Charge et valide la structure minimale du golden dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Le JSON doit contenir une liste de questions.")

    required = {"id", "question", "expected_document", "expected_pages"}
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"L'entrée {index} n'est pas un objet JSON.")

        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"L'entrée {index} ne contient pas : {', '.join(sorted(missing))}"
            )

        if not isinstance(item["expected_pages"], list):
            raise ValueError(
                f"L'entrée {item.get('id', index)} doit contenir une liste "
                "dans expected_pages."
            )

    return data


def import_retriever(import_path: str) -> Callable[..., Any]:
    """Charge une fonction avec la forme ``package.module:fonction``."""

    if ":" not in import_path:
        raise ValueError(
            "Utilise la forme package.module:fonction pour --retriever."
        )

    module_name, function_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)

    if not callable(function):
        raise TypeError(f"{import_path} n'est pas une fonction appelable.")

    return function


def call_retriever(retriever: Callable[..., Any], query: str, k: int) -> list[Any]:
    """Appelle un retriever en acceptant plusieurs signatures courantes."""

    attempts = [
        ((), {"query": query, "k": k}),
        ((), {"question": query, "top_k": k}),
        ((), {"query": query, "top_k": k}),
        ((query, k), {}),
        ((query,), {"k": k}),
        ((query,), {"top_k": k}),
        ((query,), {}),
    ]

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            results = retriever(*args, **kwargs)
            if results is None:
                raise ValueError("Le retriever a renvoyé None.")
            return list(results)[:k]
        except TypeError as error:
            last_error = error

    raise TypeError(
        "Impossible d'appeler la fonction de retrieval. "
        f"Dernière erreur : {last_error}"
    )


def get_metadata(document: Any) -> dict[str, Any]:
    """Extrait les métadonnées d'un dictionnaire ou d'un Document LangChain."""

    if isinstance(document, dict):
        metadata = document.get("metadata", {})
    else:
        metadata = getattr(document, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def get_text(document: Any) -> str:
    """Extrait le texte selon les formats courants du projet et de LangChain."""

    if isinstance(document, dict):
        for key in ("page_content", "text", "content", "chunk_text"):
            if document.get(key) is not None:
                return str(document[key])
        return ""

    for attribute in ("page_content", "text", "content"):
        value = getattr(document, attribute, None)
        if value is not None:
            return str(value)
    return ""


def first_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Retourne la première valeur non vide parmi plusieurs clés possibles."""

    for key in keys:
        if mapping.get(key) not in (None, ""):
            return mapping[key]
    return None


def as_float(value: Any) -> float | None:
    """Convertit une valeur numérique sans interrompre l'évaluation."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    """Convertit un rang en entier si possible."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_source(value: Any) -> str | None:
    """Ramène tous les chemins au simple nom de fichier."""

    if value is None:
        return None
    return Path(str(value).replace("\\", "/")).name


def normalize_page(value: Any, page_offset: int) -> int | None:
    """Normalise la page en appliquant éventuellement un offset."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value) + page_offset
    except (TypeError, ValueError):
        digits = "".join(character for character in str(value) if character.isdigit())
        return int(digits) + page_offset if digits else None


def normalize_result(raw: Any, rank: int, page_offset: int) -> RetrievedChunk:
    """Transforme un résultat dense ou hybride dans une structure commune."""

    document = raw
    tuple_score: float | None = None

    # Format LangChain fréquent : (Document, score).
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        document = raw[0]
        tuple_score = as_float(raw[1])

    metadata = get_metadata(document)
    mapping = document if isinstance(document, dict) else {}

    # Le champ score est volontairement générique : distance Chroma en dense,
    # score RRF en hybride. Les champs spécialisés ci-dessous lèvent l'ambiguïté.
    score = tuple_score
    if score is None:
        score = as_float(mapping.get("score"))
    if score is None:
        score = as_float(first_value(metadata, ("score", "distance", "similarity")))

    source = normalize_source(first_value(metadata, SOURCE_KEYS))
    page = normalize_page(first_value(metadata, PAGE_KEYS), page_offset)
    text = " ".join(get_text(document).split())
    preview = text[:220] + ("…" if len(text) > 220 else "")

    return RetrievedChunk(
        rank=rank,
        source=source,
        page=page,
        score=score,
        semantic_score=as_float(mapping.get("semantic_score")),
        bm25_score=as_float(mapping.get("bm25_score")),
        rrf_score=as_float(mapping.get("rrf_score")),
        dense_rank=as_int(mapping.get("dense_rank")),
        bm25_rank=as_int(mapping.get("bm25_rank")),
        retrieval_mode=(
            str(mapping.get("retrieval_mode"))
            if mapping.get("retrieval_mode") is not None
            else None
        ),
        preview=preview,
    )


def extract_query_information(
    raw_chunks: list[Any],
    original_question: str,
) -> tuple[str | None, str, str | None]:
    """Récupère la reformulation et le mode ajoutés par le retriever."""

    if not raw_chunks or not isinstance(raw_chunks[0], dict):
        return None, original_question, None

    first_chunk = raw_chunks[0]
    rewritten_query = first_chunk.get("rewritten_query")
    search_query = (
        first_chunk.get("search_query")
        or first_chunk.get("query_used")
        or original_question
    )
    retrieval_mode = first_chunk.get("retrieval_mode")

    return (
        str(rewritten_query) if rewritten_query is not None else None,
        str(search_query),
        str(retrieval_mode) if retrieval_mode is not None else None,
    )


def is_relevant(
    chunk: RetrievedChunk,
    expected_document: str,
    expected_pages: set[int],
) -> bool:
    """Valide un hit lorsque document et page appartiennent à la ground truth."""

    if chunk.source is None or chunk.page is None:
        return False

    same_document = (
        chunk.source.casefold() == Path(expected_document).name.casefold()
    )
    return same_document and chunk.page in expected_pages


def evaluate(
    dataset: list[dict[str, Any]],
    retriever: Callable[..., Any],
    top_ks: list[int],
    page_offset: int,
) -> tuple[dict[str, Any], list[EvaluationResult]]:
    """Évalue toutes les questions et calcule Page Hit@k et Page MRR."""

    max_k = max(top_ks)
    results: list[EvaluationResult] = []

    for index, item in enumerate(dataset, start=1):
        question = str(item["question"])
        expected_pages = [int(page) for page in item["expected_pages"]]
        raw_chunks = call_retriever(retriever, question, max_k)

        rewritten_query, search_query, retrieval_mode = extract_query_information(
            raw_chunks=raw_chunks,
            original_question=question,
        )

        chunks = [
            normalize_result(raw, rank, page_offset)
            for rank, raw in enumerate(raw_chunks, start=1)
        ]

        first_rank = next(
            (
                chunk.rank
                for chunk in chunks
                if is_relevant(
                    chunk,
                    item["expected_document"],
                    set(expected_pages),
                )
            ),
            None,
        )

        hits = {
            f"hit@{k}": first_rank is not None and first_rank <= k
            for k in top_ks
        }

        results.append(
            EvaluationResult(
                id=str(item["id"]),
                question=question,
                rewritten_query=rewritten_query,
                search_query=search_query,
                retrieval_mode=retrieval_mode,
                expected_document=str(item["expected_document"]),
                expected_pages=expected_pages,
                first_relevant_rank=first_rank,
                hits=hits,
                retrieved_chunks=chunks,
            )
        )

        status = f"rang {first_rank}" if first_rank else f"échec top {max_k}"
        mode_text = f" [{retrieval_mode}]" if retrieval_mode else ""
        print(f"[{index:02d}/{len(dataset):02d}] {item['id']}{mode_text} : {status}")

    metrics: dict[str, Any] = {
        "question_count": len(results),
        "page_offset": page_offset,
    }

    for k in top_ks:
        successes = sum(result.hits[f"hit@{k}"] for result in results)
        metrics[f"page_hit@{k}"] = {
            "successes": successes,
            "total": len(results),
            "score": successes / len(results) if results else 0.0,
        }

    metrics["page_mrr"] = (
        sum(
            1 / result.first_relevant_rank
            if result.first_relevant_rank is not None
            else 0.0
            for result in results
        )
        / len(results)
        if results
        else 0.0
    )

    return metrics, results


def save_report(
    output: Path,
    dataset: Path,
    retriever_path: str,
    metrics: dict[str, Any],
    results: list[EvaluationResult],
) -> None:
    """Enregistre un rapport reproductible et lisible par Git."""

    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": {
            "dataset": str(dataset),
            "retriever": retriever_path,
        },
        "metrics": metrics,
        "questions": [asdict(result) for result in results],
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Déclare l'interface de commande du script."""

    parser = argparse.ArgumentParser(
        description="Évaluation document + page d'un moteur de retrieval RAG."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/rag_evaluation_questions.json"),
    )
    parser.add_argument(
        "--retriever",
        help="Fonction sous la forme package.module:fonction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    parser.add_argument(
        "--page-offset",
        type=int,
        default=0,
        help="1 si les pages sont stockées à partir de 0, sinon 0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/retrieval_report.json"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée CLI."""

    args = parse_args()
    dataset = load_dataset(args.dataset)
    print(f"{len(dataset)} questions chargées depuis {args.dataset}")

    if args.validate_only:
        print("Le fichier JSON est valide.")
        return

    if not args.retriever:
        raise SystemExit(
            "Indique le retriever, par exemple :\n"
            "python -m src.evaluation.evaluate_retrieval "
            "--retriever src.retrieval.retriever:retrieve_relevant_chunks"
        )

    top_ks = sorted(set(args.top_k))
    if not top_ks or min(top_ks) < 1:
        raise ValueError("Les valeurs de --top-k doivent être positives.")

    retriever = import_retriever(args.retriever)
    metrics, results = evaluate(
        dataset=dataset,
        retriever=retriever,
        top_ks=top_ks,
        page_offset=args.page_offset,
    )

    print("\n=== Résultats globaux ===")
    for k in top_ks:
        metric = metrics[f"page_hit@{k}"]
        print(
            f"Page Hit@{k} : {metric['successes']}/{metric['total']} "
            f"({metric['score']:.1%})"
        )
    print(f"Page MRR   : {metrics['page_mrr']:.3f}")

    save_report(
        output=args.output,
        dataset=args.dataset,
        retriever_path=args.retriever,
        metrics=metrics,
        results=results,
    )
    print(f"\nRapport enregistré dans : {args.output}")


if __name__ == "__main__":
    main()
