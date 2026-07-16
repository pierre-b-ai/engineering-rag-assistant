"""Index lexical BM25 construit à partir des chunks présents dans Chroma.

Le corpus du projet reste stocké dans Chroma. Ce module relit simplement les
chunks indexés afin de construire, en mémoire, un index lexical complémentaire.
Aucune seconde base persistante n'est nécessaire pour cette première version.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass


# Le motif conserve les références techniques comme "SF70220-2", "IPX4" ou
# "0.05-MPa" sous forme de tokens uniques. Ces chaînes sont précisément les cas
# sur lesquels une recherche lexicale peut compléter la recherche dense.
_TOKEN_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[._/\-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*"
)


@dataclass(frozen=True)
class BM25SearchResult:
    """Résultat lexical retourné avant la fusion avec la recherche dense."""

    chunk: dict
    score: float
    rank: int


def tokenize_for_bm25(text: str) -> list[str]:
    """Normalise un texte en tokens adaptés aux notices techniques.

    ``casefold`` est utilisé plutôt que ``lower`` afin de gérer plus proprement
    les caractères accentués. Aucun stop-word n'est retiré : BM25 attribue déjà
    peu de poids aux termes présents dans presque tous les chunks.
    """

    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Index:
    """Implémentation légère de BM25 Okapi sans dépendance supplémentaire."""

    def __init__(
        self,
        chunks: list[dict],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.document_tokens = [
            tokenize_for_bm25(chunk.get("text", "")) for chunk in chunks
        ]
        self.term_frequencies = [Counter(tokens) for tokens in self.document_tokens]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.document_count = len(chunks)
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )

        # La fréquence documentaire compte combien de chunks contiennent un
        # terme, indépendamment du nombre de répétitions dans chaque chunk.
        self.document_frequencies: Counter[str] = Counter()
        for tokens in self.document_tokens:
            self.document_frequencies.update(set(tokens))

    def _inverse_document_frequency(self, term: str) -> float:
        """Calcule l'IDF BM25 avec une forme toujours positive et stable."""

        document_frequency = self.document_frequencies.get(term, 0)
        numerator = self.document_count - document_frequency + 0.5
        denominator = document_frequency + 0.5
        return math.log(1.0 + numerator / denominator)

    def score_document(self, query_tokens: list[str], document_index: int) -> float:
        """Calcule le score BM25 d'un chunk pour une requête tokenisée."""

        if not query_tokens or not self.document_count:
            return 0.0

        term_frequency = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]
        average_length = self.average_document_length or 1.0
        score = 0.0

        # Les occurrences répétées dans la requête ne doivent pas multiplier
        # artificiellement le poids d'un terme. On conserve donc l'ordre tout en
        # supprimant les doublons.
        unique_query_terms = list(dict.fromkeys(query_tokens))

        for term in unique_query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue

            idf = self._inverse_document_frequency(term)
            length_normalisation = 1.0 - self.b + self.b * (
                document_length / average_length
            )
            denominator = frequency + self.k1 * length_normalisation
            score += idf * (frequency * (self.k1 + 1.0)) / denominator

        return score

    def search(self, query: str, *, k: int) -> list[BM25SearchResult]:
        """Retourne les ``k`` chunks lexicaux les mieux classés."""

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens or k <= 0:
            return []

        scored_results: list[tuple[int, float]] = []
        for document_index in range(self.document_count):
            score = self.score_document(query_tokens, document_index)
            if score > 0.0:
                scored_results.append((document_index, score))

        scored_results.sort(key=lambda item: item[1], reverse=True)

        return [
            BM25SearchResult(
                chunk=self.chunks[document_index],
                score=score,
                rank=rank,
            )
            for rank, (document_index, score) in enumerate(
                scored_results[:k],
                start=1,
            )
        ]


_cached_signature: str | None = None
_cached_index: BM25Index | None = None


def _build_corpus_signature(chunks: list[dict]) -> str:
    """Produit une signature stable permettant d'invalider le cache BM25.

    Les identifiants Chroma sont basés sur le fichier, la page et le chunk. La
    longueur du texte est ajoutée afin de détecter aussi un changement de contenu
    qui conserverait par hasard le même identifiant.
    """

    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(str(chunk.get("id", "")).encode("utf-8"))
        digest.update(str(chunk.get("text", "")).encode("utf-8"))
    return digest.hexdigest()


def get_bm25_index() -> BM25Index:
    """Retourne l'index BM25 en cache ou le reconstruit si le corpus a changé."""

    global _cached_index, _cached_signature

    # Import local pour garder la classe BM25 testable sans charger Chroma.
    from src.vectorstore.chroma_store import get_indexed_chunks

    chunks = get_indexed_chunks()
    signature = _build_corpus_signature(chunks)

    if _cached_index is None or signature != _cached_signature:
        _cached_index = BM25Index(chunks)
        _cached_signature = signature

    return _cached_index


def clear_bm25_cache() -> None:
    """Force la reconstruction du BM25 lors de la prochaine recherche."""

    global _cached_index, _cached_signature
    _cached_index = None
    _cached_signature = None


def search_bm25(query: str, *, k: int) -> list[dict]:
    """Expose les résultats BM25 sous le même format général que le retriever."""

    index = get_bm25_index()
    results = index.search(query, k=k)

    return [
        {
            "id": result.chunk.get("id"),
            "text": result.chunk.get("text", ""),
            "metadata": result.chunk.get("metadata", {}),
            "bm25_score": result.score,
            "bm25_rank": result.rank,
        }
        for result in results
    ]
