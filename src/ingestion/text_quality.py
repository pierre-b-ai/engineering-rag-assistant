"""Contrôle prudent de la qualité du texte extrait des PDF.

La qualité documentaire reste informative :

- clean : texte exploitable ;
- degraded : texte partiellement corrompu, mais encore utile ;
- rejected : texte manifestement inutilisable.

Les textes dégradés restent indexés et ne sont pas pénalisés lors du retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextQualityResult:
    """Résultat de l'analyse de qualité d'un texte extrait."""

    is_valid: bool
    reason: str = "ok"
    score: float = 1.0
    status: str = "clean"  # clean | degraded | rejected


_VOWELS = set("aeiouyàâäéèêëîïôöùûüÿæœ")


def _is_suspicious_word(word: str) -> bool:
    """Détecte une longue chaîne alphabétique probablement corrompue.

    Les nombres, unités, références et acronymes courts ne sont pas considérés
    comme suspects.
    """

    letters = "".join(character for character in word if character.isalpha())

    # Les acronymes et unités techniques courts restent autorisés.
    if len(letters) < 12:
        return False

    vowel_count = sum(
        character.lower() in _VOWELS
        for character in letters
    )
    vowel_ratio = vowel_count / len(letters)

    uppercase_count = sum(character.isupper() for character in letters)
    uppercase_ratio = uppercase_count / len(letters)

    # Les mauvais mappings de police produisent souvent de longues chaînes
    # majoritairement en majuscules et relativement pauvres en voyelles.
    return vowel_ratio < 0.30 and uppercase_ratio > 0.65


def assess_text_quality(text: str | None) -> TextQualityResult:
    """Évalue la qualité d'une extraction textuelle.

    Le contrôle évite de pénaliser les contenus techniques normaux contenant
    des unités, des nombres, des acronymes ou des références.
    """

    if text is None or not text.strip():
        return TextQualityResult(
            is_valid=False,
            reason="no_text",
            score=0.0,
            status="rejected",
        )

    cleaned = " ".join(text.split())

    if len(cleaned) < 40:
        return TextQualityResult(
            is_valid=False,
            reason="text_too_short",
            score=0.1,
            status="rejected",
        )

    words = re.findall(
        r"[^\W_]+(?:[-'][^\W_]+)*",
        cleaned,
        flags=re.UNICODE,
    )

    if len(words) < 5:
        return TextQualityResult(
            is_valid=False,
            reason="too_few_words",
            score=0.1,
            status="rejected",
        )

    suspicious_words = [
        word
        for word in words
        if _is_suspicious_word(word)
    ]

    suspicious_word_ratio = len(suspicious_words) / len(words)

    suspicious_character_count = sum(
        len(word)
        for word in suspicious_words
    )
    suspicious_character_ratio = (
        suspicious_character_count / max(len(cleaned), 1)
    )

    # Séquences souvent rencontrées lorsque le mapping de caractères du PDF
    # est cassé. Les unités et références techniques courtes ne correspondent
    # normalement pas à ce motif.
    corrupted_runs = re.findall(
        r"(?<!\w)"
        r"(?=[A-Z0-9\\/()]{14,}(?!\w))"
        r"(?=[A-Z0-9\\/()]*[A-Z]{8})"
        r"[A-Z0-9\\/()]+",
        cleaned,
    )

    # Rejet seulement lorsque la corruption domine réellement le contenu.
    if (
        suspicious_word_ratio > 0.30
        and suspicious_character_ratio > 0.35
    ):
        return TextQualityResult(
            is_valid=False,
            reason="suspected_corrupted_encoding",
            score=0.0,
            status="rejected",
        )

    if (
        len(corrupted_runs) >= 8
        and suspicious_character_ratio > 0.25
    ):
        return TextQualityResult(
            is_valid=False,
            reason="suspected_gibberish_text",
            score=0.0,
            status="rejected",
        )

    # Score informatif uniquement : il ne modifie pas le classement sémantique.
    quality_score = max(
        0.0,
        1.0
        - min(suspicious_word_ratio * 1.5, 0.45)
        - min(suspicious_character_ratio, 0.35),
    )
    quality_score = round(quality_score, 3)

    if quality_score >= 0.90:
        return TextQualityResult(
            is_valid=True,
            reason="ok",
            score=quality_score,
            status="clean",
        )

    if (
        suspicious_word_ratio >= 0.03
        or suspicious_character_ratio >= 0.08
        or len(corrupted_runs) >= 2
    ):
        return TextQualityResult(
            is_valid=True,
            reason="partially_degraded_extraction",
            score=quality_score,
            status="degraded",
        )

    return TextQualityResult(
        is_valid=True,
        reason="ok",
        score=quality_score,
        status="clean",
    )