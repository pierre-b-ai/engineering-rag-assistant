"""Tests unitaires du contrôle qualité des textes extraits des PDF."""

from src.ingestion.text_quality import assess_text_quality


def test_none_text_is_rejected() -> None:
    result = assess_text_quality(None)

    assert result.is_valid is False
    assert result.status == "rejected"
    assert result.reason == "no_text"
    assert result.score == 0.0


def test_empty_text_is_rejected() -> None:
    result = assess_text_quality("   ")

    assert result.is_valid is False
    assert result.status == "rejected"
    assert result.reason == "no_text"


def test_text_too_short_is_rejected() -> None:
    result = assess_text_quality("Texte court.")

    assert result.is_valid is False
    assert result.status == "rejected"
    assert result.reason == "text_too_short"


def test_text_with_too_few_words_is_rejected() -> None:
    text = (
        "DocumentationTechniqueTresLongue "
        "SansAutresMots Exploitable"
    )

    result = assess_text_quality(text)

    assert result.is_valid is False
    assert result.status == "rejected"
    assert result.reason == "too_few_words"


def test_normal_french_text_is_clean() -> None:
    text = (
        "Avant de démarrer le lave-vaisselle, vérifiez que le robinet "
        "d'eau est ouvert et que le tuyau d'arrivée d'eau n'est pas plié. "
        "Sélectionnez ensuite le programme adapté à la vaisselle."
    )

    result = assess_text_quality(text)

    assert result.is_valid is True
    assert result.status == "clean"
    assert result.reason == "ok"
    assert result.score >= 0.90


def test_technical_text_with_acronyms_and_units_is_clean() -> None:
    text = (
        "GNSS GPS GLONASS et Galileo sont pris en charge. "
        "La fréquence de fonctionnement est comprise entre 2,4000 et "
        "2,4835 GHz, puis entre 5,725 et 5,850 GHz. "
        "La puissance de l'émetteur reste inférieure à 26 dBm selon "
        "les normes FCC, CE, SRRC et MIC. "
        "Le Wi-Fi utilise les protocoles 802.11 a/b/g/n/ac. "
        "La température de fonctionnement varie de -10 à 40 °C."
    )

    result = assess_text_quality(text)

    assert result.is_valid is True
    assert result.status == "clean"
    assert result.reason == "ok"
    assert result.score >= 0.90


def test_numeric_specification_table_is_clean() -> None:
    text = (
        "Tension nominale : 220-240 V. Fréquence : 50/60 Hz. "
        "Puissance absorbée : 1450 W. Pression maximale : 1,5 MPa, "
        "soit 15 bars. Capacité maximale du réservoir d'eau : "
        "1,8 litre. Classe de protection : IPX4."
    )

    result = assess_text_quality(text)

    assert result.is_valid is True
    assert result.status == "clean"
    assert result.reason == "ok"


def test_partially_corrupted_text_is_degraded() -> None:
    text = (
        "Réglage de la dureté de l'eau. Pour obtenir un fonctionnement "
        "correct de l'adoucisseur, sélectionnez le niveau correspondant. "
        r"HUVXUODWRXFKH SRXUPHWWUHOHDYHYDLVVHOOH "
        r"TXDQGOHYR\DQWDOOXPH SRXUVHOHFWLRQQHUOHQLYHDX "
        "Le produit de rinçage facilite le séchage de la vaisselle."
    )

    result = assess_text_quality(text)

    assert result.is_valid is True
    assert result.status == "degraded"
    assert result.reason == "partially_degraded_extraction"
    assert 0.0 < result.score < 0.90


def test_heavily_corrupted_text_is_rejected() -> None:
    corrupted_tokens = [
        "HUVXUODWRXFKH",
        "SRXUPHWWUHOHDYHYDLVVHOOH",
        r"TXDQGOHYR\DQWDOOXPH",
        "SRXUVHOHFWLRQQHUOHQLYHDX",
        "GRFXPHQWWHFKQLTXH",
        "IRQFWLRQQHPHQW",
        "SURJUDPPDWLRQ",
        "VHOHFWLRQQHU",
        "UHVHUYRLU",
        "DYHUWLVVHPHQW",
        "HQWUHWLHQ",
        "XWLOLVDWLRQ",
        "VHFXULWH",
        "DSSDUHLO",
        "FRQILJXUDWLRQ",
    ]

    text = " ".join(corrupted_tokens * 4)

    result = assess_text_quality(text)

    assert result.is_valid is False
    assert result.status == "rejected"
    assert result.reason in {
        "suspected_corrupted_encoding",
        "suspected_gibberish_text",
    }


def test_clean_text_keeps_default_reason() -> None:
    text = (
        "La machine doit être installée sur une surface stable et plane. "
        "Les pieds doivent être réglés correctement avant la première "
        "utilisation afin de limiter les vibrations pendant l'essorage."
    )

    result = assess_text_quality(text)

    assert result.is_valid is True
    assert result.reason == "ok"
    assert result.status == "clean"