def classify_query(query: str) -> str:
    """
    Classifie simplement le type de question utilisateur.

    Retourne :
    - numeric : question qui cherche une valeur chiffrée
    - procedure : question qui cherche une procédure
    - definition : question qui cherche une définition
    - general : cas général
    """

    q = query.lower()

    numeric_terms = [
        "maximum", "maximale", "minimum", "minimale",
        "combien", "valeur", "poids", "masse",
        "vitesse", "altitude", "hauteur", "distance",
        "portée", "autonomie", "durée", "température",
        "pression", "capacité", "puissance", "fréquence",
    ]

    procedure_terms = [
        "comment", "procédure", "étapes", "installer",
        "configurer", "activer", "désactiver", "brancher",
        "remplacer", "réinitialiser", "calibrer",
        "régler", "modifier",
    ]

    definition_terms = [
        "qu'est-ce", "c'est quoi", "définition",
        "signifie", "désigne", "correspond",
    ]

    if any(term in q for term in numeric_terms):
        return "numeric"

    if any(term in q for term in procedure_terms):
        return "procedure"

    if any(term in q for term in definition_terms):
        return "definition"

    return "general"