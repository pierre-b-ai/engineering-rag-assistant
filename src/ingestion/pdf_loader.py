from pathlib import Path
from pypdf import PdfReader


def load_pdf_text(pdf_path: str | Path) -> list[dict]:
    """
    Charge un PDF texte et retourne une liste de pages.
    """

    # Chemin PDF
    pdf_path = Path(pdf_path)

    # Lecture PDF
    reader = PdfReader(pdf_path)

    # Stockage pages
    pages = []

    # Parcours pages
    for page_number, page in enumerate(reader.pages, start=1):

        # Extraction texte
        text = page.extract_text()

        # Ignore pages vides
        if text and text.strip():

            # Ajout page
            pages.append(
                {
                    "text": text.strip(),
                    "metadata": {
                        "source": pdf_path.name,  # Nom fichier
                        "page": page_number,      # Numéro page
                    },
                }
            )

    # Résultat
    return pages