import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union

from PIL import Image

from src.config import PROCESSED_DIR, IMAGE_SIZE, QUALITE_JPEG

# Initialisation du logger applicatif
logger = logging.getLogger(__name__)


def transformer_images(
    liste_images: List[Dict[str, Any]],
    output_dir: Union[str, Path] = PROCESSED_DIR,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Parcourt et transforme les images brutes : rognage carré au centre (Center Crop) et redimensionnement (224x224).

    Args:
        liste_images (List[Dict[str, Any]]): Liste des métadonnées extraites par extract.py.
        output_dir (Union[str, Path]): Répertoire de destination pour les images prétraitées.
        overwrite (bool): Si False, conserve les images déjà transformées sans les retraiter.

    Returns:
        Dict[str, int]: Bilan contenant le nombre de succès, échecs et images ignorées.
    """
    # Conversion du dossier cible en objet Path et création des répertoires si nécessaire
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    succes = echecs = ignores = 0

    # Parcours des métadonnées de chaque image du jeu de données brut
    for info in liste_images:
        # Création du sous-dossier de la maladie dans data/processed/<classe>/
        dest_dir = output_path / info["classe"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Définition du chemin cible de sauvegarde au format .jpg
        dest_path = dest_dir / f"{Path(info['nom_fichier']).stem}.jpg"

        # Optimisation : sauter le traitement si le fichier existe déjà et que overwrite=False
        if not overwrite and dest_path.exists():
            ignores += 1
            continue

        try:
            # Gestion mémoire propre avec le gestionnaire de contexte Image.open()
            with Image.open(info["chemin"]) as img:
                # Conversion explicite en 3 canaux de couleur RGB
                img = img.convert("RGB")
                w, h = img.size
                cote = min(w, h)
                # Découpage carré au centre pour isoler la feuille et éliminer le bruit de fond
                img = img.crop(((w - cote) // 2, (h - cote) // 2,
                                (w + cote) // 2, (h + cote) // 2))
                # Redimensionnement vers la taille standardisée (224, 224) avec filtre Lanczos haute qualité
                img = img.resize(IMAGE_SIZE, Image.LANCZOS)
                # Sauvegarde en format JPEG optimisé avec qualité configurée (85%)
                img.save(dest_path, format="JPEG", quality=QUALITE_JPEG, optimize=True)
            succes += 1
        except Exception as e:
            # Journalisation des avertissements en cas d'erreur de lecture de fichier
            logger.warning(f"Erreur transformation {info['nom_fichier']} : {e}")
            echecs += 1

    # Bilan récapitulatif dans les logs applicatifs
    logger.info(f"Transformation terminée : {succes} succès, {echecs} échecs, {ignores} ignorés.")
    return {"succes": succes, "echecs": echecs, "ignores": ignores}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.dataset.extract import extraire_infos_donnees
    transformer_images(extraire_infos_donnees())
