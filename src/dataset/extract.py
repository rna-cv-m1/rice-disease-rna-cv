import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union
from src.config import RAW_DIR, EXTENSIONS_VALIDE, CLASSES

logger = logging.getLogger(__name__)

def extraire_infos_donnees(raw_dir: Union[str, Path] = RAW_DIR) -> List[Dict[str, Any]]:
    """Parcourt le dossier des données brutes (y compris sous-dossiers intermédiaires) et extrait les métadonnées de chaque image.

    Args:
        raw_dir (Union[str, Path]): Chemin vers le dossier racine des images brutes (data/raw).

    Returns:
        List[Dict[str, Any]]: Liste de dictionnaires contenant le chemin, la classe et la taille en Ko.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"Dossier brut introuvable : {raw_path}")
        return []

    # Dictionnaire de correspondance insensible à la casse (ex: 'brown spot' -> 'Brown Spot')
    classes_map = {cls.lower().strip(): cls for cls in CLASSES}
    images_infos: List[Dict[str, Any]] = []

    # Parcours récursif pour détecter les sous-dossiers de classes à n'importe quel niveau de profondeur
    for root, dirs, files in os.walk(raw_path):
        folder_name = Path(root).name.lower().strip()

        # Si le nom du dossier courant correspond à l'une des 6 classes officielles
        if folder_name in classes_map:
            classe_officielle = classes_map[folder_name]

            for file_name in files:
                file_path = Path(root) / file_name
                if file_path.suffix.lower() in EXTENSIONS_VALIDE:
                    images_infos.append({
                        "classe": classe_officielle,
                        "nom_fichier": file_name,
                        "chemin": file_path,
                        "taille_ko": file_path.stat().st_size / 1024.0
                    })

    logger.info(f"Extraction terminée : {len(images_infos)} images trouvées.")
    return images_infos

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extraire_infos_donnees()
