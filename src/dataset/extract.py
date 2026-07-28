import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union
from src.config import RAW_DIR, EXTENSIONS_VALIDE

logger = logging.getLogger(__name__)

def extraire_infos_donnees(raw_dir: Union[str, Path] = RAW_DIR) -> List[Dict[str, Any]]:
    """Parcourt le dossier des données brutes et extrait les métadonnées de chaque image.

    Args:
        raw_dir (Union[str, Path]): Chemin vers le dossier racine des images brutes (data/raw).

    Returns:
        List[Dict[str, Any]]: Liste de dictionnaires contenant le chemin, la classe et la taille en Ko.
    """
    # 1. Conversion du chemin en objet Path et vérification d'existence du dossier racine data/raw
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"Dossier brut introuvable : {raw_path}")
        return []

    images_infos: List[Dict[str, Any]] = []

    # 2. Parcours trié par ordre alphabétique des sous-dossiers de classes (ex: 'Brown Spot', 'Leaf Blast')
    for classe in sorted(os.listdir(raw_path)):
        chemin_classe = raw_path / classe
        
        # Ignorer les fichiers isolés à la racine du dossier brut et ne traiter que les répertoires de classes
        if not chemin_classe.is_dir():
            continue

        # 3. Parcours des images à l'intérieur du dossier de la classe
        for file_name in os.listdir(chemin_classe):
            file_path = chemin_classe / file_name
            
            # 4. Filtrage strict sur les extensions d'images valides (.jpg, .jpeg, .png, .bmp, .tiff)
            if file_path.suffix.lower() in EXTENSIONS_VALIDE:
                # 5. Extraction et stockage des métadonnées sous forme de dictionnaire typé
                images_infos.append({
                    "classe": classe,                                # Nom de la maladie / classe
                    "nom_fichier": file_name,                        # Nom d'origine de l'image
                    "chemin": file_path,                             # Chemin d'accès absolu/relatif vers le fichier
                    "taille_ko": file_path.stat().st_size / 1024.0   # Conversion de la taille binaire en Kilo-octets (Ko)
                })

    # 6. Logging d'information indiquant le nombre total d'images valides recensées
    logger.info(f"Extraction terminée : {len(images_infos)} images trouvées.")
    return images_infos

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extraire_infos_donnees()
