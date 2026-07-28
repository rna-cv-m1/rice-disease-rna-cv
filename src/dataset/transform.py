import logging
import os
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Union
from src.config import PROCESSED_DIR, IMAGE_SIZE, QUALITE_JPEG

logger = logging.getLogger(__name__)

def transformer_images(liste_images: List[Dict[str, Any]], output_dir: Union[str, Path] = PROCESSED_DIR) -> Dict[str, int]:
    """Recadre au centre et redimensionne les images brutes vers 224x224 RGB.

    Args:
        liste_images (List[Dict[str, Any]]): Liste des métadonnées d'images extraites par extract.py.
        output_dir (Union[str, Path]): Dossier de destination des images prétraitées (data/processed).

    Returns:
        Dict[str, int]: Statistiques sur le nombre de succès et d'échecs de transformation.
    """
    # 1. Préparation du répertoire de sortie des images prétraitées
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    succes, echecs = 0, 0

    # 2. Boucle de transformation sur chaque image du jeu de données brut
    for info in liste_images:
        # Création du sous-dossier correspondant à la classe (ex: 'data/processed/Brown Spot')
        classe_dir = output_path / info["classe"]
        classe_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 3. Ouverture de l'image source avec PIL et conversion explicite en 3 canaux RGB
            img = Image.open(info["chemin"])
            if img.mode != "RGB":
                img = img.convert("RGB")  # Assure un format d'image à 3 canaux de couleur (R, G, B)

            # 4. Découpage au carré central (Center Crop)
            # Permet d'isoler la zone principale de la feuille sans déformer le ratio hauteur/largeur
            w, h = img.size
            cote = min(w, h)
            img = img.crop(((w - cote) // 2, (h - cote) // 2, (w + cote) // 2, (h + cote) // 2))
            
            # 5. Redimensionnement standardisé aux dimensions cibles (224, 224) attendues par ImageNet
            img = img.resize(IMAGE_SIZE, Image.LANCZOS)

            # 6. Sauvegarde de l'image transformée au format JPEG optimisé
            dest_path = classe_dir / f"{Path(info['nom_fichier']).stem}.jpg"
            img.save(dest_path, format="JPEG", quality=QUALITE_JPEG, optimize=True)
            succes += 1

        except Exception as e:
            # Enregistrement de l'erreur si l'image est corrompue ou illisible
            logger.warning(f"Erreur transformation {info['nom_fichier']} : {e}")
            echecs += 1

    # 7. Affichage du bilan dans les logs applicatifs
    logger.info(f"Transformation terminée : {succes} succès, {echecs} échecs.")
    return {"succes": succes, "echecs": echecs}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.dataset.extract import extraire_infos_donnees
    transformer_images(extraire_infos_donnees())
