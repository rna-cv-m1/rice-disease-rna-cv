import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image

from src.config import RAW_DIR

# Configuration du logger pour l'analyse exploratoire
logger = logging.getLogger(__name__)

# Ensemble des extensions d'images valides pour une recherche rapide en O(1)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def analyser_dataset(data_dir: Union[str, Path] = RAW_DIR) -> pd.DataFrame:
    """Effectue une analyse exploratoire des données (EDA) sur les dimensions et la teinte HSV.

    Args:
        data_dir (Union[str, Path]): Répertoire racine des images (data/raw ou data/processed).

    Returns:
        pd.DataFrame: Tableau structuré contenant les statistiques de toutes les images.
    """
    # Validation de l'existence du dossier source
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Dossier introuvable : {data_path}")
        return pd.DataFrame()

    records = []
    print(f"\nAnalyse en cours : {data_path} ...")

    # Parcours arborescent récursif des répertoires de données
    for root, _, files in os.walk(data_path):
        # Le nom du dossier parent identifie la classe de la maladie
        classe = Path(root).name
        for f in files:
            # Filtrage rapide sur l'extension du fichier
            if Path(f).suffix.lower() not in _IMG_EXTS:
                continue
            file_path = Path(root) / f
            try:
                # Calcul de la taille du fichier sur disque en Kilo-octets (Ko)
                stat_size = file_path.stat().st_size / 1024.0
                # Ouverture sécurisée avec gestionnaire de contexte PIL
                with Image.open(file_path) as img:
                    w, h = img.size
                    mode = img.mode
                    # Conversion unique de l'image en espace colorimétrique HSV pour le calcul des statistiques
                    arr_hsv = np.array(img.convert("HSV"))
                
                # Enregistrement des données extraites sous forme de dictionnaire
                records.append({
                    "Classe": classe,
                    "Largeur": w,
                    "Hauteur": h,
                    "Ratio_Aspect": round(w / h, 2),
                    "Mode": mode,
                    "Taille_Ko": round(stat_size, 2),
                    "Teinte_Moy": round(float(arr_hsv[:, :, 0].mean()), 1),
                    "Saturation_Moy": round(float(arr_hsv[:, :, 1].mean()), 1),
                })
            except Exception as e:
                logger.warning(f"Erreur de lecture sur {f} : {e}")

    # Conversion de la liste de dictionnaires en DataFrame Pandas
    df = pd.DataFrame(records)
    if df.empty:
        print("[ATTENTION] Aucune image valide trouvée.")
        return df

    # ---------------------------------------------------------------------------
    # SYNTHÈSE DES RÉSULTATS STATISTIQUES (Format Texte Sans Emojis)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("   RAPPORT D'ANALYSE EXPLORATOIRE (EDA)")
    print("=" * 60)

    # 1. Distribution du nombre d'images par classe
    print(f"\n1. Distribution par classe ({len(df)} images) :")
    for cls, count in df["Classe"].value_counts().items():
        print(f"   {cls:<24} : {count:4d} ({count/len(df)*100:.1f}%)")

    # 2. Dimensions et modes de couleurs
    print(f"\n2. Dimensions des images :")
    print(f"   Largeur  min/max : {df['Largeur'].min()} / {df['Largeur'].max()} px")
    print(f"   Hauteur  min/max : {df['Hauteur'].min()} / {df['Hauteur'].max()} px")
    print(f"   Ratio aspect moy : {df['Ratio_Aspect'].mean():.2f}")
    print(f"   Modes couleur    : {dict(df['Mode'].value_counts())}")

    # 3. Teintes moyennes HSV par maladie (permet d'identifier les signatures visuelles)
    print(f"\n3. Colorimétrie HSV moyenne par classe :")
    print(df.groupby("Classe")[["Teinte_Moy", "Saturation_Moy"]].mean().to_string())
    print("=" * 60)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyser_dataset(RAW_DIR)
