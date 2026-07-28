import numpy as np
from PIL import Image
from typing import Tuple
from src.config import SEUIL_PIXELS_FEUILLE_MIN

def verifier_est_feuille_riz(image_pil: Image.Image) -> Tuple[bool, float]:
    """Effectue une vérification couleur/végétale en espace HSV pour écarter les visuels non-feuilles.

    Args:
        image_pil (Image.Image): Image d'entrée d'origine PIL.

    Returns:
        Tuple[bool, float]: 
            - bool: True si la couverture végétale est suffisante, False sinon.
            - float: Proportion de pixels végétaux détectés dans l'image [0.0, 1.0].
    """
    # 1. Conversion de l'image PIL en espace colorimétrique HSV (Hue, Saturation, Value)
    img_hsv = image_pil.convert("HSV")
    hsv_array = np.array(img_hsv)  # Tenseur NumPy de shape (Height, Width, Channels=3)

    # 2. Séparation des 3 canaux couleur
    h = hsv_array[:, :, 0]  # Canal Teinte Hue (0..255)
    s = hsv_array[:, :, 1]  # Canal Saturation (0..255)
    v = hsv_array[:, :, 2]  # Canal Valeur / Luminosité (0..255)

    # Conversion RGB pour vérifier la prédominance du canal vert végétal (G > B)
    rgb_array = np.array(image_pil.convert("RGB"))
    r, g, b_channel = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    masque_foliage_rgb = (g > b_channel) & (g > 0.45 * r)

    # 3. Définition des règles logiques pour le masque végétal & les lésions de maladies
    # - Verts de feuille saine : Teinte h entre 30 et 100 avec saturation s>=35 et v>=40
    # - Bruns/Jaunes de lésions : Teinte h entre 18 et 30 avec saturation s>=40 et v>=35
    masque_vert = (h >= 30) & (h <= 105) & (s >= 35) & (v >= 40)
    masque_brun = (h >= 18) & (h < 30) & (s >= 40) & (v >= 35)
    masque_vegetal = (masque_vert | masque_brun) & masque_foliage_rgb  # Masque booléen final

    # 4. Calcul du ratio de pixels végétaux par rapport au nombre total de pixels de l'image
    ratio_vegetal = float(np.mean(masque_vegetal))

    # 5. Évaluation par rapport au seuil minimal défini dans src/config.py (10% par défaut)
    est_conforme = ratio_vegetal >= SEUIL_PIXELS_FEUILLE_MIN
    return est_conforme, ratio_vegetal
