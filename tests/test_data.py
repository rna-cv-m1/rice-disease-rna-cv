import pytest
import os
from PIL import Image
from src.config import CLASSES, IMAGE_SIZE, PROCESSED_DIR
from src.dataset.loader import transformations_train, transformations_val

def test_config_classes() -> None:
    """Vérifie la bonne initialisation de la liste des classes du dataset."""
    # 1. Contrôle du nombre de classes configurées
    assert len(CLASSES) == 6
    
    # 2. Contrôle de la classe saine
    assert "Healthy Rice Leaf" in CLASSES


def test_transformations_shape() -> None:
    """Vérifie que la transformation PIL -> Tenseur produit exactement les dimensions PyTorch (C=3, H=224, W=224)."""
    # 1. Création d'une image PIL factice de dimension quelconque (300, 400)
    img_test = Image.new("RGB", (300, 400))
    
    # 2. Application du pipeline de transformation de validation
    tensor = transformations_val(img_test)
    
    # 3. Vérification de la shape du tenseur résultant : (Channels=3, Height=224, Width=224)
    assert tensor.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
