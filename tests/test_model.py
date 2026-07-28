import pytest
import torch
from src.config import CLASSES, IMAGE_SIZE
from src.nn.builder import cree_modele_pretrain

def test_nombre_classes_config() -> None:
    """Vérifie que la configuration contient exactement les 6 classes de maladies cibles."""
    # 1. Validation du nombre total de classes
    assert len(CLASSES) == 6
    
    # 2. Validation de la présence de la classe saine de référence
    assert "Healthy Rice Leaf" in CLASSES


def test_creation_modele() -> None:
    """Vérifie l'instanciation de l'architecture PyTorch et la dimension de sortie (forward pass)."""
    # 1. Instanciation du modèle pré-entraîné
    modele = cree_modele_pretrain(nb_classes=len(CLASSES))
    assert isinstance(modele, torch.nn.Module)
    
    # 2. Test d'une passe avant (Forward Pass) avec un tenseur aléatoire d'entrée
    # Shape d'entrée factice : (Batch_size=1, Channels=3, Height=224, Width=224)
    tensor_test = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    output = modele(tensor_test)
    
    # 3. Validation de la dimension de sortie (Logits de shape : Batch_size=1, Classes=6)
    assert output.shape == (1, 6)
