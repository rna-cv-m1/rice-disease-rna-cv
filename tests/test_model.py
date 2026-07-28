import pytest
import torch
from src.config import CLASSES, IMAGE_SIZE
from src.nn.builder import cree_modele_pretrain, ARCHITECTURES_DISPONIBLES

def test_nombre_classes_config() -> None:
    """Vérifie que la configuration contient exactement les 6 classes de maladies cibles."""
    assert len(CLASSES) == 6
    assert "Healthy Rice Leaf" in CLASSES

@pytest.mark.parametrize("arch", ARCHITECTURES_DISPONIBLES)
def test_creation_architectures_modeles(arch: str) -> None:
    """Vérifie l'instanciation des 4 architectures modernes et la dimension du passe avant (forward pass)."""
    modele = cree_modele_pretrain(nb_classes=len(CLASSES), architecture=arch)
    assert isinstance(modele, torch.nn.Module)

    # Shape d'entrée : (Batch_size=1, Channels=3, Height=224, Width=224)
    tensor_test = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    output = modele(tensor_test)

    # Logits de shape : (Batch_size=1, Classes=6)
    assert output.shape == (1, 6)
