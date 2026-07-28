import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Union, Optional
from pathlib import Path

from src.config import NB_CLASSES, ARCHITECTURES_DISPONIBLES, DEFAULT_ARCHITECTURE, MODELS_DIR


def cree_modele_pretrain(nb_classes: int = NB_CLASSES, architecture: str = DEFAULT_ARCHITECTURE) -> nn.Module:
    """Instancie le réseau de neurones pré-entraîné et remplace sa couche finale de classification.

    Architectures supportées : efficientnet_b0, resnet18, mobilenet_v3_small, convnext_tiny.

    Args:
        nb_classes (int): Nombre de classes de sortie (6).
        architecture (str): Nom du modèle souhaité.

    Returns:
        nn.Module: Modèle PyTorch configuré.
    """
    # Normalisation du nom de l'architecture en minuscules
    arch = architecture.lower().strip()

    # Case A : EfficientNet-B0 (poids par défaut pré-entraînés sur ImageNet)
    if arch in ("efficientnet", "efficientnet_b0"):
        modele = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Remplacement de la couche de classification linéaire (classifier[1])
        modele.classifier[1] = nn.Linear(modele.classifier[1].in_features, nb_classes)

    # Case B : MobileNet-V3 Small (optimisé pour la légèreté et la vitesse)
    elif arch in ("mobilenet", "mobilenet_v3_small"):
        modele = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Remplacement de la couche dense finale (classifier[3])
        modele.classifier[3] = nn.Linear(modele.classifier[3].in_features, nb_classes)

    # Case C : ConvNeXt-Tiny (architecture moderne à convolutions basées sur Vision Transformers)
    elif arch in ("convnext", "convnext_tiny"):
        modele = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        # Remplacement de la couche finale de sortie (classifier[2])
        modele.classifier[2] = nn.Linear(modele.classifier[2].in_features, nb_classes)

    # Case D : ResNet18 (architecture résiduelle classique)
    elif arch == "resnet18":
        modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remplacement de la couche fully connected finale (fc)
        modele.fc = nn.Linear(modele.fc.in_features, nb_classes)

    else:
        raise ValueError(f"Architecture non supportée : '{architecture}'. Choisir parmi : {ARCHITECTURES_DISPONIBLES}")

    return modele


def obtenir_chemin_meilleur_modele(models_dir: Union[str, Path] = MODELS_DIR) -> Path:
    """Parcourt le dossier models/ et retourne le chemin du fichier .pth ayant la meilleure précision (val_acc)."""
    m_dir = Path(models_dir)
    pth_files = list(m_dir.glob("*.pth"))

    # Si aucun modèle entraîné n'est trouvé, retourner le chemin par défaut
    if not pth_files:
        return m_dir / f"{DEFAULT_ARCHITECTURE}.pth"

    meilleur_f, meilleure_acc = pth_files[0], -1.0
    for f in pth_files:
        try:
            # Chargement sécurisé avec weights_only=True
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
            acc = float(ckpt.get("val_acc", -1)) if isinstance(ckpt, dict) else -1.0
            if acc > meilleure_acc:
                meilleure_acc, meilleur_f = acc, f
        except Exception:
            continue

    return meilleur_f


def charger_meilleur_modele(
    model_path: Optional[Union[str, Path]] = None,
    nb_classes: int = NB_CLASSES,
    device: str = "cpu",
) -> Tuple[nn.Module, str]:
    """Charge les poids d'un checkpoint .pth et retourne l'instance du modèle en mode éval avec son nom.

    Args:
        model_path (Optional[Union[str, Path]]): Chemin optionnel vers le fichier .pth.
        nb_classes (int): Nombre de classes de sortie.
        device (str): Calculateur cible ('cpu' ou 'cuda').

    Returns:
        Tuple[nn.Module, str]: (modele, nom_architecture)
    """
    # Auto-détection du meilleur modèle si model_path est None ou invalide
    path = (
        obtenir_chemin_meilleur_modele()
        if model_path is None or not Path(model_path).exists()
        else Path(model_path)
    )

    # Si aucun fichier n'existe sur le disque, instancier un modèle pré-entraîné viierge
    if not path.exists():
        modele = cree_modele_pretrain(nb_classes=nb_classes, architecture=DEFAULT_ARCHITECTURE)
        modele.to(device).eval()
        return modele, DEFAULT_ARCHITECTURE

    # Chargement sécurisé du checkpoint PyTorch (weights_only=True prévient les vulnérabilités Pickle)
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # Extraction des poids et du nom d'architecture à partir du dictionnaire de sauvegarde
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("arch", path.stem)
        weights = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        arch = ckpt.get("arch", path.stem)
        weights = ckpt["model_state_dict"]
    else:
        weights = ckpt
        arch = path.stem if path.stem in ARCHITECTURES_DISPONIBLES else DEFAULT_ARCHITECTURE

    # Instanciation de l'architecture correspondante et chargement des poids entraînés
    modele = cree_modele_pretrain(nb_classes=nb_classes, architecture=arch)
    modele.load_state_dict(weights)
    # Transfert vers le device (CPU/GPU) et passage strict en mode évaluation (.eval())
    modele.to(device).eval()

    return modele, arch
