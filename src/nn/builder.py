import pathlib
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Union, Optional, Any
from pathlib import Path

from src.config import NB_CLASSES, ARCHITECTURES_DISPONIBLES, DEFAULT_ARCHITECTURE, MODELS_DIR

# Autoriser la désérialisation de PosixPath / WindowsPath dans PyTorch 2.6+
if hasattr(torch.serialization, "add_safe_globals"):
    try:
        torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    except Exception:
        pass


def _load_checkpoint(path: Union[str, Path], map_location: Any = "cpu") -> Any:
    """Charge un fichier .pth avec gestion de la rétrocompatibilité PyTorch 2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def _detecter_architecture_depuis_weights(weights: Any, default_arch: str = DEFAULT_ARCHITECTURE) -> str:
    """Détecte dynamiquement l'architecture PyTorch sous-jacente d'après les clés du state_dict."""
    if not isinstance(weights, dict):
        return default_arch

    keys = set(weights.keys())

    # ResNet18 (conv1.weight, layer1.0..., fc.weight/fc.0.weight)
    if any(k.startswith("conv1.") or k.startswith("layer1.") or k.startswith("fc.") for k in keys):
        return "resnet18"

    # ConvNeXt-Tiny (features.0.0.block ou classifier.2)
    if any("features.0.0.block" in k or k.startswith("classifier.2.") for k in keys):
        return "convnext_tiny"

    # MobileNetV3-Small (classifier.3)
    if any(k.startswith("classifier.3.") for k in keys):
        return "mobilenet_v3_small"

    # EfficientNet-B0 (classifier.1)
    if any(k.startswith("classifier.1.") for k in keys):
        return "efficientnet_b0"

    return default_arch


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
            # Chargement sécurisé avec gestion des types PosixPath (PyTorch 2.6+)
            ckpt = _load_checkpoint(f, map_location="cpu")
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

    # Chargement sécurisé du checkpoint PyTorch (avec gestion de compatibilité PyTorch 2.6+)
    ckpt = _load_checkpoint(path, map_location=device)

    # Extraction des poids et du nom d'architecture à partir du dictionnaire de sauvegarde
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("arch", None)
        weights = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        arch = ckpt.get("arch", None)
        weights = ckpt["model_state_dict"]
    else:
        weights = ckpt
        arch = None

    # Auto-détection intelligente de l'architecture :
    # 1. 'arch' explicite dans le dictionnaire
    # 2. Nom du fichier s'il correspond à une architecture connue
    # 3. Détection par signature des clés de poids (weight keys fingerprint)
    if not arch or arch not in ARCHITECTURES_DISPONIBLES:
        if path.stem in ARCHITECTURES_DISPONIBLES:
            arch = path.stem
        else:
            arch = _detecter_architecture_depuis_weights(weights, default_arch=DEFAULT_ARCHITECTURE)

    # Instanciation de l'architecture correspondante et chargement des poids entraînés
    try:
        modele = cree_modele_pretrain(nb_classes=nb_classes, architecture=arch)
        modele.load_state_dict(weights)
    except Exception:
        # En cas d'incompatibilité de clés, forcer l'auto-détection et utiliser strict=False
        arch_recupere = _detecter_architecture_depuis_weights(weights, default_arch=DEFAULT_ARCHITECTURE)
        modele = cree_modele_pretrain(nb_classes=nb_classes, architecture=arch_recupere)
        modele.load_state_dict(weights, strict=False)
        arch = arch_recupere

    # Transfert vers le device (CPU/GPU) et passage strict en mode évaluation (.eval())
    modele.to(device).eval()

    return modele, arch
