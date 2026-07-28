import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Dict, Any, Union
from pathlib import Path
from src.config import NB_CLASSES

def cree_modele_pretrain(nb_classes: int = NB_CLASSES, architecture: str = "resnet18") -> nn.Module:
    """Crée et initialise l'architecture du réseau de neurones pré-entraîné (ResNet18 ou EfficientNetV2-S).

    Args:
        nb_classes (int): Nombre de classes de maladies à prédire (ex: 6).
        architecture (str): Nom de l'architecture ('resnet18' ou 'efficientnet').

    Returns:
        nn.Module: Instance de modèle PyTorch initialisée.
    """
    if architecture.lower() == "efficientnet":
        # 1. Chargement du squelette EfficientNetV2-S pré-entraîné sur ImageNet (1000 classes)
        modele = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

        # 2. Récupération de la dimension du vecteur de caractéristiques de sortie : Shape (B, 1280)
        num_features = modele.classifier[1].in_features

        # 3. Remplacement du classifieur final pour s'adapter à nos 6 classes de riz
        # Entrée : (B, 1280) -> Sortie : (B, 6)
        modele.classifier = nn.Sequential(
            nn.Dropout(0.3),                 # Régularisation contre le sur-apprentissage
            nn.Linear(num_features, 256),    # Réduction intermédiaire : (B, 1280) -> (B, 256)
            nn.ReLU(),                       # Activation non-linéaire
            nn.Dropout(0.2),                 # Deuxième régularisation
            nn.Linear(256, nb_classes)       # Couche finale de logits : (B, 256) -> (B, nb_classes=6)
        )
    else:
        # 1. Chargement du squelette ResNet18 pré-entraîné sur ImageNet
        modele = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 2. Stratégie de Transfer Learning : Geler les couches initiales (lignes/contours de base)
        for param in modele.parameters():
            param.requires_grad = False

        # 3. Dégeler spécifiquement 'layer4' pour apprendre les textures complexes de lésions de riz
        for param in modele.layer4.parameters():
            param.requires_grad = True

        # 4. Remplacement de la couche fully connected (fc) de ResNet18
        # Entrée : (B, 512) -> Sortie : (B, 6)
        num_features = modele.fc.in_features
        modele.fc = nn.Sequential(
            nn.Linear(num_features, 256),    # Réduction intermédiaire : (B, 512) -> (B, 256)
            nn.ReLU(),                       # Activation non-linéaire
            nn.Dropout(0.3),                 # Dropout pour éviter le sur-apprentissage
            nn.Linear(256, nb_classes)       # Couche finale de logits : (B, 256) -> (B, nb_classes=6)
        )

    return modele


def charger_meilleur_modele(
    model_path: Union[str, Path],
    nb_classes: int = NB_CLASSES,
    device: str = "cpu"
) -> Tuple[nn.Module, str]:
    """Détecte agnostiquement l'architecture d'un fichier .pth et instancie le modèle correspondant.

    Args:
        model_path (Union[str, Path]): Chemin vers le fichier de poids PyTorch (.pth).
        nb_classes (int): Nombre de classes cibles (6).
        device (str): Calculateur cible ('cpu' ou 'cuda').

    Returns:
        Tuple[nn.Module, str]: Instance du modèle chargée et nom de l'architecture ('resnet18' ou 'efficientnet').
    """
    # 1. Chargement brut du dictionnaire de poids PyTorch depuis le disque
    state_dict = torch.load(model_path, map_location=device)

    # 2. Extraction des poids si le fichier provient d'un checkpoint complet d'entraînement
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        weights = state_dict["model_state_dict"]
    else:
        weights = state_dict

    # 3. Inspection des noms de couches pour détecter automatiquement l'architecture sans config explicite
    if any("features." in k or "classifier." in k for k in weights.keys()):
        arch = "efficientnet"
    else:
        arch = "resnet18"

    # 4. Construction de l'architecture et injection du dictionnaire de poids
    modele = cree_modele_pretrain(nb_classes=nb_classes, architecture=arch)
    modele.load_state_dict(weights)
    modele.to(device)
    modele.eval()

    return modele, arch
