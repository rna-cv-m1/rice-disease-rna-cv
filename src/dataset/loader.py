import logging
import os
from pathlib import Path
from typing import Tuple, List, Union
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.config import PROCESSED_DIR, BATCH_SIZE, IMAGE_SIZE, SEED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PIPELINES DE TRANSFORMATION PYTORCH
# ---------------------------------------------------------------------------

# 1. Pipeline d'augmentation de données pour le jeu d'entraînement (Train)
# Applique des perturbations géométriques et photométriques pour rendre le modèle robuste
transformations_train = transforms.Compose([
    transforms.RandomRotation(degrees=15),                                # Rotation aléatoire [-15°, +15°]
    transforms.RandomHorizontalFlip(p=0.5),                               # Miroir horizontal (50% de chance)
    transforms.RandomVerticalFlip(p=0.3),                                 # Miroir vertical (30% de chance)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Variations d'éclairage/couleur
    transforms.Resize(IMAGE_SIZE),                                        # Redimensionnement vers (224, 224)
    transforms.ToTensor(),                                                # Tenseur de shape : (C=3, H=224, W=224)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalisation standard ImageNet
])

# 2. Pipeline déterministe pour le jeu de validation (Val)
# Applique uniquement le redimensionnement et la normalisation sans altération aléatoire
transformations_val = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                                        # Redimensionnement vers (224, 224)
    transforms.ToTensor(),                                                # Tenseur de shape : (C=3, H=224, W=224)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalisation standard ImageNet
])

def creer_dataloaders(
    data_dir: Union[str, Path] = PROCESSED_DIR, 
    batch_size: int = BATCH_SIZE, 
    ratio_val: float = 0.2, 
    seed: int = SEED
) -> Tuple[DataLoader, DataLoader, List[str], int]:
    """Crée et retourne les DataLoaders PyTorch (Train et Val) avec augmentation de données.

    Args:
        data_dir (Union[str, Path]): Chemin vers le dossier des images structurées par classe (data/processed).
        batch_size (int): Taille des lots d'images (Batch Size).
        ratio_val (float): Proportion d'images réservées à la validation (ex: 0.2 pour 20%).
        seed (int): Graine aléatoire pour la reproductibilité de la séparation train/val.

    Returns:
        Tuple[DataLoader, DataLoader, List[str], int]: 
            - DataLoader d'entraînement (Shape par lot : B, C=3, H=224, W=224)
            - DataLoader de validation (Shape par lot : B, C=3, H=224, W=224)
            - Liste des noms de classes
            - Nombre total de classes
    """
    # 1. Vérification de l'existence du répertoire de données prétraitées
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Dossier de données introuvable : {data_path}")
        raise FileNotFoundError(f"Dossier introuvable : {data_path}")

    # 2. Chargement des jeux de données avec ImageFolder
    dataset_train_complet = datasets.ImageFolder(root=str(data_path), transform=transformations_train)
    dataset_val_complet = datasets.ImageFolder(root=str(data_path), transform=transformations_val)

    # 3. Extraction de la liste des classes et calcul des tailles de découpage
    noms_classes = dataset_train_complet.classes
    total = len(dataset_train_complet)
    taille_val = int(total * ratio_val)
    taille_train = total - taille_val

    # 4. Génération d'une permutation aléatoire reproductible des indices d'images
    generateur = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generateur).tolist()

    # 5. Création des sous-ensembles d'entraînement et de validation
    dataset_train = Subset(dataset_train_complet, indices[:taille_train])
    dataset_val = Subset(dataset_val_complet, indices[taille_train:])

    # 6. Instanciation des DataLoaders PyTorch pour le chargement par lots
    # Shape du lot produit à chaque itération : (Batch_size, Channels=3, Height=224, Width=224)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"DataLoaders créés : {taille_train} train, {taille_val} val, {len(noms_classes)} classes.")
    return loader_train, loader_val, noms_classes, len(noms_classes)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    creer_dataloaders()