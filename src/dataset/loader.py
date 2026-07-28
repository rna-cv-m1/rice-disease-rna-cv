import logging
import os
from pathlib import Path
from typing import Tuple, List, Union

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from src.config import PROCESSED_DIR, BATCH_SIZE, IMAGE_SIZE, SEED, MEAN_IMAGENET, STD_IMAGENET

# Configuration du logger pour le suivi d'exécution du chargement de données
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PIPELINES DE TRANSFORMATION PYTORCH (Augmentation & Normalisation)
# ---------------------------------------------------------------------------

# 1. Pipeline d'augmentation de données pour l'entraînement (prévention du shortcut-learning)
transformations_train = transforms.Compose([
    # Zoom et rognage aléatoire sur la zone de la lésion (échantillonnage de 50% à 100% de la surface)
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
    # Miroir horizontal aléatoire (probabilité 50%)
    transforms.RandomHorizontalFlip(p=0.5),
    # Miroir vertical aléatoire (probabilité 30%)
    transforms.RandomVerticalFlip(p=0.3),
    # Rotation aléatoire de l'image dans l'intervalle [-15°, +15°]
    transforms.RandomRotation(degrees=15),
    # Variations légères de luminosité, contraste, saturation et teinte
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Conversion de l'image PIL en tenseur PyTorch (Shape : C=3, H=224, W=224, échelle [0, 1])
    transforms.ToTensor(),
    # Normalisation standard ImageNet (centrage et réduction par canal RGB)
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET),
])

# 2. Pipeline déterministe pour le jeu de validation
transformations_val = transforms.Compose([
    # Redimensionnement standardisé à (224, 224) sans augmentation aléatoire
    transforms.Resize(IMAGE_SIZE),
    # Conversion de l'image PIL en tenseur PyTorch
    transforms.ToTensor(),
    # Normalisation standard ImageNet
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET),
])


def creer_dataloaders(
    data_dir: Union[str, Path] = PROCESSED_DIR,
    batch_size: int = BATCH_SIZE,
    ratio_val: float = 0.2,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, List[str], int]:
    """Crée les DataLoaders PyTorch (Train et Val) avec rééquilibrage automatique des classes.

    Args:
        data_dir (Union[str, Path]): Chemin vers le répertoire data/processed.
        batch_size (int): Taille des lots d'images.
        ratio_val (float): Proportion d'images attribuée à la validation.
        seed (int): Graine aléatoire pour la reproductibilité du découpage.

    Returns:
        Tuple[DataLoader, DataLoader, List[str], int]: (loader_train, loader_val, noms_classes, nb_classes)
    """
    # Conversion du chemin en objet Path et vérification d'existence
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {data_path}")

    # Chargement du jeu de données complet avec les 2 pipelines de transformation distincts
    ds_train_full = datasets.ImageFolder(root=str(data_path), transform=transformations_train)
    ds_val_full = datasets.ImageFolder(root=str(data_path), transform=transformations_val)

    # Récupération de la liste officielle des noms de maladies
    noms_classes = ds_train_full.classes
    # Calcul du nombre total d'images et découpage train/val
    total = len(ds_train_full)
    taille_val = int(total * ratio_val)
    taille_train = total - taille_val

    # Génération d'une permutation aléatoire reproductible à partir de la graine (seed)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=gen).tolist()
    idx_train, idx_val = indices[:taille_train], indices[taille_train:]

    # Création des sous-ensembles d'entraînement et de validation
    ds_train = Subset(ds_train_full, idx_train)
    ds_val = Subset(ds_val_full, idx_val)

    # Calcul du sur-échantillonnage pondéré (WeightedRandomSampler) pour équilibrer les classes
    targets_train = [ds_train_full.targets[i] for i in idx_train]
    class_counts = torch.bincount(torch.tensor(targets_train))
    weights = 1.0 / class_counts.float()
    sample_weights = weights[targets_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Détection automatique du nombre optimal de threads processeurs (num_workers)
    n_workers = min(8, (os.cpu_count() or 2) // 2)
    # Activation du transfert mémoire direct (pin_memory) si un GPU est disponible
    pin = torch.cuda.is_available()

    # Instanciation du DataLoader d'entraînement avec sampler pondéré et chargement parallèle
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, sampler=sampler,
        num_workers=n_workers, pin_memory=pin, persistent_workers=(n_workers > 0),
    )
    # Instanciation du DataLoader de validation déterministe
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=pin, persistent_workers=(n_workers > 0),
    )

    # Journalisation d'information synthétique
    logger.info(f"DataLoaders créés : {taille_train} train, {taille_val} val, {len(noms_classes)} classes.")
    return loader_train, loader_val, noms_classes, len(noms_classes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    creer_dataloaders()
