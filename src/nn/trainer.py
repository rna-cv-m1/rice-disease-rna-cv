import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Union
from pathlib import Path

from src.config import MODELS_DIR, NB_EPOCHS, LEARNING_RATE, DEFAULT_ARCHITECTURE, LABEL_SMOOTHING
from src.dataset.loader import creer_dataloaders
from src.nn.builder import cree_modele_pretrain, ARCHITECTURES_DISPONIBLES

# Configuration du logger pour le suivi des métriques d'apprentissage
logger = logging.getLogger(__name__)


def entrainer(
    epochs: int = NB_EPOCHS,
    lr: float = LEARNING_RATE,
    architecture: str = DEFAULT_ARCHITECTURE,
    save_path: Union[str, Path] = None,
) -> Dict[str, Any]:
    """Exécute la boucle d'entraînement PyTorch avec AMP (précision mixte), AdamW et Cosine Annealing.

    Args:
        epochs (int): Nombre d'époques.
        lr (float): Taux d'apprentissage initial.
        architecture (str): Nom du modèle PyTorch à entraîner.
        save_path (Union[str, Path]): Chemin de destination du fichier .pth.

    Returns:
        Dict[str, Any]: Historique complet (train_loss, val_loss, val_acc, meilleure perte).
    """
    # Sélection automatique du processeur de calcul (GPU CUDA si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Activation de la précision mixte (AMP FP16) sur GPU pour accélérer les calculs par 2
    use_amp = device.type == "cuda"
    logger.info(f"Début de l'entraînement {architecture.upper()} sur {device} (AMP={use_amp})")

    # Définition du chemin de sauvegarde du modèle sous models/<architecture>.pth
    save_path = Path(save_path) if save_path else MODELS_DIR / f"{architecture}.pth"

    # Chargement des DataLoaders optimisés
    loader_train, loader_val, _, nb_classes = creer_dataloaders()
    # Instanciation et transfert du modèle sur le calculateur cible
    modele = cree_modele_pretrain(nb_classes=nb_classes, architecture=architecture).to(device)

    # Fonction de perte CrossEntropy avec lissage d'étiquettes (Label Smoothing = 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # Optimiseur AdamW (dÉcroissance des poids L2 régulière pour éviter le surapprentissage)
    optimizer = AdamW(filter(lambda p: p.requires_grad, modele.parameters()), lr=lr, weight_decay=1e-4)
    # Planificateur de taux d'apprentissage avec décroissance cosinoïdale
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    # Mise à l'échelle des gradients pour l'entraînement FP16
    scaler = GradScaler(enabled=use_amp)

    meilleure_perte = float("inf")
    # Dictionnaire d'enregistrement de l'historique
    history = {
        "architecture": architecture,
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "perte_val_min": 0.0,
        "precision_finale": 0.0,
        "model_path": save_path
    }

    # Boucle d'entraînement époque par époque
    for epoch in range(epochs):
        # ---------------------------------------------------------------------------
        # PHASE 1 : ENTRAÎNEMENT (Train Mode)
        # ---------------------------------------------------------------------------
        modele.train()
        perte_train = total_train = 0

        for images, targets in loader_train:
            # Transfert asynchrone non-bloquant des images et cibles vers le GPU
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # Réinitialisation rapide des gradients (set_to_none=True libère la mémoire)
            optimizer.zero_grad(set_to_none=True)

            # Context manager pour le calcul sous précision mixte (FP16/FP32)
            with autocast(enabled=use_amp):
                loss = criterion(modele(images), targets)

            # Rétropropagation des gradients mis à l'échelle
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            perte_train += loss.item() * images.size(0)
            total_train += targets.size(0)

        # Mise à jour du taux d'apprentissage (Cosine Annealing) à chaque fin d'époque
        scheduler.step()

        # ---------------------------------------------------------------------------
        # PHASE 2 : VALIDATION (Eval Mode)
        # ---------------------------------------------------------------------------
        modele.eval()
        perte_val = corrects = total_val = 0

        # Inférence déterministe sans calcul de gradient
        with torch.no_grad():
            for images, targets in loader_val:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    outputs = modele(images)
                perte_val += criterion(outputs, targets).item() * images.size(0)
                # Calcul rapide des prédictions exactes via argmax
                corrects += (outputs.argmax(1) == targets).sum().item()
                total_val += targets.size(0)

        # Calcul des moyennes d'époque
        train_loss = perte_train / total_train
        val_loss = perte_val / total_val
        val_acc = corrects / total_val * 100.0

        # Stockage dans l'historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(f"Époque {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Sauvegarde du meilleur modèle lorsque la perte de validation diminue
        if val_loss < meilleure_perte:
            meilleure_perte = val_loss
            history["perte_val_min"] = meilleure_perte
            history["precision_finale"] = val_acc
            
            checkpoint = {
                "arch": architecture,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history,
                "state_dict": modele.state_dict()
            }
            torch.save(checkpoint, save_path)
            logger.info(f"--> Meilleur modèle sauvegardé sous : {save_path}")

    return history


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Entraînement de modèle PyTorch")
    parser.add_argument("--arch", default=DEFAULT_ARCHITECTURE, choices=ARCHITECTURES_DISPONIBLES)
    parser.add_argument("--epochs", type=int, default=NB_EPOCHS)
    args = parser.parse_args()
    entrainer(epochs=args.epochs, architecture=args.arch)
