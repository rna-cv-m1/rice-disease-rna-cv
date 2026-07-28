import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Union
from pathlib import Path

from src.config import MODELS_DIR, NB_EPOCHS, LEARNING_RATE, DEFAULT_MODEL_PATH, CLASSES
from src.dataset.loader import creer_dataloaders
from src.nn.builder import cree_modele_pretrain

logger = logging.getLogger(__name__)

def entrainer(
    epochs: int = NB_EPOCHS, 
    lr: float = LEARNING_RATE, 
    save_path: Union[str, Path] = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """Exécute la boucle d'entraînement PyTorch et enregistre le meilleur modèle (.pth).

    Args:
        epochs (int): Nombre total d'époques d'entraînement.
        lr (float): Taux d'apprentissage initial (Learning Rate).
        save_path (Union[str, Path]): Chemin de sauvegarde du fichier .pth final.

    Returns:
        Dict[str, Any]: Historique des pertes et précision finale.
    """
    # 1. Sélection automatique du matériel de calcul (GPU CUDA si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Début de l'entraînement sur le device : {device}")

    # 2. Initialisation des DataLoaders (Shape par lot : B, C=3, H=224, W=224)
    loader_train, loader_val, _, nb_classes = creer_dataloaders()
    
    # 3. Instanciation du modèle et transfert sur le processeur graphique/CPU
    modele = cree_modele_pretrain(nb_classes=nb_classes).to(device)

    # 4. Fonction de perte (CrossEntropyLoss avec Label Smoothing pour éviter la surconfiance)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 5. Optimiseur Adam uniquement sur les paramètres dégelés (requires_grad=True)
    optimizer = Adam(filter(lambda p: p.requires_grad, modele.parameters()), lr=lr)
    
    # 6. Planificateur de taux d'apprentissage Cosine Annealing pour une décroissance fluide
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    meilleure_perte = float('inf')

    # 7. Boucle principale à travers les époques
    for epoch in range(epochs):
        # ------------------- PHASE D'ENTRAÎNEMENT -------------------
        modele.train()
        perte_train = 0.0
        
        for images, targets in loader_train:
            # images shape: (B, 3, 224, 224), targets shape: (B,)
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()                 # Remise à zéro des gradients accumulés
            outputs = modele(images)              # Forward pass : outputs shape (B, 6)
            loss = criterion(outputs, targets)    # Calcul de la perte scalaire
            loss.backward()                       # Backward pass : calcul des gradients
            optimizer.step()                      # Mise à jour des poids du réseau

            perte_train += loss.item() * images.size(0)

        scheduler.step()  # Mise à jour du learning rate pour l'époque suivante

        # ------------------- PHASE DE VALIDATION -------------------
        modele.eval()
        perte_val = 0.0
        corrects = 0
        total = 0

        with torch.no_grad():  # Désactivation du calcul des gradients pour accélérer la validation
            for images, targets in loader_val:
                images, targets = images.to(device), targets.to(device)
                outputs = modele(images)          # outputs shape : (B, 6)
                loss = criterion(outputs, targets)
                perte_val += loss.item() * images.size(0)

                # Obtenir la classe ayant la plus forte probabilité
                _, preds = torch.max(outputs, 1)   # preds shape : (B,)
                corrects += torch.sum(preds == targets.data).item()
                total += targets.size(0)

        # 8. Calcul des métriques moyennes de l'époque
        perte_val_moy = perte_val / total
        precision_val = (corrects / total) * 100.0

        logger.info(f"Époque {epoch+1}/{epochs} - Perte Val: {perte_val_moy:.4f} - Précision Val: {precision_val:.2f}%")

        # 9. Sauvegarde du checkpoint si la perte de validation s'améliore (Best Model checkpoint)
        if perte_val_moy < meilleure_perte:
            meilleure_perte = perte_val_moy
            torch.save(modele.state_dict(), save_path)
            logger.info(f"--> Meilleur modèle sauvegardé dans : {save_path}")

    return {"perte_val_min": meilleure_perte, "precision_finale": precision_val}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    entrainer()
