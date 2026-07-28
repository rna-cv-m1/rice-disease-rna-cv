import logging
import torch
from pathlib import Path
from typing import Dict, Any, Union
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DEFAULT_MODEL_PATH
from src.dataset.loader import creer_dataloaders
from src.nn.builder import charger_meilleur_modele

logger = logging.getLogger(__name__)

def evaluer(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Évalue le modèle sur le jeu de validation et affiche le rapport de classification et la matrice de confusion.

    Args:
        model_path (Union[str, Path]): Chemin vers le fichier de poids .pth du modèle.

    Returns:
        Dict[str, Any]: Dictionnaire contenant les métriques calculées et les prédictions.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable à : {model_file}")

    # 1. Sélection automatique du GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Chargement des DataLoaders de validation (Shape lot : B, C=3, H=224, W=224)
    _, loader_val, classes, _ = creer_dataloaders()
    
    # 3. Chargement agnostique de l'architecture du modèle
    modele, arch = charger_meilleur_modele(model_file, device=device)

    toutes_preds = []
    toutes_vrais = []

    # 4. Inférence sur l'ensemble de validation sans calcul de gradient
    with torch.no_grad():
        for images, labels in loader_val:
            # images shape: (B, 3, 224, 224), labels shape: (B,)
            images = images.to(device)
            outputs = modele(images)            # outputs shape: (B, 6)
            _, preds = torch.max(outputs, 1)    # preds shape: (B,)

            toutes_preds.extend(preds.cpu().numpy())
            toutes_vrais.extend(labels.numpy())

    # 5. Affichage propre du rapport de performance scikit-learn
    print("\n" + "=" * 60)
    print(f"   RAPPORT D'ÉVALUATION DU MODÈLE ({arch.upper()})")
    print("=" * 60)
    report = classification_report(toutes_vrais, toutes_preds, target_names=classes)
    matrix = confusion_matrix(toutes_vrais, toutes_preds)
    
    print(report)
    print("Matrice de Confusion :")
    print(matrix)
    print("=" * 60)

    return {"report": report, "matrix": matrix}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluer()
