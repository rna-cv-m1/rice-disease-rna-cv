import logging
import torch
from pathlib import Path
from typing import Dict, Any, Union, Optional
from sklearn.metrics import classification_report, confusion_matrix

from src.config import DEFAULT_MODEL_PATH
from src.dataset.loader import creer_dataloaders
from src.nn.builder import charger_meilleur_modele

logger = logging.getLogger(__name__)

def evaluer(model_path: Optional[Union[str, Path]] = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Évalue le modèle sur le jeu de validation et retourne le rapport texte, la matrice et les données brutes.

    Args:
        model_path (Optional[Union[str, Path]]): Chemin vers le fichier de poids .pth du modèle (si None, prend le meilleur).

    Returns:
        Dict[str, Any]: Dictionnaire contenant le rapport, la matrice, y_true, y_pred et classes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader_val, classes, _ = creer_dataloaders()
    modele, arch = charger_meilleur_modele(model_path, device=device.type)

    toutes_preds = []
    toutes_vrais = []

    with torch.no_grad():
        for images, labels in loader_val:
            images = images.to(device)
            outputs = modele(images)
            _, preds = torch.max(outputs, 1)

            toutes_preds.extend(preds.cpu().numpy())
            toutes_vrais.extend(labels.numpy())

    print("\n" + "=" * 60)
    print(f"   RAPPORT D'ÉVALUATION DU MODÈLE ({arch.upper()})")
    print("=" * 60)
    report_text = classification_report(toutes_vrais, toutes_preds, target_names=classes)
    report_dict = classification_report(toutes_vrais, toutes_preds, target_names=classes, output_dict=True)
    matrix = confusion_matrix(toutes_vrais, toutes_preds)
    
    print(report_text)
    print("Matrice de Confusion (Format Texte) :")
    print(matrix)
    print("=" * 60)

    return {
        "architecture": arch,
        "report_text": report_text,
        "report_dict": report_dict,
        "matrix": matrix,
        "y_true": toutes_vrais,
        "y_pred": toutes_preds,
        "classes": classes
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluer()
