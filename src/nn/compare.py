import logging
import time
import torch
from pathlib import Path
from typing import Dict, Any, List

from src.config import MODELS_DIR
from src.dataset.loader import creer_dataloaders
from src.nn.builder import charger_meilleur_modele, ARCHITECTURES_DISPONIBLES

logger = logging.getLogger(__name__)

def comparer_modeles(architectures: List[str] = None, epochs: int = 5) -> Dict[str, Any]:
    """Évalue et compare plusieurs architectures, désignant le meilleur modèle enregistré sous models/<arch>.pth.

    Args:
        architectures (List[str]): Liste des architectures à comparer.
        epochs (int): Nombre d'époques pour entraîner les modèles manquants.

    Returns:
        Dict[str, Any]: Dictionnaire synthétique des métriques comparatives.
    """
    if architectures is None:
        architectures = ARCHITECTURES_DISPONIBLES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader_val, _, _ = creer_dataloaders()
    resultats = {}

    print("\n" + "=" * 75)
    print("   RAPPORT COMPARATIF DES ARCHITECTURES DE MODELES")
    print("=" * 75)

    meilleur_arch = None
    meilleure_acc = -1.0

    for arch in architectures:
        model_path = MODELS_DIR / f"{arch}.pth"

        # Si le modèle n'a pas encore été entraîné, le lancer
        if not model_path.exists():
            print(f"Entraînement initial pour l'architecture '{arch}'...")
            from src.nn.trainer import entrainer
            entrainer(epochs=epochs, architecture=arch, save_path=model_path)

        modele, nom_arch = charger_meilleur_modele(model_path, device=device.type)
        taille_mo = round(model_path.stat().st_size / (1024.0 * 1024.0), 2)

        # Mesure du temps d'inférence et calcul de la précision sur le jeu de validation
        modele.eval()
        corrects, total = 0, 0
        t0 = time.time()

        with torch.no_grad():
            for images, targets in loader_val:
                images, targets = images.to(device), targets.to(device)
                outputs = modele(images)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == targets).item()
                total += targets.size(0)

        t_inf = round((time.time() - t0) * 1000 / total, 2)
        acc = round((corrects / total) * 100.0, 2)

        resultats[arch] = {
            "Architecture": nom_arch.upper(),
            "Précision (%)": acc,
            "Temps d'inférence (ms/img)": t_inf,
            "Taille (Mo)": taille_mo
        }

        print(f" - {nom_arch.upper():<20} | Précision: {acc:6.2f}% | Inférence: {t_inf:5.2f} ms/img | Taille: {taille_mo} Mo")

        if acc > meilleure_acc:
            meilleure_acc = acc
            meilleur_arch = arch

    if meilleur_arch:
        print("=" * 75)
        print(f"MEILLEUR MODELE RETENU : {meilleur_arch.upper()} (Précision: {meilleure_acc}%)")
        print(f"Modèle disponible sous : models/{meilleur_arch}.pth")
        print("=" * 75)

    return resultats

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    comparer_modeles()
