import argparse
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Union

from src.config import DEFAULT_MODEL_PATH, CLASSES, IMAGE_SIZE, SEUIL_CONFIANCE_MIN
from src.dataset.verifier import verifier_est_feuille_riz
from src.nn.builder import charger_meilleur_modele

def predire_image(image_path: Union[str, Path], model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> None:
    """Effectue l'inférence et le diagnostic complet sur une image en ligne de commande.

    Args:
        image_path (Union[str, Path]): Chemin d'accès à l'image de feuille à analyser.
        model_path (Union[str, Path]): Chemin vers les poids .pth du modèle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")

    # 1. Vérification HSV : Filtre hors-domaine (OOD)
    est_feuille, ratio = verifier_est_feuille_riz(image)

    print("\n" + "=" * 60)
    print("   RÉSULTAT DE L'ANALYSE (CLI)")
    print("=" * 60)
    print(f"Image       : {Path(image_path).name}")

    if not est_feuille:
        print("Diagnostic  : Image rejetée (Aucune feuille végétale détectée)")
        print(f"Indice      : Couverture végétale HSV insuffisante ({ratio * 100:.1f}%)")
        print("=" * 60)
        return

    # 2. Chargement du modèle agnostique et prétraitement
    modele, arch = charger_meilleur_modele(model_path, device=device)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Shape entrée tensor : (Batch_size=1, Channels=3, Height=224, Width=224)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = modele(tensor)  # Logits de shape : (1, NB_CLASSES=6)
        probs = torch.softmax(outputs[0], dim=0)  # Probabilités de shape : (6,)

    top_prob, top_idx = torch.max(probs, 0)
    classe_predite = CLASSES[top_idx.item()]
    confiance = float(top_prob.item() * 100)

    # 3. Filtrage du seuil de confiance
    if confiance < SEUIL_CONFIANCE_MIN:
        print("Diagnostic  : Image non reconnue (Confiance insuffisante)")
        print(f"Indice      : Confiance {confiance:.2f}% < {SEUIL_CONFIANCE_MIN}%")
    else:
        print(f"Diagnostic  : {classe_predite}")
        print(f"Confiance   : {confiance:.2f}% ({arch.upper()})")
    print("=" * 60)


def main() -> None:
    """Point d'entrée du script CLI d'inférence."""
    parser = argparse.ArgumentParser(description="Diagnostic CLI de la maladie du riz")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image de la feuille de riz")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Chemin du modèle .pth")
    args = parser.parse_args()

    predire_image(args.image, Path(args.model))


if __name__ == "__main__":
    main()
