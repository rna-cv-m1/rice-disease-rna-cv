# Script qui teste le modèle entraîné

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model_pretrain import cree_modele_pretrain

# ============================================================
# CONFIGURATION
# ============================================================

CHEMIN_PROPRE = os.path.join(
    os.path.dirname(__file__), "..", "data", "donne_propre"
)

CHEMIN_MODELE = os.path.join(
    os.path.dirname(__file__), "..", "models", "meilleur_modele.pth"
)

NB_CLASSES  = 6
TAILLE_BATCH = 32

# ============================================================
# TRANSFORMATIONS : aucune augmentation en évaluation
# ============================================================
transformations_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def evaluer():
    # --------------------------------------------------------
    # Étape 1 : Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 60)
    print("   ÉVALUATION DU MODÈLE")
    print("=" * 60)
    print(f"\n  Device : {device}")
    
    # --------------------------------------------------------
    # Étape 2 : Charger les données de test
    # On utilise donne_propre/ complet ici car on n'a pas
    # de dossier test séparé — dans un projet réel on aurait
    # data/test/ complètement séparé de data/train/
    # --------------------------------------------------------
    
    dataset_test = datasets.ImageFolder(
        root      = CHEMIN_PROPRE,
        transform = transformations_test
    )

    noms_classes = dataset_test.classes

    loader_test = DataLoader(
        dataset_test,
        batch_size  = TAILLE_BATCH,
        shuffle     = False,
        num_workers = 0
    )
    
    print(f"\n  Images à évaluer : {len(dataset_test)}")
    print(f"  Classes          : {noms_classes}")
    
    # --------------------------------------------------------
    # Étape 3 : Charger le modèle sauvegardé
    # On reconstruit l'architecture vide, puis on charge
    # les poids depuis meilleur_modele.pth par dessus
    # --------------------------------------------------------
    modele = cree_modele_pretrain(nb_classes=NB_CLASSES, geler_base=False)
    modele.load_state_dict(
        torch.load(CHEMIN_MODELE, map_location=device)
    )
    modele = modele.to(device)
    modele.eval()  # désactive Dropout et BatchNorm
    
    print(f"\n  Modèle chargé depuis : models/meilleur_modele.pth")
    
    # --------------------------------------------------------
    # Étape 4 : Prédictions sur tout le dataset
    # --------------------------------------------------------
    toutes_predictions = []
    toutes_etiquettes  = []
    
    with torch.no_grad():
        for images, etiquettes in loader_test:
            images     = images.to(device)
            etiquettes = etiquettes.to(device)

            predictions = modele(images)

            # torch.max retourne (valeur_max, indice_max)
            # l'indice = la classe prédite (0 à 5)
            _, predites = torch.max(predictions, 1)

            # On accumule toutes les prédictions et vraies étiquettes
            # .cpu() = ramener depuis GPU vers RAM pour scikit-learn
            toutes_predictions.extend(predites.cpu().tolist())
            toutes_etiquettes.extend(etiquettes.cpu().tolist())
    
    # --------------------------------------------------------
    # Étape 5 : Calcul des métriques
    # --------------------------------------------------------
    
    _afficher_resultats(
        toutes_etiquettes,
        toutes_predictions,
        noms_classes
    )

# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

def _afficher_resultats(etiquettes, predictions, noms_classes):
    """
    Affiche :
    - Précision globale
    - Précision par classe (precision, recall, f1-score)
    - Matrice de confusion
    """
    
    # Précision globale
    correct = sum(p == e for p, e in zip(predictions, etiquettes))
    total   = len(etiquettes)
    precision_globale = 100 * correct / total
    
    print("\n" + "=" * 60)
    print("   RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"\n  Précision globale : {precision_globale:.2f}%")
    print(f"  Correct           : {correct}/{total} images")
    
    # Rapport par classe : precision, recall, f1-score
    print("\n" + "-" * 60)
    print("  RAPPORT PAR CLASSE")
    print("-" * 60)
    rapport = classification_report(
        etiquettes,
        predictions,
        target_names=noms_classes,
        digits=3
    )
    print(rapport)
    
    # Matrice de confusion
    print("-" * 60)
    print("  MATRICE DE CONFUSION")
    print("  (ligne = vraie classe, colonne = classe prédite)")
    print("-" * 60)
    matrice = confusion_matrix(etiquettes, predictions)
    
    # Affichage lisible de la matrice
    noms_courts = [n[:10] for n in noms_classes]  # raccourcir les noms
    print(f"\n  {'':12}", end="")
    for nom in noms_courts:
        print(f"{nom:>12}", end="")
    print()
    print("  " + "-" * (12 + 12 * len(noms_classes)))

    for i, ligne in enumerate(matrice):
        print(f"  {noms_courts[i]:12}", end="")
        for val in ligne:
            print(f"{val:>12}", end="")
        print()

    print("\n" + "=" * 60 + "\n")
    
# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    evaluer()