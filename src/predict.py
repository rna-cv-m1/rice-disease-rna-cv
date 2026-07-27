# fichier de prediction qui est la version PROD DE NOTRE PROGRAMME

# src/predict.py

import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model_pretrain import cree_modele_pretrain

# ============================================================
# CONFIGURATION
# ============================================================

CHEMIN_MODELE = os.path.join(
    os.path.dirname(__file__), "..", "models", "meilleur_modele.pth"
)

NB_CLASSES   = 6
NOMS_CLASSES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
]

# Seuil de confiance minimum pour accepter une prédiction
# en dessous de 60%, le modèle dira "pas sûr"

SEUIL_CONFIANCE = 0.45

# ============================================================
# TRANSFORMATION : identique à la validation
# ============================================================

transformation = transforms.Compose([
    transforms.Resize((224, 224)),     # redimensionner comme pendant l'entraînement
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

def charger_modele(device):
    """
    Reconstruit l'architecture et charge les poids sauvegardés.
    """
    modele = cree_modele_pretrain(nb_classes=NB_CLASSES, geler_base=False)
    modele.load_state_dict(
        torch.load(CHEMIN_MODELE, map_location=device)
    )
    modele = modele.to(device)
    modele.eval()  # mode évaluation : désactive Dropout
    return modele

# ============================================================
# PRÉDICTION D'UNE IMAGE
# ============================================================

def predire_image(chemin_image, modele, device):
    """
    Prend le chemin d'une image, retourne la classe
    prédite et le niveau de confiance.

    Paramètres :
        chemin_image (str)       : chemin vers l'image choisie
        modele       (nn.Module) : modèle chargé
        device       (torch.device)

    Retourne :
        classe_predite (str)  : nom de la maladie détectée
        confiance      (float): pourcentage de confiance (0-100)
        top3           (list) : top 3 des prédictions
    """
    
    # --------------------------------------------------------
    # Étape 1 : Ouvrir et préparer l'image
    # --------------------------------------------------------
    
    img = Image.open(chemin_image).convert("RGB")
    img_tensor = transformation(img)
    
    # Ajouter une dimension batch : [3, 224, 224] → [1, 3, 224, 224]
    # Le modèle attend toujours un batch, même pour une seule image
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # --------------------------------------------------------
    # Étape 2 : Prédiction
    # --------------------------------------------------------
    
    with torch.no_grad():
        sortie = modele(img_tensor)  # [1, 6] scores bruts

        # Softmax convertit les scores bruts en probabilités [0, 1]
        # dont la somme = 1 (100%)
        probabilites = torch.softmax(sortie, dim=1)[0]
        
    # --------------------------------------------------------
    # Étape 3 : Extraire les résultats
    # --------------------------------------------------------
    # Classe avec la probabilité la plus haute
    
    confiance, indice = torch.max(probabilites, 0)
    confiance      = confiance.item() * 100
    classe_predite = NOMS_CLASSES[indice.item()]

    # Top 3 des prédictions
    top3_valeurs, top3_indices = torch.topk(probabilites, 3)
    top3 = [
        (NOMS_CLASSES[idx.item()], round(val.item() * 100, 2))
        for val, idx in zip(top3_valeurs, top3_indices)
    ]

    return classe_predite, round(confiance, 2), top3

# ============================================================
# SÉLECTION DU FICHIER VIA FENÊTRE TKINTER
# ============================================================

def choisir_image():
    """
    Ouvre une fenêtre de sélection de fichier.
    Retourne le chemin de l'image choisie.
    """
    # Initialiser tkinter sans afficher la fenêtre principale
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale vide

    chemin = filedialog.askopenfilename(
        title="Choisir une image de feuille de riz",
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Tous les fichiers", "*.*")
        ]
    )

    root.destroy()  # ferme proprement tkinter après sélection
    return chemin

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def predire():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("   DÉTECTION DE MALADIE - FEUILLE DE RIZ")
    print("=" * 60)
    print(f"\n  Device : {device}")
    print(f"\n  Chargement du modèle...")
    
    modele = charger_modele(device)
    print(f"  Modèle prêt ✓")

    # Boucle : permet de prédire plusieurs images à la suite
    while True:

        print("\n" + "-" * 60)
        print("  Ouvrez la fenêtre de sélection pour choisir une image")
        print("  (fermez la fenêtre sans choisir pour quitter)")
        print("-" * 60)

        # Ouvrir la fenêtre de sélection
        chemin_image = choisir_image()
        
        # Si l'utilisateur ferme sans choisir → on quitte
        if not chemin_image:
            print("\n  Au revoir !\n")
            break

        print(f"\n  Image sélectionnée : {os.path.basename(chemin_image)}")

        # Prédiction
        classe, confiance, top3 = predire_image(chemin_image, modele, device)
        
        # Affichage du résultat
        print("\n" + "=" * 60)
        print("   RÉSULTAT DE L'ANALYSE")
        print("=" * 60)
        
        if confiance >= SEUIL_CONFIANCE * 100:
            print(f"\n  Diagnostic    : {classe}")
            print(f"  Confiance     : {confiance:.2f}%")
        else:
            print(f"\n  Diagnostic    : Incertain (confiance trop faible)")
            print(f"  Meilleure hypothèse : {classe} ({confiance:.2f}%)")
            print(f"  → Image peut-être hors dataset ou qualité insuffisante")
            
        print(f"\n  Top 3 des probabilités :")
        for i, (nom, prob) in enumerate(top3):
            barre = "█" * int(prob / 5)  # barre visuelle proportionnelle
            print(f"    {i+1}. {nom:<25} {prob:>6.2f}%  {barre}")

        print("=" * 60)
        
        # Proposer de continuer
        continuer = input("\n  Analyser une autre image ? (o/n) : ").strip().lower()
        if continuer != "o":
            print("\n  Au revoir !\n")
            break
        
# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    predire()