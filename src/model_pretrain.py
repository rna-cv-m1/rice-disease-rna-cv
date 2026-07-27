# Définit l'architecture du CNN (les couches, les neurones)

import torch

import torch.nn as nn
from torchvision import models

# ============================================================
# ARCHITECTURE CNN PRÉ-ENTRAÎNÉ (TRANSFER LEARNING)
# Modèle choisi : ResNet18
#   - Entraîné sur ImageNet (1.2 million d'images, 1000 classes)
#   - Il "sait" déjà détecter contours, textures, formes
#   - On remplace uniquement sa dernière couche par nos 6 classes
# ============================================================

def cree_modele_pretrain(nb_classes=6, geler_base=True):
    """
    Charge ResNet18 pré-entraîné et l'adapte à nos 6 classes.

    Paramètres :
        nb_classes  (int)  : nombre de classes à prédire (6)
        geler_base  (bool) : si True, on n'entraîne QUE
                             la dernière couche (plus rapide)
                             si False, on ré-entraîne tout
                             le réseau (fine-tuning complet)
    Retourne :
        modele (nn.Module) : modèle prêt à entraîner
    """
    # --------------------------------------------------------
    # Étape 1 : charger ResNet18 avec ses poids ImageNet
    # weights=IMAGENET1K_V1 = poids officiels pré-entraînés
    # --------------------------------------------------------
    modele = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # --------------------------------------------------------
    # Étape 2 : geler toutes les couches de la base
    # "Geler" = interdire la modification des poids pendant
    # l'entraînement → on préserve ce que ResNet sait déjà
    # --------------------------------------------------------
    if geler_base:
        for parametre in modele.parameters():
            parametre.requires_grad = False
    
    # --------------------------------------------------------
    # Étape 3 : remplacer la dernière couche (fc = fully connected)
    # ResNet18 original : Linear(512, 1000) → 1000 classes ImageNet
    # Notre version     : Linear(512, 6)    → nos 6 maladies de riz
    # Cette nouvelle couche est dégelée automatiquement
    # (requires_grad=True par défaut pour un nouveau nn.Linear)
    # --------------------------------------------------------
    nb_entrees_fc = modele.fc.in_features # = 512 pour ResNet18
    modele.fc = nn.Sequential(
        nn.Linear(nb_entrees_fc, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, nb_classes)
    )
    
    return modele

# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    import torch
    
    modele = cree_modele_pretrain(nb_classes=6, geler_base=True)
    
    # Simuler un batch de 32 images 224*224 RGB
    batch_test = torch.randn(32, 3, 224, 224)
    sortie = modele(batch_test)
    
    print("=" * 55)
    print(" TEST CNN PRE-ENTRAINE (ResNet18 Transfer Learning)")
    print("=" * 55)
    print(f" Entrée  : {batch_test.shape}")
    print(f" Sortie  : {sortie.shape}")
    print(" Attendu : torch.Size([32, 6])")
    
    # Compter paramètres total vs paramètres entraînables
    total_params  = sum(p.numel() for p in modele.parameters())
    params_entrainer = sum(
        p.numel() for p in modele.parameters() if p.requires_grad
    )
    
    print(f"\n  Paramètres totaux      : {total_params:,}")
    print(f"    Paramètres entaînables : {params_entrainer:,}")
    print(f"    Paramètres gelés       : {total_params - params_entrainer:,}")
    print(f"\n  -> On n'entraîne que {params_entrainer/total_params*100:.1f}%")
    print("    du réseau - beaucoup plus rapide !")
    print("=" * 55)
    
    