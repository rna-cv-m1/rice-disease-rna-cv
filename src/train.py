# Script qui entraîne le modèle

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from src.etl.data_loader import creer_dataloaders
from src.model_pretrain import cree_modele_pretrain

# ============================================================
# CONFIGURATION
# ============================================================

NB_EPOQUES = 10 # nombre de fois qu'on parcourt tout le dataset
TAUX_APPRENT = 0.0001 # learning rate : taille du pas de correction
NB_CLASSES = 6
GRAINE = 42

# Dossier de sauvegarde du modèle entraîné
CHEMIN_MODELS = os.path.join(
    os.path.dirname(__file__), "..", "models"
)

os.makedirs(CHEMIN_MODELS, exist_ok=True)

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def entrainer():
    # --------------------------------------------------------
    # Étape 1 : Sélection du device (GPU ou CPU)
    # On envoie tout sur le GPU si disponible
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" *60)
    print(" ENTRAINEMENT DU MODELE")
    print("=" * 60)
    print(f"\n Device utilisé : {device}")
    
    if device.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    
    # --------------------------------------------------------
    # Étape 2 : Charger les données
    # --------------------------------------------------------
    
    loader_train, loader_val, noms_classes, _ = creer_dataloaders()
    
    # --------------------------------------------------------
    # Étape 3 : Charger le modèle et l'envoyer sur le GPU
    # .to(device) = déplace tous les poids vers le GPU
    # --------------------------------------------------------
    
    modele = cree_modele_pretrain(nb_classes=NB_CLASSES, geler_base=False)
    modele = modele.to(device)
    
    # --------------------------------------------------------
    # Étape 4 : Fonction de perte + Optimiseur
    #
    # CrossEntropyLoss : fonction de perte standard pour
    # la classification multi-classes
    # Elle mesure l'écart entre la prédiction et la vraie classe
    #
    # Adam : optimiseur qui ajuste automatiquement le taux
    # d'apprentissage pour chaque paramètre — plus efficace
    # que la descente de gradient classique
    #
    # params entraînables seulement (requires_grad=True)
    # = on n'optimise pas les couches gelées
    # --------------------------------------------------------
    
    critere = nn.CrossEntropyLoss()
    optimiseur = Adam(
        modele.parameters(),
        lr=TAUX_APPRENT
    )
    
    # StepLR : réduit le taux d'apprentissage de 10x
    # tous les 3 époques → affine la correction au fil du temps
    
    planificateur = StepLR(optimiseur, step_size=3, gamma=0.1)
    
    # --------------------------------------------------------
    # Étape 5 : Boucle d'entraînement
    # --------------------------------------------------------
    meilleure_precision = 0.0
    patience            = 5   # nombre d'époques sans amélioration avant d'arrêter
    compteur_patience   = 0   # compteur courant
    
    for epoque in range(1, NB_EPOQUES + 1):
        print(f"\n {'='*50}")
        print(f" Epoque {epoque}/{NB_EPOQUES}")
        print(f" {'='*50}")
        
        # ----------------------------------------------------
        # PHASE ENTRAÎNEMENT
        # modele.train() active le Dropout et BatchNorm
        # en mode entraînement
        # ----------------------------------------------------
        
        modele.train()
        perte_train = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, etiquettes) in enumerate(loader_train):
            
            # Envoyer le batch sur le GPU
            images = images.to(device)
            etiquettes = etiquettes.to(device)
            
            # Remttre les gradient à zéro avant chaque batch
            # (PyTorch accumule les gradients par défaut)
            optimiseur.zero_grad()
            
            # Passage avant : calcul des prédictions
            predictions = modele(images)
            
            # Calcul de la perte (erreur)
            perte = critere(predictions, etiquettes)
            
            # Passage arrière : calcul des gradients
            perte.backward()
            
            # Mise à jour des poids
            optimiseur.step()
            
            # Statistiques
            perte_train += perte.item()
            _, predites = torch.max(predictions, 1)
            correct_train += (predites == etiquettes).sum().item()
            total_train += etiquettes.size(0)
            
            # Afficher progression tous les 20 batches
            if (batch_idx +1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader_train)}"
                      f"| Perte : {perte.item()}")
                
        precision_train = 100 * correct_train / total_train
        perte_moy_train = perte_train / len(loader_train)
            
        # ----------------------------------------------------
        # PHASE VALIDATION
        # modele.eval() désactive Dropout et BatchNorm
        # torch.no_grad() désactive le calcul des gradients
        # (inutile en validation → économise la mémoire GPU)
        # ----------------------------------------------------
        modele.eval()
        perte_val = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, etiquettes in loader_val:
                images = images.to(device)
                etiquettes = etiquettes.to(device)
                
                predictions = modele(images)
                perte       = critere(predictions, etiquettes)
                
                perte_val += perte.item()
                _, predites = torch.max(predictions, 1)
                correct_val += (predites == etiquettes).sum().item()
                total_val += etiquettes.size(0)
                
        precision_val = 100 * correct_val / total_val
        perte_moy_val = perte_val / len(loader_val)
        
        #Afficher le bilan de l'époque
        print(f"\n Résultats époque {epoque} :")
        print(f"  Train -> Perte : {perte_moy_train:.4f} | "
              f"Précision : {precision_train:.2f}%")
        print(f"  Val -> Perte : {perte_moy_val:.4f} | "
              f"Précision : {precision_val:.2f}%")
        
        # ----------------------------------------------------
        # Sauvegarder le meilleur modèle
        # On ne sauvegarde que si la précision validation
        # est meilleure que toutes les époques précédentes
        # ----------------------------------------------------
        if precision_val > meilleure_precision:
            meilleure_precision = precision_val
            compteur_patience   = 0   # on remet le compteur à zéro
            chemin_sauvegarde = os.path.join(
                CHEMIN_MODELS, "meilleur_modele.pth"
            )
            torch.save(modele.state_dict(), chemin_sauvegarde)
            print(f"  Meilleur modèle sauvegardeé "
                  f"(précision val : {precision_val:.2f})%")
        else:
            compteur_patience += 1
            print(f"  Patience : {compteur_patience}/{patience} "
                  f"(pas d'amélioration)")
            if compteur_patience >= patience:
                print(f"\n  Arrêt anticipé à l'époque {epoque} "
                    f"— pas d'amélioration depuis {patience} époques")
                break
            
            
        # Réduire le taux d'apprentissage si nécessaire
        planificateur.step()
        
    print("\n" + "=" * 60)
    print(f" ENTRAINEMENT TERMINE")
    print(f" Meilleure précision validation : {meilleure_precision:.2f}%")
    print(f" Modèle sauvegardé dans         : models/meilleur_modele.pth")
    print("=" *60 + "\n")
    
# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    
    torch.manual_seed(GRAINE)
    entrainer()
            
            
            
            