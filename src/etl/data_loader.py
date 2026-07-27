 # Charge et prépare les images (redimensionnement, normalisation...)
 
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ============================================================
# CONFIGURATION
# ============================================================

CHEMIN_PROPRE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "donne_propre"
)

# Taille du batch : nombre d'image traitées ensemble à chaque étape
# 32 = standard industrie : bon équilibre mémoire / vitesse / précision
TAILLE_BATCH = 32

# Réparation des données : 80% entraînement, 20% validation
# On ne touche PAS encore aux données de test - elles restent
# réservées uniquement pour l'évaluation  finale du modèle
RATIO_VALIDATION = 0.2

# Graine aléatoire : garantit que la séparation train/val
# est toujours la même à chaque exécution
GRAINE = 42

# ============================================================
# TRANSFORMATIONS EN MÉMOIRE (normalisation [0,1] → [-1,1])
# ============================================================

# transforms.Compose = pipeline de transformations appliquées
# dans l'ordre, à chaque image chargée en mémoire
# APRÈS : deux pipelines séparés
transformations_train = transforms.Compose([

    # Rotation aléatoire entre -15° et +15°
    # une feuille malade reste malade même inclinée
    transforms.RandomRotation(degrees=15),

    # Retournement horizontal aléatoire (50% de chance)
    # la maladie apparaît des deux côtés de la feuille
    transforms.RandomHorizontalFlip(p=0.5),

    # Retournement vertical aléatoire (30% de chance)
    transforms.RandomVerticalFlip(p=0.3),

    # Variation aléatoire de luminosité, contraste, saturation
    # simule différentes conditions d'éclairage sur le terrain
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

transformations_val = transforms.Compose([
    # Validation : aucune augmentation
    # on évalue sur des images propres et stables
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def creer_dataloaders():
    """
    Charge les images depuis data/donne_propre/,
    applique les transformations en mémoire,
    sépare en train/validation,
    et retourne deux DataLoaders prêts pour le CNN.

    Retourne :
        loader_train    (DataLoader) : batches pour l'entraînement
        loader_val      (DataLoader) : batches pour la validation
        noms_classes    (list)       : ['Bacterial Leaf Blight', ...]
        nb_classes      (int)        : 6
    """
    
    print("\n" + "=" * 60)
    print("   CHARGEMENT DES DONNÉES PROPRES")
    print("=" * 60)
    
    # --------------------------------------------------------
    # Étape 1 : ImageFolder
    # torchvision lit automatiquement la structure de dossiers :
    # donne_propre/
    #   Bacterial Leaf Blight/  → classe 0
    #   Brown Spot/             → classe 1
    #   Healthy Rice Leaf/      → classe 2
    #   ...
    # Il assigne un index numérique à chaque classe automatiquement
    # --------------------------------------------------------
    
    dataset_train_complet = datasets.ImageFolder(
        root      = CHEMIN_PROPRE,
        transform = transformations_train
    )
    
    dataset_val_complet = datasets.ImageFolder(
        root      = CHEMIN_PROPRE,
        transform = transformations_val
    )
    
    noms_classes = dataset_train_complet.classes
    nb_classes   = len(noms_classes)
    
    print(f"\n Classes détectées ({nb_classes})")
    for i , nom in enumerate(noms_classes):
        print(f"   [{i}] : {nom}")
    
    # --------------------------------------------------------
    # Étape 2 : Séparation train / validation
    # On garde 80% pour entraîner , 20% pour valider
    # random_split garantit que les deux groupes ne se chevauchent pas
    # --------------------------------------------------------
    
    taille_total = len(dataset_train_complet)
    taille_val   = int(taille_total * RATIO_VALIDATION)
    taille_train = taille_total - taille_val
    
    # torch.Generator avec GRAINE = résultat identique à chaque run
    
    import torch
    
    generateur = torch.Generator().manual_seed(GRAINE)
    
    # On génère les indices de séparation
    indices = torch.randperm(taille_total, generator=generateur).tolist()
    indices_train = indices[:taille_train]
    indices_val   = indices[taille_val:]
    
    # Chaque subset utilise son propre dataset avec sa propre transformation
    from torch.utils.data import Subset
    dataset_train = Subset(dataset_train_complet, indices_train)
    dataset_val   = Subset(dataset_val_complet,   indices_val)
    
    print(f"\n Répartition des données :")
    print(f" Total        : {taille_total} images")
    print(f" Entraînement : {taille_train} images (80%)")
    print(f" Validation   : {taille_val} images (20%)")
    
    # --------------------------------------------------------
    # Étape 3 : Création des DataLoaders
    # Le DataLoader gère automatiquement :
    #   - La découpe en batches (TAILLE_BATCH images à la fois)
    #   - Le mélange aléatoire (shuffle) à chaque époque
    #   - Le chargement parallèle (num_workers)
    # --------------------------------------------------------
    
    loader_train = DataLoader(
        dataset_train,
        batch_size = TAILLE_BATCH,
        shuffle    = True,               # mélange à chaque époque -> évite que 
                                         # le modèle apprenne l'ordre des images
        num_workers= 0                   # 0 = thread principal (stable sur windows)
    )                                    # augmenter si sur Linux avec GPU
    
    loader_val = DataLoader(
        dataset_val,
        batch_size = TAILLE_BATCH,
        shuffle    = False,              # pas besoin de mélanger pour la validation
                                         # on veut des résultats reproductibles
        num_workers= 0                   
    )
    
    print(f"\n Configuration des batches :")
    print(f"  Taille batch : {TAILLE_BATCH} images")
    print(f"  Batches train : {len(loader_train)}")
    print(f"  Batches val : {len(loader_val)}")
    print("\n" + "=" * 60)
    
    return loader_train, loader_val, noms_classes, nb_classes

# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    import torch
    
    loader_train, loader_val, noms_classes, nb_classes = creer_dataloaders()
    
    # Vérifier la forme d'un batch
    images, etiquettes = next(iter(loader_train))
    
    print(f"  Forme d'un batch images  : {images.shape}")
    print(f"  -> [batch, canaux, hauteur, largeur]")
    print(f"  -> [{TAILLE_BATCH}, 3, 224, 224] attendu")
    print(f"\n Forme des étiquettes  : {etiquettes.shape}")
    print(f"   Valeurs min/max pixels : {images.min():.3f} / {images.max():.3f}")
    print(f"  -> Normalisation confirmée si valeurs autour de [-2, +2]")