import os
from pathlib import Path

# Racine du projet (dossier parent de src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Chemins des données et modèles
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Architectures supportées et par défaut
ARCHITECTURES_DISPONIBLES = ["efficientnet_b0", "resnet18", "mobilenet_v3_small", "convnext_tiny"]
DEFAULT_ARCHITECTURE = "efficientnet_b0"
DEFAULT_MODEL_PATH = MODELS_DIR / f"{DEFAULT_ARCHITECTURE}.pth"

# Hyperparamètres et configurations de traitement
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.0001
NB_EPOCHS = 10
SEED = 42
SEUIL_CONFIANCE_MIN = 70.0  # Seuil minimal (%) sous lequel une image est rejetée
SEUIL_PIXELS_FEUILLE_MIN = 0.10  # 10% min de couverture végétale/feuille (HSV)
LABEL_SMOOTHING = 0.1
QUALITE_JPEG = 85
EXTENSIONS_VALIDE = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Normalisation standard ImageNet (Moyenne & Écart-type RGB)
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

# Dataset Kaggle de référence
KAGGLE_DATASET_HANDLE = "anshulm257/rice-disease-dataset"

# Noms et nombre de classes
CLASSES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf scald',
    'Sheath Blight'
]
NB_CLASSES = len(CLASSES)

# Extension de fichier de modèle
EXTENSION_MODEL = ".pth"
