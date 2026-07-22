# 📦 Architecture du module `src/`

## Structure

```
src/
├── __init__.py              # Initialisation du package
├── app.py                   # Interface Streamlit principale
├── model_manager.py         # Gestion des modèles pré-entraînés
├── utils.py                 # Utilitaires (preprocessing, postprocessing)
└── [À venir]
    ├── api.py              # API FastAPI (futur)
    ├── callbacks.py         # Callbacks et observateurs
    └── config.py            # Configuration centralisée
```

---

## Modules détaillés

### 🎨 `app.py` - Interface Streamlit

L'interface web principale de l'application.

**Responsabilités :**
- Afficher l'interface utilisateur
- Gérer l'upload des images
- Afficher les résultats (diagnostic, confiance, graphiques)
- Afficher les recommandations par maladie

**Dépendances :**
- `streamlit` : Framework web
- `scipy` : Traitement scientifique
- `pandas` : DataFrames
- `PIL` : Traitement d'images

**Points d'entrée :**
```bash
streamlit run app.py
```

---

### 🤖 `model_manager.py` - Gestion des modèles

Classe `ModelManager` pour charger et gérer les modèles pré-entraînés.

**Classe :** `ModelManager`

**Méthodes principales :**

| Méthode | Description |
|---------|---|
| `list_available_models()` | Liste les modèles disponibles |
| `load_model(name, device)` | Charger un modèle en mémoire |
| `get_model_info(name)` | Informations sur un modèle |
| `clear_cache()` | Vider le cache des modèles |

**Exemple d'usage :**
```python
from model_manager import ModelManager

# Lister les modèles
models = ModelManager.list_available_models()
print(models)  # ['ResNet50', 'VGG16', ...]

# Charger un modèle
model = ModelManager.load_model("ResNet50", device="cpu")

# Utiliser le cache automatiquement
model = ModelManager.load_model("ResNet50", device="cpu")  # Chargé depuis cache
```

**Modèles supportés :**
- ResNet50
- VGG16
- EfficientNet

---

### 🛠️ `utils.py` - Utilitaires

Contient trois classes utilitaires :

#### 1. `ImagePreprocessor`

Prétraitement des images pour prédiction.

**Méthodes :**

| Méthode | Description |
|---------|---|
| `preprocess(image, size)` | Redimensionne, normalise, prépare pour modèle |
| `get_image_info(image)` | Retourne les metadata de l'image |

**Exemple :**
```python
from utils import ImagePreprocessor
from PIL import Image

image = Image.open("feuille.jpg")
processed = ImagePreprocessor.preprocess(image)  # Shape: (1, 3, 224, 224)
info = ImagePreprocessor.get_image_info(image)   # {'size': (800, 600), ...}
```

#### 2. `PredictionPostprocessor`

Posttraitement des prédictions du modèle.

**Méthodes :**

| Méthode | Description |
|---------|---|
| `process_predictions(logits, threshold)` | Convertit logits → probabilités |
| `get_top_prediction(pred_dict)` | Retourne la meilleure prédiction |
| `get_prediction_dataframe(pred_dict)` | Convertit en pandas DataFrame |

**Exemple :**
```python
from utils import PredictionPostprocessor
import numpy as np

# Sortie du modèle (supposée)
logits = np.array([-2.1, 1.5, -0.5, 3.2, 0.1])

# Traiter
predictions = PredictionPostprocessor.process_predictions(logits)
# {'Saine': 0.95, 'Cercosporiose': 0.03, ...}

top_class, top_prob = PredictionPostprocessor.get_top_prediction(predictions)
# ('Saine', 0.95)
```

#### 3. `ValidationUtils`

Validation des images.

**Méthodes :**

| Méthode | Description |
|---------|---|
| `validate_image_file(filepath)` | Valider un fichier image |
| `get_quality_warning(image)` | Avertissements de qualité |

**Exemple :**
```python
from utils import ValidationUtils

# Valider
is_valid, msg = ValidationUtils.validate_image_file("image.jpg")
print(msg)  # "Image valide" ou message d'erreur

# Warnings de qualité
warnings = ValidationUtils.get_quality_warning(image)
# ["⚠️ Résolution faible (min conseillé: 224x224)"]
```

---

## Constantes globales

### Classes de maladies (`utils.py`)

```python
DISEASE_CLASSES = {
    0: "Saine",
    1: "Cercosporiose",
    2: "Pyriculariose",
    3: "Brûlure bactérienne",
    4: "Tache brune",
}
```

### Couleurs associées (`utils.py`)

```python
CLASS_COLORS = {
    "Saine": "#2ecc71",              # Vert
    "Cercosporiose": "#f39c12",       # Orange
    "Pyriculariose": "#e74c3c",       # Rouge
    "Brûlure bactérienne": "#f39c12", # Orange
    "Tache brune": "#e67e22",         # Orange-marron
}
```

---

## 🔄 Pipeline d'analyse en image

```
[Image utilisateur]
         ↓
  [Validation] ← ValidationUtils.validate_image_file()
         ↓
[Prétraitement] ← ImagePreprocessor.preprocess()
         ↓
  [Modèle] ← ModelManager.load_model()
         ↓
[Prédiction] ← model.predict()
         ↓
[Posttraitement] ← PredictionPostprocessor.process_predictions()
         ↓
[Affichage] ← app.py (Streamlit)
```

---

## 📝 Convention de code

- **Docstrings** : Format NumPy ("""...""" avec descriptions)
- **Type hints** : Utilisés partout (`from typing import ...`)
- **Logging** : `import logging` et `logger = logging.getLogger(__name__)`
- **Naming** :
  - Classes : `PascalCase` (ex: `ImagePreprocessor`)
  - Functions/méthodes : `snake_case` (ex: `preprocess_image`)
  - Constants : `UPPER_CASE` (ex: `DEFAULT_SIZE`)

---

## 🧪 Tests

Pour tester les modules individuellement :

```bash
# Test du ModelManager
python -c "from src.model_manager import ModelManager; print(ModelManager.list_available_models())"

# Test du ImagePreprocessor
python -c "
from src.utils import ImagePreprocessor
from PIL import Image
img = Image.new('RGB', (300, 300))
processed = ImagePreprocessor.preprocess(img)
print(f'Shape: {processed.shape}')
"

# Test de validation
python -c "
from src.utils import ValidationUtils
result, msg = ValidationUtils.validate_image_file('test.jpg')
print(msg)
"
```

---

## 🚀 Extensibilité future

**Points d'extension prévus :**

1. **`api.py`** : API HTTP (FastAPI) pour intégrations externes
2. **`callbacks.py`** : Observateurs pour événements applicatifs
3. **`config.py`** : Configuration centralisée (fichier YAML/TOML)
4. **`models/custom.py`** : Modèles custom (fine-tuning, ensemble)
5. **`handlers/`** : Handlers pour différents types de traitements

---

## 📚 Documentation pour développeurs

Consulter les docstrings dans les fichiers `.py` pour plus de détails sur chaque fonction.

```bash
# Générer la documentation HTML (futur)
# python -m pdoc src/ --html -o docs/
```

---

**Dernière mise à jour :** Juillet 2026  
**Mainteneur :** Équipe Développement

