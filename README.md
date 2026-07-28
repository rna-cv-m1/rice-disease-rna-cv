# Diagnostic des maladies du riz par vision artificielle

Système automatisé d'analyse et de classification des maladies sur les feuilles de riz par apprentissage profond (PyTorch) et vision artificielle.

---

## 1. Description du projet

Le projet permet d'identifier 6 états pathologiques sur les feuilles de riz :

- **Bacterial Leaf Blight** (Brûlure bactérienne)
- **Brown Spot** (Tache brune)
- **Healthy Rice Leaf** (Feuille saine)
- **Leaf Blast** (Pyriculariose)
- **Leaf scald** (Échaudage des feuilles)
- **Sheath Blight** (Rhizoctonie)

L'architecture s'appuie sur le **Transfer Learning multi-architectures** (`EfficientNet-B0`, `ConvNeXt-Tiny`, `MobileNetV3-Small`, `ResNet18`) avec rééquilibrage automatique des classes (`WeightedRandomSampler`), précision mixte GPU (`AMP FP16`), et l'élimination du biais de fond (*Center Crop* anti-shortcut learning).

---

## 2. Structure du projet

```text
rice_disease_rna_cv/
├── pyproject.toml            # Packaging PEP 517/518 et configuration PyTest
├── requirements.txt          # Liste des dépendances Python unifiées
├── README.md                 # Documentation globale du projet
├── data/                     # Jeux de données MLOps
│   ├── raw/                  # Images brutes originales (parcours récursif)
│   ├── processed/            # Images rognées au centre (Center Crop) 224x224
│   └── test/                 # Images réservées à l'évaluation et l'inférence
├── models/                   # Fichiers de poids des modèles (.pth) par nom d'architecture
├── notebooks/                # Notebooks d'exécution pas-à-pas
│   └── colab_rice_disease_pipeline.ipynb  # Pipeline Colab complet commenté ligne par ligne
├── src/
│   ├── config.py             # Centralisation des paramètres, seuils et chemins
│   ├── dataset/              # Traitement, EDA, extraction récursive et chargement des données
│   │   ├── extract.py        # Extraction récursive des sous-dossiers
│   │   ├── eda.py            # Analyse exploratoire sans emojis (dimensions, distribution, HSV)
│   │   ├── transform.py      # Center Crop & redimensionnement 224x224
│   │   ├── loader.py         # DataLoaders PyTorch (AMP, num_workers, pin_memory, WeightedRandomSampler)
│   │   └── verifier.py       # Validation HSV pour filtrage des images hors-domaine (OOD)
│   ├── nn/                   # Réseaux de neurones PyTorch
│   │   ├── builder.py        # Instanciation et chargement agnostique (4 architectures)
│   │   ├── trainer.py        # Entraînement multi-modèles (AMP FP16, AdamW, CosineAnnealingLR)
│   │   ├── evaluator.py      # Évaluation complète et matrice de confusion
│   │   └── compare.py        # Benchmark et sélection automatique du meilleur modèle
│   ├── cli/                  # Inférence en ligne de commande
│   │   └── predict.py        # Prédiction CLI
│   └── ui/                   # Interface Web Streamlit
│       └── app.py            # Application Web (Agencement 2 colonnes)
└── tests/                    # Suite de tests unitaires automatisés
    ├── test_data.py          # Tests des transformations et DataLoaders
    └── test_model.py         # Tests des 4 architectures PyTorch
```

---

## 3. Guide de démarrage rapide

### Installation locale

1. Activer l'environnement virtuel :

```bash
source .venv/bin/activate
```

2. Installer le projet et ses dépendances :

```bash
pip install -e .
```

---

## 4. Modes d'utilisation

### Option 1 : Google Colab (notebook pas-à-pas)

Le projet intègre un notebook d'exécution complet prêt pour Colab sur GPU :

- **Fichier** : `notebooks/colab_rice_disease_pipeline.ipynb`
- **Contenu** : Guide d'installation des clés (Colab Secrets `GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`), téléchargement Kaggle, EDA, transformation récursive, entraînement de toutes les architectures, visualisations complètes (matrice de confusion brute/normalisée, bar chart precision/recall/F1 par classe, support, curves, benchmark, radar chart) et lancement Streamlit via tunnel Cloudflare.

### Option 2 : Interface web Streamlit (UI)

Interface utilisateur  sur 2 colonnes :

- **Colonne 1** : Importation de photo et aperçu compact .
- **Colonne 2** : Diagnostic, niveau de confiance, filtrage OOD (images non-feuilles), graphique de probabilités et détail.

```bash
streamlit run src/ui/app.py
```

Accès direct sur `http://localhost:8501`.

### Option 3 : Ligne de commande (CLI)

Analyse et diagnostic rapides d'une image en terminal sans serveur web :

```bash
python -m src.cli.predict --image "data/test/Leaf scald/RRDI_LeafScald1.jpg"
```

---

## 5. Maintenance, benchmark et tests

### Analyse exploratoire des données (EDA)

```bash
python -m src.dataset.eda
```

### Prétraitement et transformation des images

```bash
python -m src.dataset.transform
```

### Entraîner une architecture spécifique

```bash
python -m src.nn.trainer --arch efficientnet_b0 --epochs 10
```

*Architectures au choix* : `efficientnet_b0`, `convnext_tiny`, `mobilenet_v3_small`, `resnet18`.

### Benchmark comparatif des architectures

```bash
python -m src.nn.compare
```

### Évaluation du meilleur modèle

```bash
python -m src.nn.evaluator
```

### Exécuter la suite de tests unitaires (pytest)

```bash
pytest
```
