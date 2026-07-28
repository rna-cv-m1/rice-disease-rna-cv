# Détection des Maladies du Riz par Vision Artificielle

Système automatisé d'analyse et de classification des maladies sur les feuilles de riz par apprentissage profond (PyTorch) et vision artificielle.

---

## 1. Description du Projet

Le projet permet d'identifier 6 états pathologiques sur les feuilles de riz :

- Bacterial Leaf Blight (Brûlure bactérienne)
- Brown Spot (Tache brune)
- Healthy Rice Leaf (Feuille saine)
- Leaf Blast (Pyriculariose)
- Leaf scald (Échaudage des feuilles)
- Sheath Blight (Rhizoctonie)

L'architecture s'appuie sur un réseau de neurones ResNet18 ré-entraîné (Transfer Learning).

---

## 2. Structure du Projet

```text
rice_disease_rna_cv/
├── pyproject.toml            # Métadonnées et packaging du projet
├── requirements.txt          # Liste des dépendances Python
├── README.md                 # Documentation globale
├── data/                     # Jeux de données MLOps
│   ├── raw/                  # Images brutes originales
│   ├── processed/            # Images nettoyées et structurées par classe
│   └── test/                 # Images réservées à l'évaluation et inférence
├── models/                   # Fichiers de poids du modèle (.pth)
├── src/
│   ├── config.py             # Centralisation des paramètres et chemins
│   ├── dataset/              # Traitement et chargement des images
│   ├── nn/                   # Architectures, entraînement et évaluation du modèle
│   │   ├── builder.py
│   │   ├── compare.py        # Comparaison automatique (ResNet18 vs EfficientNet)
│   │   ├── evaluator.py
│   │   └── trainer.py
│   ├── cli/                  # Inférence en ligne de commande
│   └── ui/                   # Interface graphique Web (Streamlit)
└── tests/                    # Tests unitaires automatisés
    ├── test_data.py          # Tests des transformations et du chargement des données
    └── test_model.py         # Tests de l'architecture PyTorch et des sorties
```

---

## 3. Guide de Démarrage

### Installation

1. Activer l'environnement virtuel :

```bash
source .venv/bin/activate
```

2. Installer le projet et ses dépendances :

```bash
uv pip install -e .
```

---

## 4. Utilisation

Le projet propose deux modes d'utilisation au choix :

### Option 1 : Interface Web (UI Streamlit)

Idéal pour une utilisation visuelle et interactive dans le navigateur.

```bash
streamlit run src/ui/app.py
```

Accès direct sur `http://localhost:8501`.

### Option 2 : Ligne de commande (CLI)

Idéal pour une analyse rapide ou automatisée en terminal sans serveur web.

```bash
python -m src.cli.predict --image "data/test/Leaf scald/RRDI_LeafScald1.jpg"
```

---

## 5. Maintenance et Tests

### Évaluer le modèle

```bash
python -m src.nn.evaluator
```

### Exécuter les tests unitaires

```bash
pytest
```
