# 🚀 Guide de démarrage rapide - Interface

## Installation express (< 2 min)

### 1️⃣ Installer Streamlit

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit
```

### 2️⃣ Démarrer l'application

```bash
cd /home/tahina/tp/2026/ia/rna/rice-disease-rna-cv
source .venv/bin/activate
streamlit run src/app.py
```

### 3️⃣ Accéder à l'interface

- L'app s'ouvre automatiquement sur `http://localhost:8501`
- Si non, ouvrir manuellement ce lien dans votre navigateur

---

## 📖 Utilisation rapide

### Analyser une image

1. **Cliquer** sur "Choisir une image de feuille de riz"
2. **Sélectionner** une image (JPG, PNG ou BMP)
3. **Attendre** le diagnostic (< 1 seconde)
4. **Consulter** les recommandations affichées

### Paramètres

- **Modèle** : Choisir dans le menu latéral
- **Seuil de confiance** : Ajuster le curseur pour accepter/rejeter des prédictions

---

## 🛠️ Troubleshooting

### L'app refuse de démarrer

```bash
# Vérifier la version Python
python --version        # Doit être 3.10+

# Recréer l'environnement virtuel
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit

# Relancer
streamlit run src/app.py
```

### "Port déjà utilisé"

```bash
# Utiliser un autre port
streamlit run src/app.py --server.port 8502
```

### Les modèles ne sont pas trouvés

Vérifier que le dossier `models/` contient les fichiers de modèles pré-entraînés.

---

## 📚 Documentation complète

Pour plus de détails → Voir [INTERFACE.md](../INTERFACE.md)

---

**Bon diagnostic ! 🌾**
