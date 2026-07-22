#!/usr/bin/env bash
# script-install-interface.sh
# Script d'installation rapide de l'interface Streamlit

set -e  # Sortir en cas d'erreur

VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/bin/python"

ensure_venv() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "📦 Création de l'environnement virtuel local..."
        python3 -m venv "$VENV_DIR"
    fi
}

echo "🌾 Installation - Interface de Détection des Maladies du Riz"
echo "============================================================"
echo ""

# Vérifier Python
echo "✓ Vérification de Python..."
python3 --version | grep -q "3.1[0-9]" || {
    echo "❌ Python 3.10+ requis"
    exit 1
}

# Vérifier pip
echo "✓ Vérification de pip..."
pip --version > /dev/null || {
    echo "❌ pip non trouvé"
    exit 1
}

ensure_venv

echo "✓ Utilisation de l'environnement virtuel : $VENV_DIR"

# Options d'installation
echo ""
echo "Choisir le mode d'installation :"
echo "1) Dépendances minimales (Streamlit seul)"
echo "2) Dépendances complètes (requirements-interface.txt)"
echo "3) Dépendances du projet (pyproject.toml)"
echo ""
read -p "Choix (1/2/3) [1] : " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "📦 Installation de Streamlit..."
        "$PYTHON_BIN" -m pip install --upgrade pip
        "$PYTHON_BIN" -m pip install streamlit
        ;;
    2)
        echo ""
        echo "📦 Installation des dépendances complètes..."
        "$PYTHON_BIN" -m pip install --upgrade pip
        "$PYTHON_BIN" -m pip install -r requirements-interface.txt
        ;;
    3)
        echo ""
        echo "📦 Installation du projet complet..."
        "$PYTHON_BIN" -m pip install --upgrade pip
        "$PYTHON_BIN" -m pip install -e .
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

echo ""
echo "✅ Installation terminée!"
echo ""
echo "🚀 Lancer l'interface :"
echo "   source $VENV_DIR/bin/activate && streamlit run src/app.py"
echo "   ou : $PYTHON_BIN -m streamlit run src/app.py"
echo ""
echo "📖 Documentation :"
echo "   cat INTERFACE.md          # Documentation complète"
echo "   cat QUICKSTART.md         # Guide rapide"
echo "   python test_interface.py  # Tests des modules"
echo ""
