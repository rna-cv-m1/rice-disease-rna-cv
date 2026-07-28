import sys
from pathlib import Path
from typing import Tuple

# Inclusion dynamique du dossier racine du projet dans sys.path pour les imports relatifs
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

from src.config import DEFAULT_MODEL_PATH, CLASSES, IMAGE_SIZE, SEUIL_CONFIANCE_MIN
from src.dataset.verifier import verifier_est_feuille_riz
from src.nn.builder import charger_meilleur_modele, cree_modele_pretrain

# ============================================================
# 1. CONFIGURATION & STYLES (UI Épurée et Adaptative)
# ============================================================

st.set_page_config(page_title="Diagnostic Maladie du Riz", layout="wide")

st.markdown("""
    <style>
    /* Masquer la sidebar latérale pour garder une vue fluide et minimaliste */
    [data-testid="stSidebar"] { display: none; }

    /* Titre principal vert sobre */
    .main-title { color: #2E7D32; font-size: 2rem; font-weight: 700; margin-bottom: 5px; }

    /* Badges de statut adaptatifs (compatibles mode clair et sombre) */
    .status-ok { background-color: rgba(46, 125, 50, 0.15); color: #2E7D32; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .status-err { background-color: rgba(198, 40, 40, 0.15); color: #C62828; padding: 4px 12px; border-radius: 20px; font-weight: 600; }

    /* Cartes de résultats stylisées avec bordures colorées */
    .result-card { border: 1px solid rgba(128, 128, 128, 0.2); border-left: 5px solid #2E7D32; padding: 15px; border-radius: 8px; }
    .result-title { font-size: 1.3rem; color: #2E7D32; font-weight: 700; }

    /* Traduction personnalisée du bouton d'upload en français */
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"]::before { content: "Parcourir..."; visibility: visible; }
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] { font-size: 0px !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. CHARGEMENT DU MODÈLE (Détection Automatique & Mise en Cache)
# ============================================================

@st.cache_resource
def charger_modele_ui() -> Tuple[torch.nn.Module, torch.device, str]:
    """Charge et met en cache le meilleur modèle PyTorch disponible.

    Returns:
        Tuple[torch.nn.Module, torch.device, str]: Modèle chargé, calculateur (CPU/GPU) et nom de l'architecture.
    """
    # 1. Sélection automatique de l'accélérateur matériel (GPU CUDA si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Détection agnostique de l'architecture et chargement des poids du fichier .pth
    if DEFAULT_MODEL_PATH.exists():
        modele, arch = charger_meilleur_modele(DEFAULT_MODEL_PATH, device=device)
    else:
        modele = cree_modele_pretrain().to(device)
        arch = "ResNet18"
    
    return modele, device, arch

# ============================================================
# 3. EN-TÊTE & BADGES DE STATUT
# ============================================================

st.markdown('<h1 class="main-title">Diagnostic des maladies du riz</h1>', unsafe_allow_html=True)

# Tentative de chargement du modèle et affichage des 2 badges distincts (Modèle et Calcul)
try:
    modele, device, arch = charger_modele_ui()
    model_pret = True
    device_txt = "GPU" if device.type == "cuda" else "CPU"
    st.markdown(f'''
        <span class="status-ok">Modèle : {arch.upper()}</span>
        <span class="status-ok" style="margin-left: 8px;">Calcul : {device_txt}</span>
    ''', unsafe_allow_html=True)
except Exception as e:
    model_pret = False
    st.markdown(f'<span class="status-err">Erreur modèle : {e}</span>', unsafe_allow_html=True)

st.write("")
st.write("Choisissez une photo de feuille de riz à analyser.")

# ============================================================
# 4. SÉLECTION ET ANALYSE D'IMAGE
# ============================================================

# Composant de téléversement d'image en français
uploaded_file = st.file_uploader("Fichier", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

if uploaded_file and model_pret:
    # Lecture et conversion de l'image téléversée en mode RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # Disposition sur 2 colonnes égales
    col1, col2 = st.columns(2, gap="medium")

    # Colonne 1 : Affichage de l'image d'origine de l'utilisateur
    with col1:
        st.image(image, caption="Photo importée", use_container_width=True)

    # Colonne 2 : Pipeline d'analyse et résultats
    with col2:
        # Étape A : Filtrage Hors-Domaine (OOD) via vérification de couleur végétale HSV
        est_feuille, ratio = verifier_est_feuille_riz(image)

        if not est_feuille:
            # Rejet immédiat si aucune feuille de riz n'est détectée (ex: capture d'écran, tableau)
            st.error("Image rejetée : Aucune feuille de riz végétale détectée sur la photo.")
            st.markdown(f"""
                <div class="result-card" style="border-left-color: #C62828;">
                    <div class="result-title" style="color: #C62828;">Diagnostic : Image non valide / Hors domaine</div>
                    <div>Couverture végétale détectée : <b>{ratio * 100:.1f}%</b> (Seuil minimum : 10%)</div>
                    <div style="margin-top:8px; font-size:0.9rem; color:#6B7280;">Veuillez importer une vraie photo de feuille de riz (pas de tableau, texte ou objet neutre).</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Étape B : Prétraitement de la photo et inférence PyTorch
            with st.spinner("Analyse en cours..."):
                # Prétraitement standard de l'image vers tenseur shape : (C=3, H=224, W=224)
                tf = transforms.Compose([
                    transforms.Resize(IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                # Conversion en batch unique : shape (Batch_size=1, C=3, H=224, W=224)
                tensor = tf(image).unsqueeze(0).to(device)
                
                # Inférence PyTorch sans calcul de gradient
                with torch.no_grad():
                    probs = torch.softmax(modele(tensor)[0], dim=0)  # Vector de probabilités shape : (6,)

                # Extraction de la probabilité maximale et de l'indice de classe
                top_prob, top_idx = torch.max(probs, 0)
                classe = CLASSES[top_idx.item()]
                confiance = float(top_prob.item() * 100.0)

            # Étape C : Vérification du seuil de confiance minimal (70%)
            if confiance < SEUIL_CONFIANCE_MIN:
                st.warning("Attention : Confiance insuffisante pour établir un diagnostic certain.")
                st.markdown(f"""
                    <div class="result-card" style="border-left-color: #F57C00;">
                        <div class="result-title" style="color: #F57C00;">Diagnostic : Incertain</div>
                        <div>Meilleure hypothèse : <b>{classe}</b> ({confiance:.2f}%)</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Diagnostic valide et confirmé
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-title">Diagnostic : {classe}</div>
                        <div>Confiance : <b>{confiance:.2f}%</b></div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("<b>Distribution des probabilités :</b>", unsafe_allow_html=True)

            # Étape D : Construction du graphique de probabilités interactif
            df_probs = pd.DataFrame({
                "Maladie": CLASSES, 
                "Probabilité (%)": [round(float(p.item() * 100.0), 2) for p in probs]
            })
            st.bar_chart(df_probs.set_index("Maladie"), color="#2E7D32")

            # Étape E : Tableau récapitulatif détaillé sous panneau déroulant
            with st.expander("Voir le détail des probabilités"):
                st.dataframe(
                    df_probs.sort_values(by="Probabilité (%)", ascending=False).reset_index(drop=True), 
                    use_container_width=True
                )
