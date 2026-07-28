import sys
import pathlib
from pathlib import Path
import torch

# Autoriser la désérialisation de PosixPath / WindowsPath dans PyTorch 2.6+ pour Streamlit
if hasattr(torch.serialization, "add_safe_globals"):
    try:
        torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    except Exception:
        pass

# Inclusion dynamique du dossier racine du projet dans sys.path pour valider les imports relatifs
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

from src.config import CLASSES, IMAGE_SIZE, SEUIL_CONFIANCE_MIN, MEAN_IMAGENET, STD_IMAGENET
from src.dataset.verifier import verifier_est_feuille_riz
from src.nn.builder import charger_meilleur_modele

# ============================================================
# 1. CONFIGURATION & STYLES (UI Épurée et Responsive)
# ============================================================

# Configuration de la page Streamlit en mode large (wide layout)
st.set_page_config(page_title="Diagnostic Maladie du Riz", layout="wide")

# Injection CSS personnalisée pour affichage propre sans sidebar
st.markdown("""
    <style>
    /* Masquer le menu latéral latéral Streamlit */
    [data-testid="stSidebar"] { display: none; }
    
    /* Titre principal vert sobre */
    .main-title { color: #2E7D32; font-size: 1.8rem; font-weight: 700; margin-bottom: 5px; }
    
    /* Badges de statut adaptatifs (Mode Clair / Mode Sombre) */
    .status-ok  { background: rgba(46,125,50,.15); color: #2E7D32; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .status-err { background: rgba(198,40,40,.15);  color: #C62828; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    
    /* Carte de résultat avec bordure latérale verte */
    .result-card { border: 1px solid rgba(128,128,128,.2); border-left: 5px solid #2E7D32;
                   padding: 15px; border-radius: 8px; margin-bottom: 15px; }
    .result-title { font-size: 1.2rem; color: #2E7D32; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. PIPELINE DE TRANSFORMATION PRÉ-CONSTRUIT
# ============================================================

# Définition unique et globale du pipeline de prétraitement pour l'inférence
_INFERENCE_TF = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET),
])


def _center_crop(img: Image.Image) -> Image.Image:
    """Effectue un rognage carré centré sur l'image pour supprimer le bruit d'arrière-plan."""
    w, h = img.size
    c = min(w, h)
    return img.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))

# ============================================================
# 3. CHARGEMENT & MISE EN CACHE DU MODÈLE PYTORCH
# ============================================================

@st.cache_resource
def charger_modele_ui():
    """Charge et met en cache le meilleur modèle PyTorch disponible dans models/."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modele, arch = charger_meilleur_modele(device=device.type)
    return modele, device, arch

# ============================================================
# 4. EN-TÊTE & BADGES D'INFORMATION (Modèle & Calcul)
# ============================================================

st.markdown('<h1 class="main-title">Diagnostic des maladies du riz</h1>', unsafe_allow_html=True)

try:
    modele, device, arch = charger_modele_ui()
    model_pret = True
    device_txt = "GPU" if device.type == "cuda" else "CPU"
    # Affichage des 2 badges de statut séparés conformément aux consignes
    st.markdown(
        f'<span class="status-ok">Modèle : {arch.upper()}</span>'
        f'<span class="status-ok" style="margin-left:8px;">Calcul : {device_txt}</span>',
        unsafe_allow_html=True,
    )
except Exception as e:
    model_pret = False
    st.markdown(f'<span class="status-err">Erreur modèle : {e}</span>', unsafe_allow_html=True)

st.write("")

# ============================================================
# 5. DISPOSITION EN 2 COLONNES (Input en Col 1, Résultats en Col 2)
# ============================================================

col1, col2 = st.columns(2, gap="large")

# --- COLONNE 1 : Téléversement et Prévisualisation compacte ---
with col1:
    st.subheader("Image de la feuille")
    uploaded_file = st.file_uploader(
        "Importer une photo de feuille", type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # Prévisualisation compacte (width=300) pour éviter tout défilement vertical (scroll)
        st.image(image, caption="Photo importée", width=300)

# --- COLONNE 2 : Résultats du Diagnostic, Probabilités et Tableaux ---
with col2:
    st.subheader("Résultat du diagnostic")

    if not uploaded_file:
        st.markdown("""
            <div class="result-card" style="border-left-color: #2E7D32;">
                <div style="font-size: 0.95rem; font-weight: 500;">Veuillez sélectionner une photo dans la colonne de gauche.</div>
            </div>
        """, unsafe_allow_html=True)
    elif model_pret:
        # Étape A : Filtrage Hors-Domaine (OOD) via analyse de couverture végétale HSV
        est_feuille, ratio = verifier_est_feuille_riz(image)

        if not est_feuille:
            st.error("Image rejetée : aucune feuille de riz détectée sur la photo.")
            st.markdown(f"""
                <div class="result-card" style="border-left-color:#C62828;">
                    <div class="result-title" style="color:#C62828;">Diagnostic : image hors domaine</div>
                    <div>Couverture végétale détectée : <b>{ratio*100:.1f}%</b> (Seuil minimum : 10%)</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Étape B : Prétraitement et inférence rapide
            with st.spinner("Analyse en cours..."):
                # Application de Center Crop -> Resize (224,224) -> ToTensor -> Normalize
                tensor = _INFERENCE_TF(_center_crop(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(modele(tensor)[0], dim=0)

            # Étape C : Identification de la classe prédite et du score de confiance
            top_prob, top_idx = torch.max(probs, 0)
            classe = CLASSES[top_idx.item()]
            confiance = top_prob.item() * 100.0

            # Étape D : Affichage direct et épuré de la maladie avec la plus haute probabilité
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{classe}</div>
                    <div>Probabilité : <b>{confiance:.2f}%</b></div>
                </div>
            """, unsafe_allow_html=True)

            # Étape E : Graphique de distribution des probabilités
            df_probs = pd.DataFrame({
                "Maladie": CLASSES,
                "Probabilité (%)": [round(p.item() * 100, 2) for p in probs],
            })
            st.write("<b>Distribution des probabilités :</b>", unsafe_allow_html=True)
            st.bar_chart(df_probs.set_index("Maladie"), color="#2E7D32")

            # Étape F : Tableau explicatif des détails de la prédiction
            with st.expander("Voir le détail des probabilités"):
                st.dataframe(
                    df_probs.sort_values("Probabilité (%)", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                )
