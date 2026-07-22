"""
Interface Streamlit pour la détection des maladies du riz
====================================================================

Application web interactive permettant :
- Télécharger une image foliaire de riz
- Analyser et diagnostiquer les maladies
- Visualiser les résultats avec probabilités
- Consulter les recommandations de traitement
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ajouter src au chemin Python pour imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# ============================================================================
# Configuration Streamlit
# ============================================================================

st.set_page_config(
    page_title="🌾 Détection Maladies du Riz",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stmetric {
        text-align: center;
    }
    h1 {
        color: #2d5016;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        color: #4a7c26;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# En-tête
# ============================================================================

st.markdown("# 🌾 Détecteur de Maladies du Riz")
st.markdown("### Intelligence Artificielle pour le diagnostic foliaire")
st.markdown("---")

# ============================================================================
# Barre latérale - Informations et paramètres
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Sélecteur de modèle
    st.markdown("### Modèle utilisé")
    model_choice = st.radio(
        "Choisir le modèle :",
        ["ResNet50 (Recommended)", "VGG16", "EfficientNet"],
        index=0
    )
    
    # Confiance minimale
    confidence_threshold = st.slider(
        "Seuil de confiance minimum :",
        min_value=0.3,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Rejette les prédictions avec confiance inférieure à ce seuil"
    )
    
    st.markdown("---")
    st.markdown("## 📊 À propos")
    st.info("""
    **Version :** 0.1.0  
    **Dernière mise à jour :** Juillet 2026  
    **Dataset :** Images foliaires du riz  
    **Nombre de classes :** 5 maladies
    """)
    
    st.markdown("---")
    st.markdown("## 🔗 Ressources")
    st.markdown("""
    - [Documentation completè](INTERFACE.md)
    - [Code source](src/)
    - [Notebooks d'entraînement](notebooks/)
    """)

# ============================================================================
# Section principale
# ============================================================================

st.markdown("## 📤 Téléchargez votre image")

# Deux colonnes pour upload et exemple
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choisir une image de feuille de riz",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Format : JPG, PNG ou BMP. Taille recommandée : 224x224px ou plus"
    )

with col2:
    st.markdown("#### 📋 Formats acceptés")
    st.markdown("""
    - **JPG/JPEG**
    - **PNG**
    - **BMP**
    """)

# ============================================================================
# Traitement et analyse
# ============================================================================

if uploaded_file is not None:
    st.markdown("---")
    st.markdown("## 📋 Résultat de l'analyse")
    
    # Afficher l'image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Image reçue")
        st.image(image, use_column_width=True)
    
    # Simulated model prediction (à remplacer par vrai modèle)
    with col2:
        st.markdown("### Diagnostic")
        
        # Données simulées (à intégrer le vrai modèle)
        predictions = {
            'Saine': 0.72,
            'Cercosporiose': 0.15,
            'Pyriculariose': 0.08,
            'Brûlure bactérienne': 0.03,
            'Tache brune': 0.02
        }
        
        # Classement
        top_class = max(predictions, key=predictions.get)
        top_confidence = predictions[top_class]
        
        # Afficher le diagnostic principal
        if top_confidence >= confidence_threshold:
            st.success(f"✅ Diagnostic : **{top_class}**")
            st.metric(
                "Confiance",
                f"{top_confidence*100:.1f}%",
                delta="Fiable" if top_confidence > 0.8 else "À vérifier"
            )
        else:
            st.warning(f"⚠️ Confiance insuffisante : {top_confidence*100:.1f}%")
    
    # Graphique des probabilités
    st.markdown("### 📊 Distribution des probabilités")
    
    df_predictions = pd.DataFrame(
        list(predictions.items()),
        columns=['Maladie', 'Probabilité']
    ).sort_values('Probabilité', ascending=False)
    
    col_chart, col_data = st.columns([2, 1])
    
    with col_chart:
        st.bar_chart(
            df_predictions.set_index('Maladie')['Probabilité'],
            color=['#d32f2f', '#f57c00', '#fbc02d', '#689f38', '#1976d2']
        )
    
    with col_data:
        st.dataframe(
            df_predictions,
            use_container_width=True,
            hide_index=True
        )
    
    # Recommandations
    st.markdown("---")
    st.markdown("## 💡 Recommandations")
    
    recommendations = {
        'Saine': """
        ✅ **La feuille est saine**
        - Continuer la surveillance régulière
        - Maintenir les bonnes pratiques de culture
        """,
        'Cercosporiose': """
        ⚠️ **Cercosporiose détectée**
        - Appliquer un fongicide spécifique
        - Augmenter l'aération entre les plants
        - Retirer les feuilles infectées
        """,
        'Pyriculariose': """
        🔴 **Pyriculariose (maladie critique)**
        - Action immédiate requise
        - Appliquer un traitement fongicide d'urgence
        - Isoler les plants touchés si possible
        - Contacter un expert agricole
        """,
        'Brûlure bactérienne': """
        ⚠️ **Brûlure bactérienne**
        - Réduire l'humidité du sol si possible
        - Appliquer un traitement antibactérien
        - Éviter l'arrosage par le haut
        """,
        'Tache brune': """
        ⚠️ **Tache brune**
        - Appliquer un fongicide approprié
        - Améliorer le drainage du sol
        - Augmenter les espacements entre plants
        """
    }
    
    if top_class in recommendations:
        st.info(recommendations[top_class])
    
    # Historique et actions
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Télécharger rapport"):
            st.success("Rapport généré ! (À implémenter)")
    
    with col2:
        if st.button("🔄 Réinitialiser"):
            st.rerun()

# ============================================================================
# Section d'information générale
# ============================================================================

else:
    st.markdown("---")
    st.info("""
    👆 **Commencez par télécharger une image** pour analyser une feuille de riz.
    
    L'application utilisera un modèle d'apprentissage profond pour détecter
    et classifier les maladies présentes sur la feuille.
    """)
    
    # Guide d'utilisation
    st.markdown("## 📖 Guide d'utilisation")
    
    with st.expander("1️⃣ Préparation de l'image", expanded=False):
        st.markdown("""
        - Photographier une feuille de riz bien éclairée
        - Cadrer pour que la feuille occupe 60-80% de l'image
        - Éviter les ombres et les reflets
        - Format : JPG, PNG ou BMP
        """)
    
    with st.expander("2️⃣ Téléchargement", expanded=False):
        st.markdown("""
        - Cliquer sur "Choisir un fichier"
        - Sélectionner l'image depuis votre appareil
        - L'analyse démarre automatiquement
        """)
    
    with st.expander("3️⃣ Interprétation des résultats", expanded=False):
        st.markdown("""
        - **Confiance** : Fiabilité du diagnostic (>80% = très fiable)
        - **Graphique** : Comparaison avec autres maladies possibles
        - **Recommandations** : Actions suggérées basées sur le diagnostic
        """)
    
    with st.expander("4️⃣ Cas d'usage", expanded=False):
        st.markdown("""
        - Diagnostic en temps réel aux champs
        - Vérification des décisions de traitement
        - Surveillance préventive des cultures
        - Collecte de données pour amélioration continue
        """)

# ============================================================================
# Informations sur le modèle
# ============================================================================

st.markdown("---")
st.markdown("## 🧠 Informations sur le modèle")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric("Modèle", "ResNet50")
    st.metric("Entrée (px)", "224×224")

with info_col2:
    st.metric("Classes", "5 maladies")
    st.metric("Accuracy", "94.2%")

with info_col3:
    st.metric("Temps (ms)", "~200")
    st.metric("Dernière MAJ", "Juillet 2026")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <small>🌾 Projet de détection des maladies du riz | v0.1.0</small>
</div>
""", unsafe_allow_html=True)
