# 🌾 Documentation - Interface de Détection des Maladies du Riz

**Version :** 0.1.0  
**Dernière mise à jour :** Juillet 2026

---

## 📋 Table des matières

1. [Installation et démarrage](#installation-et-démarrage)
2. [Guide d'utilisation](#guide-dutilisation)
3. [Fonctionnalités](#fonctionnalités)
4. [Interprétation des résultats](#interprétation-des-résultats)
5. [Diagnostics supportés](#diagnostics-supportés)
6. [Recommandations par maladie](#recommandations-par-maladie)
7. [Limitations et bonnes pratiques](#limitations-et-bonnes-pratiques)
8. [Support et assistance](#support-et-assistance)

---

## 🚀 Installation et démarrage

### Prérequis

- **Python** ≥ 3.10
- **pip** ou **uv** pour la gestion des paquets
- **Git** pour le contrôle de version

### Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/votre-org/rice-disease-rna-cv.git
   cd rice-disease-rna-cv
   ```

2. **Installer les dépendances**
   ```bash
   # Avec uv (recommandé)
   uv sync
   
   # Ou avec pip
   pip install -e .
   pip install streamlit
   ```

3. **Télécharger les modèles pré-entraînés**
   ```bash
   # Les modèles sont stockés dans models/
   # Assurez-vous que les fichiers sont présents
   ls models/
   ```

### Lancer l'interface

```bash
# Démarrer l'application Streamlit
streamlit run src/app.py

# L'application s'ouvrira à http://localhost:8501
```

---

## 📖 Guide d'utilisation

### Étape 1 : Préparer l'image

Avant de télécharger une image, vérifiez les points suivants :

✅ **Bonne pratique**
- Image bien éclairée (lumière naturelle ou artificielle uniforme)
- Feuille net et en focus
- Feuille occupe 60-80% de l'image
- Pas d'ombres ou de reflets importants
- Format JPG, PNG ou BMP
- Résolution : 224×224px minimum (recommandé 400×400px+)

❌ **À éviter**
- Images floues ou mal éclairées
- Trop de feuillage en arrière-plan
- Reflets ou ombres dominantes
- Feuilles partiellement coupées
- Formats non supportés (WEBP, GIF, etc.)

### Étape 2 : Télécharger l'image

1. Cliquer sur le bouton **"Choisir une image de feuille de riz"**
2. Sélectionner le fichier depuis votre appareil
3. L'analyse démarre automatiquement

### Étape 3 : Consulter les résultats

L'interface affiche :
- **Image reçue** : Aperçu de l'image téléchargée
- **Diagnostic** : Maladie identifiée (ou "Saine")
- **Confiance** : Pourcentage de certitude de la prédiction
- **Graphique** : Distribution des probabilités pour chaque classe
- **Recommandations** : Actions sugérées

### Étape 4 : Agir en fonction des recommandations

Consulter la section **"Recommandations par maladie"** ci-dessous pour les actions appropriées.

---

## ✨ Fonctionnalités

### Analyse d'images

- **Upload simple** : Interface intuitive pour télécharger des images
- **Traitement rapide** : Diagnostic en ~200 millisecondes
- **Seuil de confiance** : Paramètre ajustable dans la barre latérale

### Visualisation des résultats

- **Graphique en barres** : Probabilités pour chaque maladie
- **Tableau détaillé** : Liste ordonnée des prédictions
- **Indicateur de fiabilité** : "Fiable" (>80%) ou "À vérifier" (<80%)

### Recommandations automatiques

- Suggestions adaptées à la maladie détectée
- Actions prioritaires clairement identifiées
- Liens vers ressources externes (si disponibles)

### Téléchargement de rapports

- Bouton pour générer un PDF/CSV du diagnostic
- Historique des analyses (fonctionnalité future)

---

## 📊 Interprétation des résultats

### Diagnostic principal

Le diagnostic affiché est la classe (maladie) avec la **probabilité la plus élevée**.

```
Exemple :
- Saine : 72%  ← Diagnostic principal
- Cercosporiose : 15%
- Pyriculariose : 8%
- Autres : 5%
```

### Confiance (Confidence Score)

La **confiance** représente la certitude du modèle sur sa prédiction.

| Plage | Interprétation | Action |
|-------|---|---|
| > 85% | **Très fiable** | Agir selon les recommandations |
| 70-85% | **Fiable** | Agir avec observation supplémentaire |
| < 70% | **À vérifier** | Conseil expert recommandé |
| Seuil atteint | **Faible** | Rejeter la prédiction (ajustable) |

### Graphique des probabilités

Montre la **distribution complète** des probabilités pour chaque classe :
- L'axe Y = Probabilité (0 à 1)
- L'axe X = Types de maladies
- Une seule barre haute = Diagnostic très clair
- Plusieurs barres proches = Diagnostic ambigu (vérifier visuellement)

---

## 🔍 Diagnostics supportés

L'interface peut identifier les états suivants :

| État | Symbole | Signification |
|------|---------|---|
| **Saine** | ✅ | La feuille ne présente aucun symptôme détectable |
| **Cercosporiose** | ⚠️ | Taches circulaires brunes avec halo |
| **Pyriculariose** | 🔴 | Taches gris-noir (très contagieuse - urgent) |
| **Brûlure bactérienne** | ⚠️ | Lesions jaunes bordées de rouge |
| **Tache brune** | ⚠️ | Taches brunes diffuses sur la feuille |

---

## 💊 Recommandations par maladie

### ✅ Feuille saine

```
Aucun traitement nécessaire

Actions :
✓ Continuer la surveillance régulière (hebdomadaire)
✓ Maintenir les bonnes pratiques agricoles
✓ Vérifier d'autres feuilles de la plante
```

---

### ⚠️ Cercosporiose (Leaf Spot)

```
Fongicide requis - Sévérité : MOYENNE

Symptômes :
- Taches circulaires (2-5mm) de couleur brunâtre
- Halo jaune caractéristique autour de la tache
- Progression lente mais régulière

Actions prioritaires :
1. Appliquer un fongicide (ex: Propiconazole, Carbendazim)
2. Augmenter l'aération entre les plants
3. Réduire l'humidité foliaire (ne pas arroser le soir)
4. Retirer les feuilles gravement infectées
5. Observer 5-7 jours après traitement

Prévention :
- Rotation des cultures
- Élimination des débris végétaux
- Espacements appropriés
```

---

### 🔴 Pyriculariose (Rice Blast) - URGENT

```
Traitement d'urgence - Sévérité : CRITIQUE

Symptômes :
- Taches ovales gris-noir
- Centre gris clair avec bordure brun-noir
- Progression très rapide (48-72h)
- Très contagieuse

Actions immédiates (24h) :
1. ISOLER les plants infectés
2. Appliquer un fongicide d'urgence (ex: Fénarimol, Tricyclazole)
3. Augmenter drastiquement l'aération
4. Réduire l'humidité (cessez l'arrosage temporairement)
5. Retirer toutes les feuilles infectées
6. Contacter un expert agricole local

Suivi :
- Vérifications quotidiennes les 7 premiers jours
- Traitements répétés si nécessaire (72-96h)
- Destruction des plants trop atteints si possible

Impactée :
- Rendement très réduit (-50% à -100% si non traité)
- Contamination rapide aux plants voisins
```

---

### ⚠️ Brûlure bactérienne (bacterial leaf blight)

```
Traitement obligatoire - Sévérité : ÉLEVÉE

Symptômes :
- Lesions jaunes (premier stade)
- Bordures rouge-brun progressives
- Mucilage jaune translucide au revers
- Aspect "huileux" caractéristique

Actions :
1. Appliquer un traitement antibactérien (ex: Tétracycline, Streptomycine)
2. Réduire l'humidité du sol et foliaire
3. Améliorer le drainage
4. Éviter l'arrosage par aspersion/overhead
5. Espacer les plants davantage
6. Retirer les feuilles très atteintes

Prévention :
- Utiliser des semences certifiées
- Éviter l'excès d'azote dans la fertilisation
- Gestion stricte de l'eau

Suivi : 7-10 jours
```

---

### ⚠️ Tache brune (Brown spot)

```
Fongicide recommandé - Sévérité : MOYENNE à ÉLEVÉE

Symptômes :
- Taches brunes (2-3mm) multiples et diffuses
- Absence de halo caractéristique (différence avec Cercosporiose)
- Distribution uniforme sur la feuille
- Progression lente à modérée

Causes associées :
- Carence en potassium
- Humidité excessive
- Stress hydrique (ironiquement)

Actions :
1. Appliquer un fongicide (ex: Mancozèbe, Captane)
2. Analyser et corriger la nutrition (potassium notamment)
3. Optimiser l'irrigation (ni trop sec, ni trop humide)
4. Augmenter l'aération
5. Retirer les feuilles très atteintes

Suivi : 5-7 jours

Prévention :
- Fertilisation équilibrée
- Gestion rigoureuse de l'eau
```

---

## ⚠️ Limitations et bonnes pratiques

### Limitations du modèle

1. **Qualité de l'image**
   - Le diagnostic dépend entièrement de la qualité de l'image
   - Images très floues ou mal éclairées = résultats peu fiables

2. **Stade de la maladie**
   - Les stades très précoces peuvent ne pas être détectés
   - Les stades très avancés sont généralement bien identifiés

3. **Chevauchement des symptômes**
   - Certaines maladies ont des symptômes similaires
   - Si la confiance est < 70%, vérification manuelle recommandée

4. **Conditions non testées**
   - Dommages mécaniques ou nutritionnels peuvent imiter des maladies
   - Phénomènes physiologiques peuvent confondre le modèle

### ✅ Bonnes pratiques

1. **Toujours vérifier visuellement**
   - Ne jamais baser une décision sur le diagnostic seul
   - Examiner d'autres feuilles et la plante entière
   - Consulter un expert en cas de doute

2. **Utiliser un historique**
   - Garder un journal des diagnostics
   - Comparer avec d'autres plantes de la parcelle
   - Suivre l'évolution après traitement

3. **Prendre plusieurs images**
   - Une seule image peut être trompeuse
   - Analyser 3-5 feuilles de la même plante
   - Comparer les résultats

4. **Documenter les actions**
   - Noter la date du diagnostic
   - Enregistrer les traitements appliqués
   - Observer l'efficacité 5-7 jours après

---

## 📞 Support et assistance

### Dépannage

#### L'application ne démarre pas

```bash
# Vérifier Python
python --version  # Doit être ≥ 3.10

# Reinstaller Streamlit
pip install --upgrade streamlit

# Relancer
streamlit run src/app.py
```

#### Les modèles ne sont pas trouvés

```bash
# Vérifier le dossier models/
ls -la models/

# Télécharger les modèles depuis Drive/Cloud si nécessaire
# (À adapter selon votre setup)
```

#### Résultats suspects ou peu fiables

1. Vérifier la qualité de l'image (lumière, netteté, résolution)
2. Prendre une nouvelle photo dans de meilleures conditions
3. Vérifier que le modèle est à jour
4. Consulter un expert si doute persiste

### Ressources

| Ressource | Lien / Localisation |
|-----------|---|
| **Code source** | `src/` |
| **Notebooks** | `notebooks/` |
| **Modèles** | `models/` |
| **Données (local)** | `data/` (non versionnée) |
| **Règles du projet** | `RULES.md` |
| **README** | `README.md` |

### Signaler un bug

Si vous rencontrez un problème :

1. Décrire le problème en détail
2. Inclure les informations de l'environnement
3. Joindre l'image qui pose problème (si possible)
4. Ouvrir une issue sur GitHub

---

## 📈 Roadmap future

**Fonctionnalités prévues :**

- [ ] Historique des analyses avec graphiques de tendance
- [ ] Export PDF/CSV détaillé des diagnostics
- [ ] Mode batch (analyser plusieurs images)
- [ ] Caméra en direct (intégration webcam)
- [ ] Localisation des lésions sur l'image (heatmap)
- [ ] Support multilingue (FR, EN, Mg, etc.)
- [ ] API HTTP pour intégration externe
- [ ] Mobile app (iOS/Android)
- [ ] Modèles améliorés avec ensemble (ensemble learning)

---

## 📝 Notes

- Cette interface est actuellement en **version de développement (0.1.0)**
- Faire des tests extensifs avant déploiement en production
- Feedback utilisateurs bienvenu pour amélioration continue

---

**Dernière modification :** Juillet 2026  
**Auteur :** Équipe RNA-CV Riz  
**Licence :** À définir

