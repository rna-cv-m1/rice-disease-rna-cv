# Détection des maladies du riz par vision artificielle

Projet d'intelligence artificielle dédié au diagnostic automatisé des maladies du riz à partir d'images foliaires.

---

## Structure du projet

```plaintext
rice-disease-rna-cv/
├── pyproject.toml      # Gestion des dépendances Python (géré via uv)
├── data/               # Corpus d'images (exclu de Git)
├── notebooks/          # Notebooks Colab d'expérimentation (00 à 04)
├── src/                # Code source Python modulaire
├── models/             # Sauvegarde des poids du réseau de neurones
└── RULES.md            # Directives et règles indispensables pour l'équipe
```

---

## Outils et environnement

| Outil | Rôle / Usage |
| :--- | :--- |
| **Google Colab** | Environnement d'exécution principal avec GPU (T4). |
| **uv** | Gestionnaire de dépendances ultra-rapide basé sur `pyproject.toml`. |
| **Google Drive** | Stockage permanent des poids de modèles et des checkpoints. |
| **Git & GitHub** | Suivi de version collaboratif sur la branche `develop`. |

---

<div style="background-color: #f7fafc; border-left: 3px solid #4a5568; padding: 10px 14px; margin: 16px 0; border-radius: 4px;">
  <span style="color: #2d3748;">Directives et Règles du Projet</span><br>
  Pour les consignes de sécurité, la gestion du stockage et la routine de développement, consultez le document dédié :<br>
  <a href="RULES.md">Fichier des Directives & Règles (RULES.md)</a>
</div>
