# Détection des maladies du riz par vision artificielle

Projet d'intelligence artificielle dédié au diagnostic automatisé des maladies du riz à partir d'images foliaires.

## Structure du projet

```plaintext
rice-disease-rna-cv/
├── pyproject.toml      # Gestion des dépendances Python
├── data/               # Corpus d'images (hébergé sur Google Drive, exclu de Git)
├── notebooks/          # Notebooks Colab d'expérimentation (00 à 04)
├── src/                # Code source Python modulaire (dataset, model, metrics)
└── models/             # Sauvegarde des poids du réseau de neurones
```

## Outils et environnement

* **Google Colab (GPU T4) :** Environnement d'exécution principal. Aucun IDE local (comme VS Code) n'est nécessaire.
* **Google Drive :** Stockage permanent du jeu de données et des modèles.
* **`uv` :** Installation ultra-rapide des packages Python définis dans `pyproject.toml`.
* **Git & GitHub :** Suivi de version collaboratif sur la branche `develop`.

## Configuration des données (Google Drive)

Pour éviter que chaque membre ne ré-uploade les images :

1. Le dossier du projet `rice-disease-rna-cv` est hébergé sur le Google Drive du responsable et partagé avec l'équipe.
2. Chaque membre doit aller dans **« Partagés avec moi »** sur son Google Drive, faire un clic droit sur le dossier `rice-disease-rna-cv` et cliquer sur **« Ajouter un raccourci dans Drive »** (à la racine de *Mon Drive*).

Ainsi, le dossier `data/` sera accessible au même chemin pour toute l'équipe.

## Initialisation dans Google Colab

> 📌 **Consigne :** Pour démarrer votre session, ouvrez et exécutez la **Section 2** du notebook [`notebooks/00_initialisation_colab.ipynb`](notebooks/00_initialisation_colab.ipynb). Elle connecte Google Drive, vous place automatiquement sur la branche `develop`, effectue le `git pull` et installe les dépendances.

## Workflow Git (Routine de travail)

Tout le monde travaille exclusivement sur la branche `develop`.

1. **Début de session (PULL) :** L'exécution de la Section 2 du notebook `00_initialisation_colab.ipynb` vous place automatiquement sur `develop` et récupère les dernières modifications. Si vous souhaitez vérifier manuellement :
   ```bash
   !git checkout develop
   !git pull origin develop
   ```

2. **Travail :** Écrivez et exécutez votre code directement dans votre notebook dans Colab.

3. **Fin de session (PUSH) :** Sauvegardez et envoyez vos modifications sur la branche `develop` :
   ```bash
   !git add .
   !git commit -m "feat: description de votre travail"
   !git push origin develop
   ```
