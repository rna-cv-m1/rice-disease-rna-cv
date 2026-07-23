# Directives et Règles du Projet (Google Colab & Git)

Ce document récapitule les consignes de sécurité, les avertissements et la routine de développement à appliquer sur Google Colab.

---

## Avertissements Majeurs

<div style="background-color: #fcf8f8; border-left: 3px solid #e53e3e; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <span style="color: #9b2c2c;">1. Stockage Colab éphémère & Sauvegarde des travaux</span><br>
  Le disque local <code>/content/</code> est réinitialisé à chaque déconnexion (inactivité > 90 min ou session > 12 h). Tout travail en cours (avancement du projet, pourcentage d'entraînement, fichiers modifiés) sera définitivement supprimé par Colab s'il n'est pas commité sur Git ou sauvegardé sur Google Drive.
</div>

<div style="background-color: #fffaf0; border-left: 3px solid #dd6b20; padding: 10px 14px; margin: 12px 0; border-radius: 4px;">
  <span style="color: #9c4221;">2. Interdiction de commiter le jeu de données (<code>data/</code>)</span><br>
  Le dossier <code>data/</code> (> 1 Go) est exclu par <code>.gitignore</code>. Ne jamais forcer son ajout dans Git (<code>git add -f data/</code>), sous peine de bloquer définitivement le dépôt distant.
</div>

---

## Routine de Travail & Workflow

### Règle 1 : Démarrage de session (Initialisation)
Au lancement de chaque session Colab, ouvrir le notebook <a href="notebooks/00_initialisation_colab.ipynb"><code>notebooks/00_initialisation_colab.ipynb</code></a> et exécuter :
- **Section 2** : Pour synchroniser le code source (`git pull origin develop`) et télécharger le dataset Kaggle dans `data/raw/`.
- **Section 3** : Pour consulter le guide de configuration des clés API (`GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) via les secrets Colab (`userdata`).

### Règle 2 : Workflow Git & Sauvegardes Fréquentes
- Développer exclusivement sur la branche <code>develop</code>.
- N'oubliez pas de sauvegarder vos notebooks et d'effectuer des commits régulièrement pour ne pas perdre l'avancement de votre travail :
  ```bash
  git add .
  git commit -m "feat: description de l'avancement"
  git push origin develop
  ```

### Règle 3 : Checkpoints & Sauvegarde de la Progression d'Entraînement
- Monter Google Drive (<code>drive.mount('/content/drive')</code>) et enregistrer les poids du modèle (<code>.pth</code> ou <code>.h5</code>) ainsi que la progression (pourcentage/époques) à la fin de chaque époque.
- Configurer les scripts pour charger automatiquement le dernier checkpoint disponible afin d'éviter tout recommencement en cas de fermeture automatique de la session.
