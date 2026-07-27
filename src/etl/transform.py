import os
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

# Taille cible : 224*224 (standard pour les CNN - ImageNet)

TAILLE_CIBLE = (224, 224)

# Dossier de sortie : data/donne_propre/

CHEMIN_PROPRE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "donne_propre"
)

# Qualité de compression JPEG (1 à 100) :

QUALITE_JPEG = 85

# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def transformer_images(liste_images):
    """
    Reçoit la liste retournée par extract.py.
    Pour chaque image :
        1. Ouvre l'image originale
        2. Recadrage intelligent (crop carré centré)
        3. Redimensionnement vers 224x224
        4. Conversion en RGB (supprime canal alpha si PNG)
        5. Normalisation : sauvegardée implicitement en JPEG [0,255]
           → la vraie normalisation [0,1] se fera dans data_loader.py
           car les tenseurs PyTorch/TensorFlow gèrent ça mieux que Pillow
        6. Sauvegarde dans data/donne_propre/<classe>/

    Paramètres :
        liste_images (list) : liste de dicts venant d'extract.py

    Retourne :
        rapport (dict) : statistiques de la transformation
    """
    
    print("\n" + "=" *60)
    print(" TRANSFORMATION DES IMAGES VERS DONNEES PROPRES")
    print("=" *60)
    
    # Créer le dossier donne_propre s'il n'existe pas
    os.makedirs(CHEMIN_PROPRE, exist_ok=True)
    
    # Compteurs pour le rapport final
    succes = 0
    echecs = 0
    erreurs = []
    
    # Regrouper par classe pour afficher la progression par classe
    classes = {}
    
    for img in liste_images:
        classe = img["classe"]
        if classe not in classes:
            classes[classe] = []
        classes[classe].append(img)
    
    # ============================================================
    # FONCTION PRINCIPALE
    # ============================================================
    
    for nom_classe, images in classes.items():
        
        # Créer le sous-dossier de la classe dans le donne_prore
        dossier_classe_propre = os.path.join(CHEMIN_PROPRE, nom_classe)
        os.makedirs(dossier_classe_propre, exist_ok=True)
        
        print(f"\n -> Traitement : {nom_classe}  ({len(images)} images)")
        
        for info in images:
            try:
                # --------------------------------------------
                # ÉTAPE 1 : Ouvrir l'image
                # --------------------------------------------
                img = Image.open(info["chemin"])
                
                # --------------------------------------------
                # ÉTAPE 2 : Conversion RGB
                # Supprime le canal alpha (transparence) des PNG
                # Le modèle CNN attend toujours 3 canaux : R, G, B
                # --------------------------------------------
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                # --------------------------------------------
                # ÉTAPE 3 : Crop carré centré
                # Évite la déformation si l'image n'est pas carrée
                # Ex : 400x300 → on prend le carré 300x300 au centre
                # --------------------------------------------
                largeur, hauteur = img.size
                cote = min(largeur, hauteur) # le plus petit cote
                gauche = (largeur - cote) // 2
                haut = (hauteur - cote) // 2
                droite = gauche + cote
                bas = haut + cote
                img = img.crop((gauche, haut, droite, bas))
                
                # --------------------------------------------
                # ÉTAPE 4 : Redimensionnement 224x224
                # LANCZOS = meilleur algorithme pour réduire
                # une image sans perdre les détails importants
                # --------------------------------------------
                img = img.resize(TAILLE_CIBLE, Image.LANCZOS)
                
                # --------------------------------------------
                # ÉTAPE 5 : Sauvegarde en JPEG compressé
                # quality=85 : excellent compromis
                # optimize=True : Pillow cherche la meilleure
                # compression possible pour ce niveau de qualité
                # --------------------------------------------
                nom_sortie = os.path.splitext(info["nom_fichier"])[0] + ".jpg"
                chemin_sortie = os.path.join(dossier_classe_propre, nom_sortie)
                
                img.save(chemin_sortie, format="JPEG", quality=QUALITE_JPEG, optimize=True) 
                
                succes += 1
                
            except Exception as e:
                # Si une image est corrompue, on note l'erreur
                # et on continue sans bloquer tout le pipeline
                echecs += 1
                erreurs.append({
                    "fichier" : info["nom_fichier"],
                    "classe"  : info["classe"],
                    "erreur"   : str(e)
                })
        print(f"{len(images)} images sauvegardées dans donne_propre/{nom_classe}/")
    
    # --------------------------------------------------------
    # Rapport final
    # --------------------------------------------------------
    rapport = _afficher_rapport(succes, echecs, erreurs)
    return rapport

# ============================================================
# RAPPORT FINAL
# ============================================================

def _afficher_rapport(succes, echecs, erreurs):
    """ Afficher le bilan de la transformation."""
    
    print("\n" + "=" *60)
    print(" RAPPORT DE LA TRANSFORMATION")
    print("=" *60)
    print(f"Images transformées avec succès : {succes}")
    print(f"Images échouées : {echecs}")
    
    if erreurs:
        print("\n Detail des erreurs :")
        for e in erreurs:
            print(f" - {e['classe']}/{e['fichier']} : {e['erreur']}")
            
    taille_totale = _calculer_taille_dossier()
    
    print(f"\n Taille totale donne_propre : {taille_totale:.2f} Mo")
    print(" Forma                          : JPEG, 224x224, RGB")
    print(f" Qualité de compression          : {QUALITE_JPEG}/95")
    print("=" *60 + "\n")
    
    return {
        "succes": succes,
        "echecs": echecs,
        "erreurs": erreurs,
    }
    
def _calculer_taille_dossier():
    """ Calcule la taille totale du dossier donne_propre/ en Mo."""
    total = 0
    for racine, dossiers, fichiers in os.walk(CHEMIN_PROPRE):
        for f in fichiers:
            total += os.path.getsize(os.path.join(racine, f))
    return total / (1024 * 1024)  # Convertir en Mo

# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    # On import extract ici uniquement pour le test direct
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.etl.extract import extraire_infos_donnees
    
    liste = extraire_infos_donnees()
    rapport = transformer_images(liste)
    print(f" -> Pipeline ETL terminé : {rapport['succes']} images prêtes pour le ML")
