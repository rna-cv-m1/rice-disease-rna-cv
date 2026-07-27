import os

# configuration de chemin vers les données brutes

# os.path.dirname(__file__) = notre emplacement dans extract.py
# on remonte deux fois (..) pour arriver à la racine du projet
# puis on descend vers data/donner_brute

CHEMIN_BRUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "donner_brute"
)

# Extraction d'image qu'on accepte

EXTENSIONS_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ============================================================
# FONCTION PRINCIPALE : extraire les infos des données brutes
# ============================================================

def extraire_infos_donnees():
    
    print("\n" + "=" *60)
    print(" EXTRACTION ET EXPLORATION DES DONNEES BRUTES")
    print("=" *60)
    
    if not os.path.exists(CHEMIN_BRUT):
        print(f"[ERREUR] Dossier brut introuvalble : {CHEMIN_BRUT}")
        return []
    
    # Etape 1 : parcourir chaque classe (sous-dossier)
    
    liste_images = []  #contiendra une entrée par image
    stats_par_classe = {} #pour le tableau récapitulatif
    
    classes = sorted(os.listdir(CHEMIN_BRUT)) # ordre alphabétique
    
    for nom_classe in classes :
        chemin_classe = os.path.join(CHEMIN_BRUT, nom_classe)
        
        # Ignorer si ce n'est pas un dossier
        if not os.path.isdir(chemin_classe):
            continue
        
        # Etape 2 : parcourir les images dans chaque classe
        
        image_de_la_classe = []
        
        for nom_fichier in os.listdir(chemin_classe):
             extension = os.path.splitext(nom_fichier)[1].lower()
             
             if extension not in EXTENSIONS_IMAGE:
                 continue
             chemin_complet = os.path.join(chemin_classe, nom_fichier)
             
             # Taille du fichier en kilo-octets
             
             # Stocker les info de cette image
             
             info_image = {
                 "classe": nom_classe,
                 "nom_fichier": nom_fichier,
                 "chemin": chemin_complet,
                 "taille_ko": os.path.getsize(chemin_complet) / 1024
             }
             
             image_de_la_classe.append(info_image)
             liste_images.append(info_image)
             
        # Etape 3 : calcule les stats pour cette classe
             
        if image_de_la_classe :
            tailles = [img["taille_ko"] for img in image_de_la_classe]
            stats_par_classe[nom_classe] = {
                "nombre"            : len(image_de_la_classe),
                "taille_min_ko"     : round(min(tailles), 2),
                "taille_max_ko"     : round(max(tailles), 2),
                "taille_moy_ko"     : round(sum(tailles)/len(tailles), 2)
            }             
        
        # Etape 4 : afficher le tableau récapitulatif
        
    _afficher_tableau(stats_par_classe, liste_images)
    
    return liste_images

# ============================================================
# FONCTION D'AFFICHAGE : tableau statistique
# ============================================================

def _afficher_tableau(stats_par_classe, liste_images):
    
    print(f"\n{'Classe':<25} {'Nb images':>10} {'Min (Ko)':>10} "
          f"{'Max (Ko)':>10} {'Moy (Ko)':>10}")
    print("-" * 68)
    
    for classe, stats in stats_par_classe.items():
        print(
            
            f"{classe:<25} {stats['nombre']:>10} "
            f"{stats['taille_min_ko']:>10} "
            f"{stats['taille_max_ko']:>10} "
            f"{stats['taille_moy_ko']:>10}"
        )
    
    print("-" * 68)
    
    # Totaux
    
    total_images = sum(stats['nombre'] for stats in stats_par_classe.values())
    image_plus_lourde = max(liste_images, key= lambda x: x['taille_ko'])
    
    print(f"\nTOTAL images            : {total_images}")
    print(f" Image la plus lourde    : {image_plus_lourde['nom_fichier']}")
    print(f" Classe                   : {image_plus_lourde['classe']}")
    print(f" Taille (Ko)              : {round(image_plus_lourde['taille_ko'], 2)}")
    
    print("=" * 68 + "\n")
    
    
# ============================================================
# TEST RAPIDE : exécuter ce fichier directement pour vérifier
# ============================================================

if __name__ == "__main__":
    donnees = extraire_infos_donnees()
    print(f"  → {len(donnees)} images récupérées et prêtes pour transform.py")
    
    