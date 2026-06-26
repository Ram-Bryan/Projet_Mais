"""
Ce fichier fait partie de la couche 'Model' dans l'architecture MVC.
Sa responsabilité unique (SOC) est de traiter les images (données brutes)
et d'en extraire des caractéristiques mathématiques (features) pour notre algorithme.
Aucun affichage (print ou interface graphique) ne doit se faire ici.
"""

# Importation des bibliothèques nécessaires
import os          # Permet de naviguer dans les dossiers et lister les fichiers
import cv2         # OpenCV : bibliothèque de traitement d'images
import numpy as np # NumPy : bibliothèque de calcul mathématique (tableaux, matrices)
import pandas as pd # Pandas : bibliothèque de manipulation de données (tableaux, CSV)
from skimage.measure import shannon_entropy # skimage : scikit-image pour calculer l'entropie

def charger_image(chemin_image):
    """
    Étape 1.1 — Charge une image depuis le disque et la convertit en format RGB.
    
    Explication :
    Par défaut, OpenCV charge les images au format BGR (Bleu, Vert, Rouge).
    Nous le convertissons en RGB (Rouge, Vert, Bleu) qui est le format standard
    et plus intuitif pour la manipulation de couleurs.
    """
    # Vérifier d'abord si le fichier existe vraiment
    if not os.path.exists(chemin_image):
        # Si le fichier n'existe pas, on retourne None avec une erreur silencieuse (pas de print)
        return None
    
    # Lecture de l'image (format BGR par défaut pour cv2)
    image_bgr = cv2.imread(chemin_image)
    
    # Vérification que l'image a bien été chargée sans erreur
    if image_bgr is None:
        return None
        
    # Conversion de l'image de l'espace colorimétrique BGR vers RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb

def extraire_pct_rouille(image):
    """
    Étape 1.2 — Calcule le pourcentage de pixels de couleur "rouille" dans l'image.
    
    Explication :
    Nous convertissons l'image au format HSV (Hue, Saturation, Value).
    L'espace HSV sépare la teinte (la couleur pure) de l'éclairage, 
    ce qui le rend très efficace pour isoler une couleur spécifique.
    Ici, nous créons des 'masques' pour isoler l'orange et le marron.
    """
    # Conversion de l'image RGB vers l'espace colorimétrique HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Plage 1 : Couleur orangée
    # En OpenCV, H (Hue) va de 0 à 179. S et V vont de 0 à 255.
    borne_basse_orange = np.array([10, 50, 50])
    borne_haute_orange = np.array([25, 255, 255])
    
    # Création d'un masque binaire (1=orange, 0=autre couleur)
    masque_orange = cv2.inRange(image_hsv, borne_basse_orange, borne_haute_orange)
    
    # Plage 2 : Couleur brun/marron
    borne_basse_marron = np.array([5, 40, 30])
    borne_haute_marron = np.array([15, 255, 255])
    
    # Création du deuxième masque binaire
    masque_marron = cv2.inRange(image_hsv, borne_basse_marron, borne_haute_marron)
    
    # Fusion des deux masques (soit c'est orange, soit c'est marron)
    # L'opération bitwise_or prend le pixel actif si l'un ou l'autre masque est actif
    masque_fusion = cv2.bitwise_or(masque_orange, masque_marron)
    
    # Compter le nombre de pixels blancs dans notre masque fusionné
    pixels_rouille = np.count_nonzero(masque_fusion)
    
    # Compter le nombre total de pixels dans l'image (hauteur * largeur)
    total_pixels = image.shape[0] * image.shape[1]
    
    # Calcul du ratio (pourcentage entre 0 et 1)
    pct_rouille = pixels_rouille / total_pixels
    
    return float(pct_rouille)

def extraire_rugosite(image):
    """
    Étape 1.3 — Mesure la rugosité de l'image via la variance des gradients (filtre de Sobel).
    
    Explication :
    Le filtre de Sobel détecte les contours et irrégularités (changements brusques d'intensité).
    Plus la feuille est rugueuse (présence de pustules de rouille), plus il y aura
    des variations brusques. La variance de la magnitude totale exprime cette rugosité.
    """
    # Conversion de l'image en niveaux de gris (simplification colorimétrique)
    image_gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Application du filtre Sobel sur l'axe X (horizontal) pour capter les gradients
    # CV_64F est un type 'float' de haute précision pour garder le sens positif/négatif
    sobel_x = cv2.Sobel(image_gris, cv2.CV_64F, 1, 0, ksize=3)
    
    # Application du filtre Sobel sur l'axe Y (vertical)
    sobel_y = cv2.Sobel(image_gris, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcul de la magnitude exacte de ces deux vecteurs (Théorème de Pythagore)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calcul de la variance mathématique de la magnitude
    rugosite = np.var(magnitude)
    
    return float(rugosite)

def extraire_entropie_texture(image):
    """
    Étape 1.4 — Calcule l'entropie de Shannon de l'image (mesure de désordre visuel).
    
    Explication :
    L'entropie quantifie l'uniformité des pixels.
    Si les pixels se ressemblent tous (feuille saine uniforme), l'entropie est très faible.
    Si les pixels sont chaotiques (feuille malade à pustules multiples), l'entropie est forte.
    """
    # Conversion au préalable en image niveaux de gris
    image_gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calcul grâce à l'algorithme de Shannon (disponible depuis skimage.measure)
    entropie = shannon_entropy(image_gris)
    
    return float(entropie)

def extraire_features_image(chemin_image, label):
    """
    Étape 1.5 — Orchestre l'extraction des 3 caractéristiques pour une seule image.
    
    Explication :
    Gèront l'assemblage complet du pipeline d'une image.
    Nous appelerons successivement toutes nos fonctions d'extractions pour former un dictionnaire.
    """
    # Étape 1: Charger l'image proprement
    img = charger_image(chemin_image)
    
    # Si le chargement échoue, on arrête là l'extraction
    if img is None:
        return None
        
    # Étape 2: Calcul des 3 "Features" algorithmiques
    pct_rouille = extraire_pct_rouille(img)
    rugosite = extraire_rugosite(img)
    entropie = extraire_entropie_texture(img)
    
    # Récupérer juste le nom du fichier (et non le chemin complet vers les répertoires)
    nom_fichier = os.path.basename(chemin_image)
    
    return {
        "ID_Image": nom_fichier,
        "pct_rouille": pct_rouille,
        "rugosite": rugosite,
        "entropie_texture": entropie,
        "label_malade": label
    }

def construire_dataframe(dossier_saines, dossier_malades):
    """
    Étape 1.6 — Parcourt tous les dossiers d'images et construit le tableau final en CSV.
    
    Explication :
    À partir de multiples images, nous allons structurer tous nos "dictionnaires" en
    un seul et même tableau global pour que la Partie 2 puisse alimenter notre intelligence artificielle.
    """
    toutes_les_lignes = []
    
    # --- Traitement des images saines (label technique = 0) ---
    if os.path.exists(dossier_saines):
        for nom_fichier in os.listdir(dossier_saines):
            # Vérifier l'extension pour ignorer les fichiers système (ex: .DS_Store sous Mac)
            if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                chemin_complet = os.path.join(dossier_saines, nom_fichier)
                
                # Le label attribué par définition de la catégorie est `0`
                donnees_image = extraire_features_image(chemin_complet, label=0)
                
                if donnees_image is not None:
                    toutes_les_lignes.append(donnees_image)
                    
    # --- Traitement des images malades (label technique = 1) ---
    if os.path.exists(dossier_malades):
        for nom_fichier in os.listdir(dossier_malades):
            if nom_fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                chemin_complet = os.path.join(dossier_malades, nom_fichier)
                
                # Le label attribué par définition de la catégorie est `1`
                donnees_image = extraire_features_image(chemin_complet, label=1)
                
                if donnees_image is not None:
                    toutes_les_lignes.append(donnees_image)
                    
    # Si on n'a trouvé ou réussi à charger aucune image, on prévient en retournant un df vide.
    if len(toutes_les_lignes) == 0:
        # Renvoie un dataframe à colonnes justes même s'il est vide
        return pd.DataFrame(columns=["ID_Image", "pct_rouille", "rugosite", "entropie_texture", "label_malade"])

    # Conversion de cette vaste liste en objet DataFrame (Tableau)
    df_features = pd.DataFrame(toutes_les_lignes)
    
    # Préparation avant la sauvegarde: création de "data" si pas encore fait
    os.makedirs('data', exist_ok=True)
    chemin_csv = os.path.join('data', 'features.csv')
    
    # Écriture du fichier au format CSV dans /data/. index=False enlève la numérotation automatique native.
    df_features.to_csv(chemin_csv, index=False)
    
    return df_features
