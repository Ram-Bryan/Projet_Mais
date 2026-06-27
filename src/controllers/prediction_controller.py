"""
Couche Controller : Le Pont entre le Web et l'IA !
Ce fichier reçoit une image du site web, invoque notre modèle IA (conçu à la partie 1 et 3), 
et dicte au système de fichiers de sauvegarder le résultat.
"""
import os
import shutil
import pickle
import numpy as np
import datetime
from src.models.feature_extractor import extraire_features_image

def predire_image(chemin_image):
    """
    Étape 4.1 — Traite une nouvelle image et prédit si elle est malade.
    """
    # 1. On va chercher le "cerveau" entraîné (le Pickle de notre Forêt Scikit-Learn)
    chemin_modele = os.path.join('saved_models', 'best_model.pkl')
    if not os.path.exists(chemin_modele):
        raise FileNotFoundError("Le modèle n'existe pas. Vous devez d'abord lancer 'python main.py' pour construire l'IA (Partie 3C) !")
        
    with open(chemin_modele, 'rb') as f:
        # On charge (on décongèle) l'intelligence artificielle
        modele = pickle.load(f)
        
    # 2. Utilisation du savoir de la Partie 1 pour transformer l'image en 3 variables
    # (Label arbitraire : -1 car on n'en sait rien encore de ce que c'est !)
    donnees = extraire_features_image(chemin_image, label=-1)
    
    if donnees is None:
        raise ValueError("L'image est illisible ou corrompue.")
        
    pct_rouille = donnees["pct_rouille"]
    rugosite = donnees["rugosite"]
    entropie = donnees["entropie_texture"]
    
    # 3. L'algorithme prend sa décision grâce à ses variables formatées (Entre double crochets pour Scikit-Learn)
    X_nouveau = np.array([[pct_rouille, rugosite, entropie]])
    prediction_array = modele.predict(X_nouveau)
    
    prediction_finale = int(prediction_array[0]) # 0 (Saine) ou 1 (Malade)
    
    # 4. Le module d'archivage (Historique) avec format de date demandé
    nom_fichier_original = os.path.basename(chemin_image)
    
    # Obtenir la date courante (YYYYMMDD-HHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if prediction_finale == 1:
        prefixe = f"{timestamp}-MALADE-"
    else:
        prefixe = f"{timestamp}-SAINE-"
        
    # Cela donnera quelque chose comme : 20260101-143000-MALADE-Feuille3.jpg
    nouveau_nom = prefixe + nom_fichier_original
    chemin_final = os.path.join('uploads', nouveau_nom)
    
    os.makedirs('uploads', exist_ok=True)
    # Copie physique de notre fichier temporaire web vers l'archive perpétuelle avec son nouveau nom
    shutil.copy(chemin_image, chemin_final)
    
    return prediction_finale, pct_rouille, rugosite, entropie
