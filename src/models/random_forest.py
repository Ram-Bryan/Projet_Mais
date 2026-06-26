"""
Ce fichier fait partie de la couche 'Model' dans l'architecture MVC.
Sa responsabilité unique est de planter une véritable "Forêt Aléatoire"
en se basant sur de multiples de nos "Arbres de Décision" faits maison.
"""

import numpy as np
from src.models.decision_tree import construire_arbre, predire_tous

def echantillonner_avec_remplacement(X, y):
    """
    Étape 3B.1 — C'est la technique du "Bagging". Au lieu de donner toutes les données 
    à un seul arbre, on tire au sort (avec remise) un échantillon pour chaque arbre.
    """
    X = np.array(X)
    y = np.array(y)
    n_echantillons = len(X)
    
    # On tire `n_echantillons` indices aléatoires. 
    # replace=True veut dire qu'on a le droit de piocher plusieurs fois le même ticket !
    indices_aleatoires = np.random.choice(n_echantillons, size=n_echantillons, replace=True)
    
    # On extrait nos fausses données légèrement déformées/redondantes
    X_echantillon = X[indices_aleatoires]
    y_echantillon = y[indices_aleatoires]
    
    return X_echantillon, y_echantillon

def entrainer_foret(X, y, n_arbres=10, max_profondeur=5):
    """
    Étape 3B.2 — La pépinière. On va construire plusieurs arbres indépendants.
    C'est la différence entre le "Decision Tree" et le "Random Forest".
    """
    liste_arbres = []
    
    # On plante autant d'arbres que l'utilisateur a demandé
    for i in range(n_arbres):
        
        # 1. On "modifie" mentalement les données (Bagging)
        X_echantillon, y_echantillon = echantillonner_avec_remplacement(X, y)
        
        # 2. On fait grandir notre arbre sur cette vision tronquée de la réalité
        arbre = construire_arbre(X_echantillon, y_echantillon, profondeur=0, max_profondeur=max_profondeur)
        
        # 3. L'arbre est grand, on l'ajoute à la Forêt
        liste_arbres.append(arbre)
        
    return liste_arbres

def predire_foret(liste_arbres, X):
    """
    Étape 3B.3 — Le moment du vote (Vote Majoritaire).
    Pour classifier une image, chaque arbre donne son avis, et la majorité l'emporte !
    """
    X = np.array(X)
    
    # matrice qui va contenir le vote de TOUS les arbres pour TOUTES les images.
    tous_les_votes = []
    for arbre in liste_arbres:
        # Chaque arbre prononce sa sentence (un tableau de 0 et de 1)
        predictions = predire_tous(arbre, X)
        tous_les_votes.append(predictions)
        
    # On transforme ça en tableau Numpy et on le pivote (.T = Transposé)
    # Pour que chaque ligne corresponde à 1 image, et les colonnes soient les votes des N arbres
    tous_les_votes = np.array(tous_les_votes).T
    
    predictions_finales = []
    # On passe en revue les urnes, image par image
    for ligne_de_votes in tous_les_votes:
        ligne_int = ligne_de_votes.astype(int)
        
        # bincount compte les voix pour 0 et pour 1. argmax désignera le vainqueur !
        vote_final = np.argmax(np.bincount(ligne_int))
        predictions_finales.append(vote_final)
        
    return np.array(predictions_finales)
