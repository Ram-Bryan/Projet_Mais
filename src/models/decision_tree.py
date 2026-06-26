"""
Ce fichier fait partie de la couche 'Model' dans l'architecture MVC.
Sa responsabilité est de construire un "Arbre de Décision" de toutes pièces (From Scratch)
en utilisant notre métrique Max-Minority créée dans la Partie 2.
"""

import numpy as np
from src.models.max_minority import trouver_meilleur_split, calculer_purete

class Noeud:
    """
    Étape 3A.1 — Structure de données représentant un "Nœud" de notre arbre.
    
    Explication :
    Un arbre comprend des "Nœuds de question" et des "Feuilles de réponse". 
    Un nœud de question teste (ex: "La rugosité <= 17.5 ?") => il aura un `feature_index`, un `seuil`, et des enfants `gauche/droite`.
    Un nœud de réponse n'a pas d'enfants, mais il porte la décision `prediction` (0 = saine, 1 = malade).
    """
    def __init__(self, feature_index=None, seuil=None, gauche=None, droite=None, prediction=None):
        self.feature_index = feature_index  # Index de la colonne que l'on vérifie (ex: 1 pour rugosité)
        self.seuil = seuil                  # Valeur seuil (ex: 17.5)
        self.gauche = gauche                # Nœud enfant contenant ceux qui sont <= seuil
        self.droite = droite                # Nœud enfant contenant ceux qui sont > seuil
        self.prediction = prediction        # Si c'est le point final, on arrête et on met 0 ou 1

def trouver_meilleur_split_global(X, y):
    """
    Étape 3A.2 — Teste toutes les colonnes de notre jeu de données pour trouver la MEILLEURE
    caractéristique et le MEILLEUR seuil qui isoleront au maximum les feuilles malades.
    """
    X = np.array(X)
    y = np.array(y)
    
    meilleure_feature_index = None
    meilleur_seuil_global = None
    purete_maximale_globale = -1.0
    
    # X.shape[1] correspond au nombre de colonnes de caractéristiques
    n_features = X.shape[1]
    
    # On boucle sur chacune des colonnes (ex: index 0: pct_rouille, 1: rugosité, etc.)
    for feature_index in range(n_features):
        X_colonne = X[:, feature_index]
        
        # On fait appel à la magie mathématique de notre Partie 2 !
        seuil_trouve, purete_obtenue = trouver_meilleur_split(X_colonne, y)
        
        # On vérifie si ce seuil a une meilleure pureté que tout ce qu'on a vu précédemment
        if seuil_trouve is not None and purete_obtenue > purete_maximale_globale:
            purete_maximale_globale = purete_obtenue
            meilleure_feature_index = feature_index
            meilleur_seuil_global = seuil_trouve
            
    return meilleure_feature_index, meilleur_seuil_global

def construire_arbre(X, y, profondeur=0, max_profondeur=5):
    """
    Étape 3A.3 — Construction de l'arbre DE MANIÈRE RÉCURSIVE.
    
    Explication de la récursivité :
    Cette fonction va s'appeler elle-même pour traiter les petits sous-groupes crées lors 
    d'un découpage. Elle ne s'arrête que si elle tombe sur de bonnes "conditions d'arrêt".
    """
    X = np.array(X)
    y = np.array(y)
    
    # Éviter les plantages au cas où y est vide
    if len(y) == 0:
        return Noeud(prediction=0)
        
    # Calcul au préalable de la "classe majoritaire".
    # np.bincount compte combien il y a de 0 et de 1. np.argmax choisit l'indice du plus grand tas.
    y_int = y.astype(int)
    classe_majoritaire = np.argmax(np.bincount(y_int))
    
    # --- Condition d'arrêt 1 : Données 100% Pures ---
    if calculer_purete(y) >= 1.0:
        return Noeud(prediction=classe_majoritaire)
        
    # --- Condition d'arrêt 2 : Taille max atteinte ---
    # Pour ne pas que notre arbre devienne énorme et apprenne "par cœur" au lieu de généraliser.
    if profondeur >= max_profondeur:
        return Noeud(prediction=classe_majoritaire)
        
    # --- Condition d'arrêt 3 : Trop peu de données ---
    if len(X) < 2:
        return Noeud(prediction=classe_majoritaire)
        
    # On demande de chercher comment on devrait couper les données à cette étape !
    feature_index, seuil = trouver_meilleur_split_global(X, y)
    
    # En cas de situation impossible à découper correctement (ex: toutes les valeurs sont les mêmes)
    if feature_index is None or seuil is None:
        return Noeud(prediction=classe_majoritaire)
        
    # --- Le Découpage (Split) ---
    X_colonne = X[:, feature_index]
    
    # Masque pour diriger le trafic de données vers la GAUCHE ou vers la DROITE
    masque_gauche = X_colonne <= seuil
    X_gauche, y_gauche = X[masque_gauche], y[masque_gauche]
    
    masque_droite = X_colonne > seuil
    X_droite, y_droite = X[masque_droite], y[masque_droite]
    
    # --- Création du Nœud ---
    # Nous créons une question que nous plaçons dans l'arbre !
    noeud_actuel = Noeud(feature_index=feature_index, seuil=seuil)
    
    # --- Et maintenant, LA RECURSIVITE ! ---
    # Notre Nœud demande à la fonction de construire automatiquement les blocs suivants 
    # pour ses données de gauche et de droite. (La profondeur augmente de 1)
    noeud_actuel.gauche = construire_arbre(X_gauche, y_gauche, profondeur + 1, max_profondeur)
    noeud_actuel.droite = construire_arbre(X_droite, y_droite, profondeur + 1, max_profondeur)
    
    return noeud_actuel

def predire_un(noeud, x):
    """
    Étape 3A.4 — Suit le parcours d'UNE seule image de maïs (x) dans notre labyrinthe qu'est l'Arbre !
    """
    # 1. Si on tombe sur une fin (une feuille de décision), on donne le oui/non direct !
    if noeud.prediction is not None:
        return noeud.prediction
        
    # 2. Sinon, il nous dit quelle "caractéristique" regarder et quelle valeur limite
    valeur_feature = x[noeud.feature_index]
    
    if valeur_feature <= noeud.seuil:
        # Allez à gauche camarade, et répétez la procédure ! (Récursivité simple)
        return predire_un(noeud.gauche, x)
    else:
        # Allez à droite camarade !
        return predire_un(noeud.droite, x)

def predire_tous(noeud_racine, X):
    """
    Étape 3A.5 — Teste tout un grand tableau d'images en les faisant passer
    une par une dans la moulinette `predire_un`.
    """
    X = np.array(X)
    predictions = []
    
    # On fait tomber toutes nos "lignes" (caractéristiques par images) dans l'arbre
    for ligne in X:
        p = predire_un(noeud_racine, ligne)
        predictions.append(p)
        
    return np.array(predictions)
