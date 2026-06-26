"""
Ce fichier fait partie de la couche 'Model' dans l'architecture MVC.
Sa responsabilité unique (SOC) est d'implémenter la logique mathématique
de la métrique de pureté 'Max-Minority' et la recherche du meilleur seuil (split),
afin de construire plus tard nos arbres de décision.
"""

import numpy as np # Numpy est indispensable pour faire des calculs rapides sur des tableaux

def calculer_purete(y):
    """
    Étape 2.1 — Calcule la pureté d'un nœud (un ensemble de données) selon la métrique Max-Minority.
    
    Explication :
    'y' contient les réponses réelles (ex: [0, 1, 0, 0] pour Saine, Malade, Saine, Saine).
    La pureté est simplement le pourcentage de la classe la plus représentée (la "majorité").
    Si on a 90% de saines, la pureté est 0.9. Si c'est 50/50, la pureté est de 0.5.
    L'algorithme cherchera toujours à avoir la pureté maximale (proche de 1.0).
    """
    y = np.array(y)

    # 1. Si on nous donne un tableau vide, on arrête tout et on dit que la pureté est de 0.
    if len(y) == 0:
        return 0.0
        
    # 2. Compter le nombre total d'éléments dans ce groupe
    N = len(y)
    
    # 3. Compter combien il y a de "0" (feuilles saines)
    # y == 0 renvoie True (1) ou False (0) pour chaque élément, puis np.sum additionne les True.
    n0 = np.sum(y == 0)
    
    # 4. Compter combien il y a de "1" (feuilles malades)
    n1 = np.sum(y == 1)
    
    # 5. Calculer le pourcentage de la classe 0 (Saine) par rapport au total
    proportion_0 = n0 / N
    
    # 6. Calculer le pourcentage de la classe 1 (Malade)
    proportion_1 = n1 / N
    
    # 7. La règle Max-Minority définit la pureté comme le "max" entre ces deux proportions.
    return max(proportion_0, proportion_1)


def calculer_purete_split(y_gauche, y_droite):
    """
    Étape 2.2 — Calcule la pureté globale après qu'on ait coupé (split) nos données en deux.
    
    Explication :
    Quand on coupe nos données en deux groupes (gauche et droite) avec un seuil, 
    on veut savoir si cette "coupe" est bonne globale.
    On calcule la pureté de chaque groupe, puis on fait une "moyenne pondérée" :
    le groupe le plus gros comptera plus fort dans la note finale.
    """

    y_gauche = np.array(y_gauche)
    y_droite = np.array(y_droite)

    # 1. On compte le nombre total d'individus dans l'ensemble (gauche + droite)
    N_total = len(y_gauche) + len(y_droite)
    
    # Si le total est vide (extrêmement rare en pratique, mais bon pour la sécurité)
    if N_total == 0:
        return 0.0
        
    # 2. Calcul de la pureté du sous-groupe gauche multiplié par son "poids" (sa taille / total)
    poids_gauche = len(y_gauche) / N_total
    purete_gauche = calculer_purete(y_gauche)
    score_gauche = poids_gauche * purete_gauche
    
    # 3. Calcul de la pureté du sous-groupe droit multiplié par son "poids"
    poids_droite = len(y_droite) / N_total
    purete_droite = calculer_purete(y_droite)
    score_droite = poids_droite * purete_droite
    
    # 4. La pureté globale (P_split) est l'addition de ces deux scores
    p_split = score_gauche + score_droite
    
    return float(p_split)


def trouver_meilleur_split(X_colonne, y):
    """
    Étape 2.3 — Trouve mathématiquement la "meilleure valeur numérique" (seuil) 
    pour couper nos données en deux d'après une "feature" (ex: la rugosité).
    
    Explication :
    On teste plein de valeurs possibles pour la rugosité en coupant (ceux en dessous vs ceux au-dessus).
    A chaque fois, on calcule la pureté. On garde le seuil qui nous donne la note (pureté) la plus haute !
    """
    # IMPORTANT : numpy array sont nécessaires
    X_colonne = np.array(X_colonne)
    y = np.array(y)
    
    # Étape A : Trier les données. 
    # argsort() nous donne un "plan d'assemblage" : comment ranger les éléments du plus petit au plus grand
    indices_tries = np.argsort(X_colonne)
    
    # On réorganise X_colonne et y dans le bon ordre
    X_trie = X_colonne[indices_tries]
    y_trie = y[indices_tries]
    
    # Étape B : Préparer de la mémoire pour enregistrer notre futur "meilleur résultat"
    meilleur_seuil = None
    meilleure_purete = 0.0
    
    # Étape C : Boucler sur les valeurs consécutives.
    # On cherche toutes les "valeurs possibles" pour couper. Les valeurs uniques de X.
    valeurs_uniques = np.unique(X_trie)
    
    # On parcourt les valeurs uniques 2 par 2 (l'élément i et l'élément i+1)
    for i in range(len(valeurs_uniques) - 1):
        
        valeur_courante = valeurs_uniques[i]
        valeur_suivante = valeurs_uniques[i+1]
        
        # Le candidat "seuil" est tout simplement le juste milieu entre les deux valeurs 
        # (ex: entre 10 et 12, on va couper à 11)
        seuil = (valeur_courante + valeur_suivante) / 2.0
        
        # Séparation !
        # Le "groupe de gauche" (True ou False pour chaque élément, puis on récupère les labels correspondants)
        masque_gauche = X_trie <= seuil
        y_gauche = y_trie[masque_gauche]
        
        # Le "groupe de droite"
        masque_droite = X_trie > seuil
        y_droite = y_trie[masque_droite]
        
        # Calculer l'efficacité de cette coupe via la formule de l'Étape 2.2
        p_split = calculer_purete_split(y_gauche, y_droite)
        
        # Si cette coupe est la meilleure trouvée depuis le début du parcours, on la garde !
        if p_split > meilleure_purete:
            meilleure_purete = p_split
            meilleur_seuil = seuil
            
    # Étape D : On a fini tous les essais. On retourne donc le grand gagnant (Le seuil, et sa note).
    return meilleur_seuil, meilleure_purete
