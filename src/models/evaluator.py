"""
Couche Model : Évaluation et Mathématiques !
Aucun affichage ici. Uniquement du calcul de justesse d'algorithmes.
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def diviser_donnees(df):
    """
    Étape 3C.1 — Sépare le dataset dans une proportion de 80%/20%.
    80% des données serviront d'école (entraînement).
    20% seront gardées secrètes pour tester la "vraie" intelligence de l'algorithme après coup.
    """
    # Si le tableau est trop petit on ne peut rien faire...
    if len(df) == 0:
        return None, None, None, None
        
    # On récupère seulement les valeurs chiffrées des 3 dernières colonnes analytiques
    X = df[['pct_rouille', 'rugosite', 'entropie_texture']].values
    
    # On récupère le corrigé du professeur (0 = saine, 1 = malade)
    y = df['label_malade'].values
    
    # L'outil train_test_split (qui vient de scikit-learn) fait ce travail mieux que personne !
    # random_state=42 permet d'avoir le même "hasard" à chaque exécution du code sur votre machine.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def calculer_metriques(y_reel, y_predit):
    """
    Étape 3C.2 — Calcule les scores de réussite.
    
    - L'Accuracy (% score brut) a ses limites : Si on a 99 feuilles saines et 1 malade, 
      un algo qui dit "Toujours Sain" a 99% d'accuracy! Pourtant il est inutile...
    - D'où la Précision et le Rappel, cruciaux pour un contexte médical ou agronomique.
    """
    if len(y_reel) == 0:
        # Sécurité si les données sont vides
        return {"accuracy": 0.0, "precision": 0.0, "rappel": 0.0, "confusion": [[0,0],[0,0]]}
        
    acc = accuracy_score(y_reel, y_predit)
    
    # zero_division=0 permet d'éviter les crashs dans le terminal si l'ordinateur rate totalement une catégorie
    prec = precision_score(y_reel, y_predit, zero_division=0)
    rap = recall_score(y_reel, y_predit, zero_division=0)
    
    conf = confusion_matrix(y_reel, y_predit, labels=[0, 1])
    
    return {
        "accuracy": acc,
        "precision": prec,
        "rappel": rap,
        "confusion": conf
    }
