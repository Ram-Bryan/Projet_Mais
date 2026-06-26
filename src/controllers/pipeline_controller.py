"""
Couche Controller : Chef d'Orchestre !
Il organise intelligemment le dialogue entre notre extraction (Model), 
nos calculs mathématiques d'IA (Model) et ce qu'on voit sur l'écran (View).
"""
import os
import pickle # Pickle permet de "congeler" un objet en python pour le stocker sur votre disque dur
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --- IMPORTATION DES MODULES MAISON (Models & Views) ---
from src.models.feature_extractor import construire_dataframe
from src.models.decision_tree import construire_arbre, predire_tous as predire_arbre_perso
from src.models.random_forest import entrainer_foret, predire_foret as predire_foret_perso
from src.models.evaluator import diviser_donnees, calculer_metriques
from src.views.results_view import afficher_tableau_comparatif, afficher_message


def preparer_donnees_test(dossier_saines, dossier_malades):
    """
    Fonction bonus pour vous permettre de passer le test sans télécharger 100 images sur Google !
    - Vérifie si les répertoires ont des images, sinon invente un Dataframe simulé parfait.
    """
    # Ça lit vos dossiers
    df = construire_dataframe(dossier_saines, dossier_malades)
    
    # Si vous n'aviez pas mis de photos... On en simule mathématiquement !
    if df.empty or len(df) < 10:
        afficher_message("Pas assez de vraies images trouvées. Simulation de 100 'Virtual-Images' pour tester les Performances...")
        np.random.seed(42) # Rendant le hasard "prévisible" à chaque réexécution
        
        # Variables : [pct_rouille, rugosite, entropie_texture, classe_label]
        # Les feuilles malades auront une très forte rugosité (20-40), les saines auront une très faible (5-15)
        saines = np.column_stack((
            np.random.uniform(0.0, 0.2, 50),
            np.random.uniform(5.0, 15.0, 50),
            np.random.uniform(2.0, 4.0, 50),
            np.zeros(50)
        ))
        malades = np.column_stack((
            np.random.uniform(0.6, 0.9, 50),
            np.random.uniform(20.0, 40.0, 50),
            np.random.uniform(6.0, 8.0, 50),
            np.ones(50)
        ))
        
        df = pd.DataFrame(np.vstack((saines, malades)), columns=['pct_rouille', 'rugosite', 'entropie_texture', 'label_malade'])
        # Sauvegarde
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/features.csv', index=False)
        
    return df

def lancer_pipeline_complet():
    """
    Étape 3C.4 — Relie absolument tous les câbles du projet depuis la feature extraction jusqu'au Pickle.
    """
    afficher_message("Demarrage du Super Pipeline Controller !")
    
    # ÉTAPE A : Collecte
    dossier_saines = os.path.join('dataset', 'saines')
    dossier_malades = os.path.join('dataset', 'malades')
    df = preparer_donnees_test(dossier_saines, dossier_malades)
    afficher_message(f"Collecte terminée ! Total des feuilles étudiées : {len(df)}")
    
    # ÉTAPE B : Scission Entrainement / Professeur Testeur (80% / 20%)
    X_train, X_test, y_train, y_test = diviser_donnees(df)
    
    # ÉTAPE C : Phase de l'Apprentissage (Fit / Construire)
    afficher_message("Lancement des cerveaux informatiques... Les modèles s'entraînent !")
    
    modele_arbre_perso = construire_arbre(X_train, y_train, max_profondeur=5)
    modele_foret_perso = entrainer_foret(X_train, y_train, n_arbres=10, max_profondeur=5)
    
    modele_arbre_sk = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    modele_arbre_sk.fit(X_train, y_train)
    
    modele_foret_sk = RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=5, random_state=42)
    modele_foret_sk.fit(X_train, y_train)
    
    # ÉTAPE D : Le Grand Examen Vaudou (L'ordinateur n'a jamais vu X_test)
    pred_arbre_perso = predire_arbre_perso(modele_arbre_perso, X_test)
    pred_foret_perso = predire_foret_perso(modele_foret_perso, X_test)
    pred_arbre_sk    = modele_arbre_sk.predict(X_test)
    pred_foret_sk    = modele_foret_sk.predict(X_test)
    
    # ÉTAPE E : Notation du Professeur (Évaluation des copies)
    resultats = {
        "Arbre From Scratch (MaxMinority)": calculer_metriques(y_test, pred_arbre_perso),
        "Forêt From Scratch (MaxMinority)": calculer_metriques(y_test, pred_foret_perso),
        "Arbre Scikit-Learn (Gini)":        calculer_metriques(y_test, pred_arbre_sk),
        "Forêt Scikit-Learn (Gini)":        calculer_metriques(y_test, pred_foret_sk)
    }
    
    # ÉTAPE F : Affichage à l'Écran (Le seul droit pour le visuel a la Vue locale)
    afficher_tableau_comparatif(resultats)
    
    # ÉTAPE G : Sauvegarde Physique (Export)
    # L'App Web de la Partie 4 a besoin d'avoir accès au champion.
    # Nous gelons notre "Forêt Scikit-Learn", la plus robuste pour de futurs prédictions d'agriculteurs.
    os.makedirs('saved_models', exist_ok=True)
    chemin_export = os.path.join('saved_models', 'best_model.pkl')
    
    with open(chemin_export, 'wb') as fichier:
        pickle.dump(modele_foret_sk, fichier)
        
    afficher_message(f"Félicitation ! Le Modèle 'Forêt Scikit-Learn' a été archivé en sécurité sous {chemin_export}")
