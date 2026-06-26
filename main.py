import os
import sys
import numpy as np

# On s'assure que python peut trouver nos fichiers dans le dossier src
chemin_base = os.path.join(os.path.dirname(__file__))
sys.path.append(chemin_base)

from src.controllers.pipeline_controller import lancer_pipeline_complet

# Importations
from src.models.feature_extractor import construire_dataframe
from src.models.max_minority import calculer_purete, trouver_meilleur_split
from src.models.decision_tree import construire_arbre, predire_tous
from src.models.random_forest import entrainer_foret, predire_foret

def tester_extracteur():
    print("--- Démarrage du test d'extraction de features ---")
    dossier_saines = os.path.join('dataset', 'saines')
    dossier_malades = os.path.join('dataset', 'malades')
    print(f"Recherche d'images dans : {dossier_saines} et {dossier_malades}")
    try:
        df = construire_dataframe(dossier_saines, dossier_malades)
        if df.empty:
            print("\n❌ Oups, le tableau est vide ! Aucun fichier image n'a été trouvé.")
        else:
            print("\n✅ Succès ! L'extraction a marché. Voici les premières lignes :")
            print(df.head())
    except Exception as e:
        print(f"\n❌ Erreur : {e}")

def tester_max_minority():
    print("--- 🧠 Démarrage du test mathématique Max-Minority ---")
    print("\n[Test 1] Calcul de Pureté d'un groupe")
    groupe_feuilles = np.array([0, 0, 0, 1])
    purete = calculer_purete(groupe_feuilles)
    print(f"-> Groupe : {groupe_feuilles}")
    print(f"-> Pureté trouvée par l'algorithme : {purete} (Attendu: 0.75)")

    print("\n[Test 2] Recherche du meilleur découpage (Split)")
    rugosites = np.array([10.5, 12.0, 15.0, 20.0, 22.0, 25.0])
    labels    = np.array([0,    0,    0,    1,    1,    1])
    meilleur_seuil, purete_obtenue = trouver_meilleur_split(rugosites, labels)
    print(f"-> 🎯 Meilleur seuil trouvé par l'ordinateur : {meilleur_seuil} (Attendu: 17.5)")
    print(f"-> 📈 Pureté globale après coupe : {purete_obtenue} (Attendu: 1.0 car coupe parfaite !)")

def tester_arbres():
    print("--- 🌲 Démarrage du test d'Intelligence Artificielle (3A & 3B) ---")
    print("\nCréation de 5 exemples 'Sains' (0) et 5 exemples 'Malades' (1)...")
    X = np.array([
        [0.10, 10.5], [0.12, 11.0], [0.05,  8.0], [0.15, 12.5], [0.08,  9.5], 
        [0.85, 25.0], [0.90, 28.0], [0.75, 22.0], [0.88, 26.5], [0.95, 30.0]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    X_mystere = np.array([[0.83, 24.0]])
    print("Feuille mystère à deviner : Beaucoup de rouille (0.83) et très rugueuse (24.0).")
    
    print("\n[PARTIE 3A] L'Arbre de Décision est en train d'apprendre...\n")
    mon_arbre = construire_arbre(X, y, max_profondeur=3)
    prediction_arbre = predire_tous(mon_arbre, X_mystere)
    print(f"-> 🤖 L'Arbre unique a voté : {prediction_arbre[0]} (0 = Saine, 1 = Malade)")
    if prediction_arbre[0] == 1:
        print("   ✅ Bonne réponse !")
        
    print("\n[PARTIE 3B] La Forêt de 10 arbres est en train d'apprendre par Bagging...\n")
    ma_foret = entrainer_foret(X, y, n_arbres=10, max_profondeur=3)
    prediction_foret = predire_foret(ma_foret, X_mystere)
    print(f"-> 🤖 La Forêt (loi de la majorité) a voté : {prediction_foret[0]}")
    if prediction_foret[0] == 1:
        print("   ✅ Parfaite déduction de la Forêt !")
        
    print("\nSi les deux algorithmes ont répondu 1, notre IA artisanale fonctionne à merveille ! 🚀")
    
def tester_le_gros_pipeline():
    """
    Lance le test ultime et majestueux de la Partie 3C !
    On laisse le Controller orchestrer l'intégralité du projet de A à Z.
    """
    print("\n")
    print("*"*60)
    print(" Lancement de l'Analyse Globale des Données ".center(60, '*'))
    print("*"*60)
    
    lancer_pipeline_complet()

if __name__ == "__main__":
    tester_le_gros_pipeline()
