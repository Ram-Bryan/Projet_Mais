"""
Couche View : Interface du terminal.
La responsabilité de ce fichier est uniquement de faire des "print" chics, jolis 
et lisibles par l'utilisateur (le chercheur ou l'agriculteur). 
Aucune modification de la donnée source ne doit avoir lieu ici !
"""
import pandas as pd

def afficher_tableau_comparatif(resultats):
    """
    Étape 3C.3 — Imprime un tableau récapitulatif des performances des 4 modèles concurents.
    `resultats` est un dictionnaire envoyé par le contrôleur.
    """
    print("\n" + "="*70)
    print("🏆 TABLEAU DE COMPARAISON DES PERFORMANCES MACHINES 🏆")
    print("="*70)
    
    # Transformation des statistiques brutes en vrais pourcentages jolis à l'œil
    lignes_tableau = []
    for nom_modele, metriques in resultats.items():
        # ex: 0.8550 -> "85.50%"
        acc = f"{metriques['accuracy'] * 100:.2f}%"
        prec = f"{metriques['precision'] * 100:.2f}%"
        rap = f"{metriques['rappel'] * 100:.2f}%"
        
        lignes_tableau.append({
            "Modèle": nom_modele,
            "Accuracy": acc,
            "Précision": prec,
            "Rappel": rap
        })
        
    df_affichage = pd.DataFrame(lignes_tableau)
    # L'outil to_string de pandas permet d'imprimer proprement en alignant les colonnes (index=False cache la colonne 0,1,2,3)
    print("\n" + df_affichage.to_string(index=False) + "\n")
    
    print("-" * 70)
    print("🧩 INTERPRÉTATION TERRAIN (Validation via la Matrice de Confusion)")
    print("-" * 70)
    
    # Pour un agriculteur Malgache, le coût d'une erreur varie :
    # Faux Malade (Faux Positif) => l'agriculteur paie des produits toxiques pour rien (Gaspillage)
    # Faux Sain (Faux Négatif) => L'épidémie détruit le champ entier (CATASTROPHE !)
    for nom_modele, metriques in resultats.items():
        cm = metriques['confusion']
        print(f"\n👉 {nom_modele} :")
        
        # S'il y a assez de catégories de test (au moins 1 sain et 1 malade), la matrice fera une taille normale (2x2)
        if cm.shape == (2,2):
            vrai_sain, faux_malade = cm[0][0], cm[0][1]
            faux_sain, vrai_malade = cm[1][0], cm[1][1]
            
            print(f"   VRAIS Sains (Succès) : {vrai_sain}   |  FAUX Malades (Gaspillage) : {faux_malade}")
            print(f"   FAUX Sains  (DANGER) : {faux_sain}   |  VRAIS Malades (Bien Géré)  : {vrai_malade}")
        else:
            print(f"   Matrice illisible pour la restrancription humaine : \n{cm}")

def afficher_message(message):
    """Petite fonction pour centraliser tous nos affichages de log avec un tag '[INFO]'."""
    print(f"\n[INFO] {message}")
