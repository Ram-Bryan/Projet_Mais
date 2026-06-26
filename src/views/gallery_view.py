"""
Couche View : Composant visuel Streamlit de la Galerie
Ce fichier contient la portion de code Streamlit responsable de dessiner la galerie, 
on utilise cela pour éviter que "app.py" ne devienne illisible.
"""
import os
import streamlit as st

def afficher_galerie():
    """
    Étape 4.2 — Affiche la mémoire tampon des images étudiées sous forme de galerie (grille).
    """
    dossier_uploads = "uploads"
    
    # Sécurité: le dossier n'existe peut-être pas au tout premier lancement
    if not os.path.exists(dossier_uploads):
        st.info("Aucune image n'a encore été analysée ou enregistrée.")
        return
        
    # Liste de tout ce qui a été téléchargé
    fichiers = [f for f in os.listdir(dossier_uploads) if f.startswith("MALADE_") or f.startswith("SAINE_")]
    
    if len(fichiers) == 0:
        st.info("La galerie de détection est vide.")
        return
        
    # On crée 4 colonnes virtuelles (système de grilles type Bootstrap/Pinterest)
    colonnes = st.columns(4)
    
    # On parcourt chaque photo, la modulo (%) permet de passer de la col 1 à 4 en boucle infinie
    for i, nom_fichier in enumerate(fichiers):
        chemin_complet = os.path.join(dossier_uploads, nom_fichier)
        colonne_actuelle = colonnes[i % 4]
        
        # On dit à Streamlit de dessiner dans cette colonne spécifique
        with colonne_actuelle:
            
            # Affichage de l'image (miniature automatique)
            st.image(chemin_complet, use_column_width=True)
            
            # Utilisation du petit hack mis en place par le controller (Le mot dans le nom du fichier)
            if nom_fichier.startswith("MALADE_"):
                st.error("Diagnostiquée : Malade")
            elif nom_fichier.startswith("SAINE_"):
                st.success("Diagnostiquée : Saine")
