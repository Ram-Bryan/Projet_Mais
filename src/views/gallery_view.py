"""
Couche View : Composant visuel Streamlit de la Galerie
Ce fichier contient la portion de code Streamlit responsable de dessiner la galerie, 
on utilise cela pour éviter que "app.py" ne devienne illisible.
"""
import os
import streamlit as st
from PIL import Image, ImageOps
from src.models.feature_extractor import extraire_features_image

def rendre_image_carree_pour_galerie(chemin_image, size=(300, 300)):
    """
    Rogne automatiquement les images par le centre pour qu'elles aient toutes les mêmes
    proportions (ex: 300x300, un carré naturel). Cela permet de créer des 'cartes uniformes' 
    (Cards) dans le layout ! Moche pour la botanique fine, mais parfait pour la galerie visuelle !
    """
    try:
        img = Image.open(chemin_image)
        # ImageOps.fit taille et crop si nécessaire pour forcer les dimensions.
        img_cropped = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        return img_cropped
    except Exception:
        # Si une erreur survient (format étrange, image cassée...), tant pis on l'affiche normale
        return chemin_image

def afficher_galerie():
    """
    Étape 4.2 — Affiche la mémoire tampon des images étudiées sous forme de galerie (Grille Symétrique 3 par 3).
    """
    dossier_uploads = "uploads"
    
    # Sécurité: le dossier n'existe peut-être pas au tout premier lancement
    if not os.path.exists(dossier_uploads):
        st.info("Aucune image n'a encore été analysée ou enregistrée.")
        return
        
    # Liste des images triées (Maintenant qu'on a un timestamp dans le nom, ça triera automatiquement par Nouveauté)
    fichiers = [f for f in os.listdir(dossier_uploads) if "-MALADE-" in f or "-SAINE-" in f]
    fichiers.sort(reverse=True) # Les plus récents en haut !
    
    if len(fichiers) == 0:
        st.info("La galerie de détection est vide.")
        return
        
    # --- Création du système de Grille Symétrique (Cards & 3 colonnes) ---
    # range(0, 10, 3) va incrémenter de 3 en 3 (0, 3, 6, 9) ce qui crée ligne par ligne
    for i in range(0, len(fichiers), 3):
        # On définit exactement une ligne de 3 colonnes espacées également
        colonnes = st.columns(3)
        
        # On extrait jusqu'à 3 images spécifiques pour cette itération de "ligne"
        pour_cette_ligne = fichiers[i:i+3]
        
        # Pour chaque image que l'on assigne à sa propre colonne (de 0 à 2 : col gauche, col centre, col droite)
        for index_colonne, nom_fichier in enumerate(pour_cette_ligne):
            chemin_complet = os.path.join(dossier_uploads, nom_fichier)
            
            with colonnes[index_colonne]:
                # On utilise "st.container(border=True)" pour encadrer chaque cellule (Card design)
                with st.container(border=True):
                    
                    # On affiche l'image après l'avoir obligée de manière cachée à être 100% propre !
                    # use_container_width force l'image modifiée à parfaitement épouser le conteneur du tableau bordé !
                    image_uniforme = rendre_image_carree_pour_galerie(chemin_complet)
                    st.image(image_uniforme, use_container_width=True)
                    
                    # Vérification dans la nouvelle norme nominale ("-MALADE-" au lieu de "MALADE_")
                    if "-MALADE-" in nom_fichier:
                        st.error("Diagnostic : Malade")
                    elif "-SAINE-" in nom_fichier:
                        st.success("Diagnostic : Saine")
                        
                    # --- Analyse détaillée conservée ---
                    with st.expander("📊 Voir l'analyse détaillée"):
                        donnees = extraire_features_image(chemin_complet, label=-1)
                        if donnees:
                            st.caption(f"Rouille (Pixels): {donnees['pct_rouille'] * 100:.2f}%")
                            st.caption(f"Rugosité (Sobel): {donnees['rugosite']:.2f}")
                            st.caption(f"Chaos (Shannon): {donnees['entropie_texture']:.2f}")
                        else:
                            st.caption("Erreur de calcul")
