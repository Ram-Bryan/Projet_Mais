"""
Point d'entrée du Site Web (Vue Globale)
C'est le fichier qu'on lance nativement via la commande `streamlit run app.py` !
Il ne doit contenir QUE du design et du texte (les calculs sont ailleurs).
"""
import os
import sys
import streamlit as st

# Connexion aux autres fichiers du projet
chemin_base = os.path.join(os.path.dirname(__file__))
if chemin_base not in sys.path:
    sys.path.append(chemin_base)

# On importe les outils dont Streamlit a besoin pour communiquer avec l'IA
from src.controllers.prediction_controller import predire_image
from src.views.gallery_view import afficher_galerie

# --- CONFIGURATION DE LA PAGE ---
# Donnons un petit côté moderne à la page...
st.set_page_config(page_title="IA Agricole - Madagascar", page_icon="🌽", layout="wide")


# --- SECTION 1 : ZONE DE TRAVAIL (Haut de page) ---
st.title("🌽 IA Diagnostic : Rouille Polysora (Madagascar)")
st.write("Bienvenue sur le prototype officiel. Soumettez une photo de feuille de maïs du terrain et l'Intelligence Artificielle en extraira immédiatement le verdict de maladie.")

# Widget Streamlit classique pour charger une image
fichier_upload = st.file_uploader("Prenez une photo ou sélectionnez une image...", type=["png", "jpg", "jpeg"])

if fichier_upload is not None:
    # 1. Sauvegarde TEMPORAIRE
    # Gérer la mémoire vive (Streamlit garde le fichier en RAM), nous le voulons en vrai fichier
    os.makedirs("uploads/temp", exist_ok=True)
    chemin_tempo = os.path.join("uploads/temp", fichier_upload.name)
    with open(chemin_tempo, "wb") as f:
        f.write(fichier_upload.getbuffer())
        
    st.write("---")
    
    # Organisation web : Moitié gauche (image), Moitié droite (résultats)
    col_image, col_resultat = st.columns([1, 1])
    
    with col_image:
        st.image(chemin_tempo, caption="Photographie soumise à examen", use_column_width=True)
        
    with col_resultat:
        st.subheader("Rapport d'Analyse Algorithmique...")
        
        # Petit effet de style informatisé le temps du calcul...
        with st.spinner("Extraction de la rouille, de la texture et de la régosité de la surface en cours..."):
            try:
                # 2. L'APPEL AU CONTROLEUR ! C'est ce qui déclenche l'IA
                prediction, pct_rouille, rugosite, entropie = predire_image(chemin_tempo)
                
                # Le contrôleur l'ayant mis aux archives, on efface l'immondice temporaire
                if os.path.exists(chemin_tempo):
                    os.remove(chemin_tempo)
                
                # 3. VERDICT VISUEL
                if prediction == 1:
                    st.error("🚨 ALERTE : Feuille Malade ! Présence quasi certaine du champignon de la Rouille Polysora.")
                else:
                    st.success("✅ SAINE : Rien à signaler ! La feuille montre une texture homogène conforme à la norme agricole.")
                    
                # 4. PREUVES MATHÉMATIQUES (La "Data")
                st.write("**Valeurs perçues par l'ordinateur :**")
                st.metric("Concentration Pixels Rouille", f"{pct_rouille * 100:.2f} %")
                st.metric("Indice Irrégularité (Sobel)", f"{rugosite:.2f}")
                st.metric("Niveau Chaos Visuel (Shannon)", f"{entropie:.2f}")
                
            except Exception as e:
                st.error("⚠️ Il y a eu un problème technique de traitement avec cette image.")
                st.warning(f"Message interne: {e}")
                

st.write("---")
# --- SECTION 2 : MEMOIRE DU SERVEUR (Bas de page) ---
st.subheader("📚 Archives des anciennes détections agronomiques")
st.write("Retrouvez ici-bas toutes les images validées ou censurées par votre système dans le passé.")

# Un simple appel d'une ligne à notre composant de Vue dédié !
afficher_galerie()
