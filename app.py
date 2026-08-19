import streamlit as st
from transformers import pipeline

# Configuration de la page en mode large
st.set_page_config(page_title="Traducteur Universel Vocal & Texte", layout="wide")

st.title("🌍 Traducteur Universel Texte")
st.write("Écrivez un texte, choisissez la langue cible et obtenez la traduction instantanément !")

# Dictionnaire des modèles ultra-performants et légers
model_map = {
    "Français": "Helsinki-NLP/opus-mt-en-fr",
    "Espagnol": "Helsinki-NLP/opus-mt-en-es",
    "Allemand": "Helsinki-NLP/opus-mt-en-de",
    "Italien": "Helsinki-NLP/opus-mt-en-it"
}

@st.cache_resource
def get_translator(lang_name):
    return pipeline("translation", model=model_map[lang_name])

# Création des deux colonnes (comme sur Gradio)
col_gauche, col_droite = st.columns(2)

# --- COLONNE DE GAUCHE : LES ENTRÉES ---
with col_gauche:
    st.subheader("1. Ou écrivez ici")
    texte_a_traduire = st.text_area("Ou tapez votre texte ici si vous n'avez pas parlé...", height=150)
    
    st.subheader("2. Langue cible")
    langue_cible = st.selectbox("Choisissez la langue :", list(model_map.keys()))
    
    bouton_traduire = st.button("Traduire et Parler 🚀", use_container_width=True)

# --- COLONNE DE DROITE : LES RÉSULTATS ---
with col_droite:
    st.subheader("Résultat Traduit (Texte)")
    
    # Zone de résultat dans la colonne de droite
    resultat_box = st.empty()
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            with st.spinner(f"Traduction en {langue_cible} par l'IA..."):
                translator = get_translator(langue_cible)
                traduction = translator(texte_a_traduire)[0]['translation_text']
                resultat_box.success(traduction)
        else:
            resultat_box.warning("Veuillez d'abord entrer du texte à gauche.")
