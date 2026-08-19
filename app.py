import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Traducteur Expert", layout="wide")
st.title("🌍 Traducteur Expert (Fr, Es, De, It)")

# On définit les modèles spécialisés (Opus-MT sont les meilleurs pour la traduction pure)
model_map = {
    "Français": "Helsinki-NLP/opus-mt-en-fr",
    "Espagnol": "Helsinki-NLP/opus-mt-en-es",
    "Allemand": "Helsinki-NLP/opus-mt-en-de",
    "Italien": "Helsinki-NLP/opus-mt-en-it"
}

# Fonction pour charger le traducteur selon la langue choisie
@st.cache_resource
def get_translator(lang_code):
    return pipeline("translation", model=model_map[lang_code])

col_gauche, col_droite = st.columns(2)

with col_gauche:
    texte = st.text_area("Entrez votre texte en anglais :", height=150)
    langue = st.selectbox("Langue cible :", list(model_map.keys()))
    bouton = st.button("Traduire 🚀")

with col_droite:
    if bouton and texte:
        with st.spinner(f"Traduction en {langue}..."):
            translator = get_translator(langue)
            resultat = translator(texte)[0]['translation_text']
            st.success(resultat)
    elif bouton:
        st.warning("Veuillez entrer du texte.")
