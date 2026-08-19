import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Mon Traducteur IA Léger", layout="wide")

st.title("🌍 Mon Traducteur IA Léger")
st.write("Traduction rapide optimisée pour le web gratuit !")

# Chargement d'un pipeline de traduction très léger avec mise en cache
@st.cache_resource
def load_translator():
    # Utilisation d'un modèle plus petit qui ne dépasse pas la limite de mémoire
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

with st.spinner("Chargement du modèle..."):
    translator = load_translator()

col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("1. Entrée du texte (en Anglais vers Français)")
    texte_a_traduire = st.text_area("Tapez votre texte en anglais ici :", height=150)
    bouton_traduire = st.button("Traduire 🚀", use_container_width=True)

with col_droite:
    st.subheader("Résultat Traduit")
    resultat_box = st.empty()
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            with st.spinner("Traduction en cours..."):
                traduction = translator(texte_a_traduire)[0]['translation_text']
                resultat_box.success(traduction)
        else:
            resultat_box.warning("Veuillez d'abord entrer du texte.")
