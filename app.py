import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import os

# Configuration de la page
st.set_page_config(page_title="Traducteur Universel Vocal & Texte", layout="wide")

st.title("🌍 Traducteur Universel Vocal & Texte")
st.write("Écrivez un texte, choisissez la langue cible et obtenez la traduction écrite et parlée !")

# Dictionnaire des modèles
model_map = {
    "Français": "Helsinki-NLP/opus-mt-en-fr",
    "Espagnol": "Helsinki-NLP/opus-mt-en-es",
    "Allemand": "Helsinki-NLP/opus-mt-en-de",
    "Italien": "Helsinki-NLP/opus-mt-en-it"
}

# Fonction sécurisée pour charger le modèle et le tokenizer
@st.cache_resource
def get_model_and_tokenizer(lang_name):
    model_name = model_map[lang_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Création des deux colonnes
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("1. Ou écrivez ici")
    texte_a_traduire = st.text_area("Tapez votre texte en anglais ici...", height=150)
    
    st.subheader("2. Langue cible")
    langue_cible = st.selectbox("Choisissez la langue :", list(model_map.keys()))
    
    bouton_traduire = st.button("Traduire et Parler 🚀", use_container_width=True)

with col_droite:
    st.subheader("Résultat Traduit (Texte)")
    resultat_box = st.empty()
    
    st.subheader("🔊 Résultat Traduit (Voix de l'IA)")
    audio_box = st.empty()
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            with st.spinner("Traduction et génération vocale en cours..."):
                # 1. Traduction
                tokenizer, model = get_model_and_tokenizer(langue_cible)
                inputs = tokenizer(texte_a_traduire, return_tensors="pt", padding=True)
                translated_tokens = model.generate(**inputs)
                traduction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                
                resultat_box.success(traduction)
                
                # 2. Génération de la voix (Audio)
                lang_code_map = {"Français": "fr", "Espagnol": "es", "Allemand": "de", "Italien": "it"}
                tts = gTTS(text=traduction, lang=lang_code_map[langue_cible], slow=False)
                audio_path = "traduction.mp3"
                tts.save(audio_path)
                
                # Lecture de l'audio sur Streamlit
                audio_box.audio(audio_path, format="audio/mp3")
        else:
            resultat_box.warning("Veuillez d'abord entrer du texte à gauche.")
