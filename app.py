import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration de la page
st.set_page_config(page_title="Mon Traducteur IA Personnel", layout="wide")

st.title("🌍 Mon Traducteur IA Personnel")
st.write("Entrez votre texte (en anglais), choisissez la langue cible et laissez l'IA traduire instantanément.")

# Dictionnaire des modèles (Anglais vers d'autres langues)
model_map = {
    "Anglais (Source)": None,
    "Français": "Helsinki-NLP/opus-mt-en-fr",
    "Espagnol": "Helsinki-NLP/opus-mt-en-es",
    "Allemand": "Helsinki-NLP/opus-mt-en-de",
    "Italien": "Helsinki-NLP/opus-mt-en-it"
}

# Chargement intelligent avec cache
@st.cache_resource
def get_model_and_tokenizer(lang_name):
    model_name = model_map[lang_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Création des deux colonnes
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("1. Entrée du texte")
    
    # Option pour simuler ou ajouter un micro (Streamlit natif intègre st.audio_input pour la voix)
    st.write("🎙️ Entrée vocale (Optionnelle) :")
    audio_data = st.audio_input("Parlez ici pour enregistrer votre voix")
    if audio_data:
        st.info("Audio enregistré ! (La transcription vocale complète nécessite une bibliothèque de reconnaissance audio comme Whisper, tu peux taper le texte ci-dessous en attendant).")

    texte_a_traduire = st.text_area("Tapez votre texte en anglais ici :", height=150, placeholder="Type your text in English here...")
    
    st.subheader("2. Langue cible")
    langue_cible = st.selectbox("Choisissez la langue de destination :", [k for k in model_map.keys() if k != "Anglais (Source)"])
    
    bouton_traduire = st.button("Traduire 🚀", use_container_width=True)

with col_droite:
    st.subheader("Résultat Traduit")
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            with st.spinner(f"Traduction en {langue_cible} en cours..."):
                try:
                    tokenizer, model = get_model_and_tokenizer(langue_cible)
                    inputs = tokenizer(texte_a_traduire, return_tensors="pt", padding=True)
                    translated_tokens = model.generate(**inputs)
                    traduction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    
                    st.success(traduction)
                except Exception as e:
                    st.error(f"Erreur lors de la traduction : {e}")
        else:
            st.warning("Veuillez d'abord entrer du texte à gauche.")
