import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration de la page
st.set_page_config(page_title="Mon Traducteur IA Personnel", layout="wide")

st.title("🌍 Mon Traducteur IA Personnel")
st.write("Entrez votre texte, choisissez la langue cible et laissez l'IA traduire instantanément.")

# Dictionnaire des modèles (Anglais vers d'autres langues)
model_map = {
    "Français": "Helsinki-NLP/opus-mt-en-fr",
    "Espagnol": "Helsinki-NLP/opus-mt-en-es",
    "Allemand": "Helsinki-NLP/opus-mt-en-de",
    "Italien": "Helsinki-NLP/opus-mt-en-it"
}

# Chargement intelligent avec cache global pour éviter les lenteurs
@st.cache_resource
def get_model_and_tokenizer(lang_name):
    model_name = model_map[lang_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Création des deux colonnes (entrées à gauche, résultats à droite)
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("1. Entrée du texte")
    texte_a_traduire = st.text_area("Tapez votre texte ici (en anglais pour l'instant) :", height=150, placeholder="Écrivez votre texte ici...")
    
    st.subheader("2. Langue cible")
    langue_cible = st.selectbox("Choisissez la langue de destination :", list(model_map.keys()))
    
    bouton_traduire = st.button("Traduire 🚀", use_container_width=True)

with col_droite:
    st.subheader("Résultat Traduit")
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            with st.spinner(f"Traduction en {langue_cible} en cours..."):
                try:
                    # Récupère le modèle correspondant
                    tokenizer, model = get_model_and_tokenizer(langue_cible)
                    
                    # Traitement et traduction par l'IA
                    inputs = tokenizer(texte_a_traduire, return_tensors="pt", padding=True)
                    translated_tokens = model.generate(**inputs)
                    traduction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                    
                    st.success(traduction)
                except Exception as e:
                    st.error(f"Erreur lors de la traduction : {e}")
        else:
            st.warning("Veuillez d'abord entrer du texte à gauche.")
