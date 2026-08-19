import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration de la page
st.set_page_config(page_title="Mon Traducteur IA Personnel", layout="wide")

st.title("🌍 Mon Traducteur IA Personnel")
st.write("Entrez votre texte, choisissez la langue cible et laissez l'IA traduire instantanément.")

# Dictionnaire des modèles ultra-performants et légers
model_map = {
    "Français": "Helsinki-NLP/opus-mt-en-fr",
    "Espagnol": "Helsinki-NLP/opus-mt-en-es",
    "Allemand": "Helsinki-NLP/opus-mt-en-de",
    "Italien": "Helsinki-NLP/opus-mt-en-it"
}

# Chargement du modèle avec cache pour aller vite
@st.cache_resource
def get_model_and_tokenizer(lang_name):
    model_name = model_map[lang_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Chargement des modèles..."):
    # Par défaut on pré-charge le français
    tokenizer, model = get_model_and_tokenizer("Français")

# Création des deux colonnes (entrées à gauche, résultats à droite)
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("1. Entrée du texte")
    texte_a_traduire = st.text_area("Tapez votre texte en anglais ici :", height=150)
    
    st.subheader("2. Langue cible")
    langue_cible = st.selectbox("Choisissez la langue :", list(model_map.keys()))
    
    bouton_traduire = st.button("Traduire 🚀", use_container_width=True)

with col_droite:
    st.subheader("Résultat Traduit")
    resultat_box = st.empty()
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            with st.spinner(f"Traduction en {langue_cible}..."):
                # Charge le modèle correspondant à la langue choisie
                tokenizer, model = get_model_and_tokenizer(langue_cible)
                inputs = tokenizer(texte_a_traduire, return_tensors="pt", padding=True)
                translated_tokens = model.generate(**inputs)
                traduction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                
                resultat_box.success(traduction)
        else:
            resultat_box.warning("Veuillez d'abord entrer du texte à gauche.")
