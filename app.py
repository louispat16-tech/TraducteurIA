import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Configuration de la page en mode large
st.set_page_config(page_title="Mon Traducteur IA Personnel", layout="wide")

st.title("🌍 Mon Traducteur IA Personnel")
st.write("Entrez votre texte, choisissez la langue cible et laissez l'IA traduire instantanément.")

# Chargement du modèle avec cache pour ne pas le recharger à chaque clic
@st.cache_resource
def load_models():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Chargement des modèles en cours, veuillez patienter..."):
    tokenizer, model = load_models()

# Création de deux colonnes pour une belle présentation
col_gauche, col_droite = st.columns(2)

# --- COLONNE DE GAUCHE : ENTRÉES ---
with col_gauche:
    st.subheader("1. Entrée du texte")
    texte_a_traduire = st.text_area("Tapez votre texte ici :", height=150)
    
    langue_cible = st.selectbox(
        "2. Langue cible (fr=Français, en=Anglais, es=Espagnol, de=Allemand, etc.)", 
        ["fr", "en", "es", "de", "it", "pt", "zh", "ja"]
    )
    
    bouton_traduire = st.button("Traduire 🚀", use_container_width=True)

# --- COLONNE DE DROITE : RÉSULTATS ---
with col_droite:
    st.subheader("Résultat Traduit")
    
    # Zone d'affichage dynamique pour le résultat
    resultat_box = st.empty()
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            with st.spinner("Traduction en cours..."):
                # Configuration de la langue source (détectée ou par défaut en anglais)
                tokenizer.src_lang = "en" 
                encoded = tokenizer(texte_a_traduire, return_tensors="pt")
                
                generated_tokens = model.generate(
                    **encoded, 
                    forced_bos_token_id=tokenizer.get_lang_id(langue_cible)
                )
                traduction = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Affichage de la traduction réussie
                resultat_box.success(traduction)
        else:
            resultat_box.warning("Veuillez d'abord entrer du texte à gauche.")
