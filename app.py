import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration de la page
st.set_page_config(page_title="Mon Traducteur IA Personnel", layout="wide")

st.title("🌍 Mon Traducteur IA Personnel")
st.write("Sélectionnez vos langues, entrez votre texte ou utilisez le micro, et laissez l'IA traduire.")

# Dictionnaire des langues supportées pour un modèle universel rapide
# Utilisation de m2m100 (multilingue universel)
model_name = "facebook/m2m100_418M"

@st.cache_resource
def get_translator_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Chargement de l'IA multilingue..."):
    tokenizer, model = get_translator_model()

# Mapping des langues vers les codes ISO de Facebook M2M100
lang_codes = {
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Italien": "it"
}

# Création des deux colonnes
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("1. Source et Texte")
    
    # Choix de la langue source
    langue_source = st.selectbox("Langue d'origine :", list(lang_codes.keys()))
    
    # Entrée vocale propre
    st.markdown("🎙️ **Entrée vocale :**")
    audio_data = st.audio_input("Enregistrer votre voix")
    if audio_data:
        st.info("Audio capturé. (La transcription automatique nécessite un module Whisper additionnel, écrivez votre texte ci-dessous en attendant).")

    texte_a_traduire = st.text_area("Ou tapez votre texte ici :", height=130, placeholder="Écrivez votre texte à traduire...")
    
    st.subheader("2. Langue cible")
    langue_cible = st.selectbox("Choisissez la langue de destination :", list(lang_codes.keys()))
    
    bouton_traduire = st.button("Traduire 🚀", use_container_width=True)

with col_droite:
    st.subheader("Résultat Traduit")
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            if langue_source == langue_cible:
                st.warning("La langue source et la langue cible doivent être différentes !")
            else:
                with st.spinner(f"Traduction du {langue_source} vers le {langue_cible}..."):
                    try:
                        # Configuration de la langue source pour le modèle M2M100
                        tokenizer.src_lang = lang_codes[langue_source]
                        encoded = tokenizer(texte_a_traduire, return_tensors="pt")
                        
                        # Génération de la traduction vers la langue cible
                        generated_tokens = model.generate(
                            **encoded, 
                            forced_bos_token_id=tokenizer.get_lang_id(lang_codes[langue_cible])
                        )
                        traduction = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                        
                        st.success(traduction)
                    except Exception as e:
                        st.error(f"Erreur lors de la traduction : {e}")
        else:
            st.warning("Veuillez d'abord entrer du texte à gauche.")
