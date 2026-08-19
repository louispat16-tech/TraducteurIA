import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

st.title("Mon Traducteur IA Personnel 🌍")

# Chargement du modèle (avec un cache pour aller plus vite)
@st.cache_resource
def load_models():
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

with st.spinner("Chargement des modèles..."):
    tokenizer, model = load_models()

# Interface utilisateur
texte_a_traduire = st.text_area("Entrez le texte à traduire :")

langue_cible = st.selectbox("Langue cible :", ["fr", "en", "es", "de", "it"])

if st.button("Traduire"):
    if texte_a_traduire:
        tokenizer.src_lang = "en" # ou détectée automatiquement
        encoded = tokenizer(texte_a_traduire, return_tensors="pt")
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(langue_cible))
        traduction = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        st.success("Traduction :")
        st.write(traduction)
    else:
        st.warning("Veuillez entrer du texte.")
