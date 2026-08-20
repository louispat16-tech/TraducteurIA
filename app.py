import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration de la page
st.set_page_config(page_title="Mon Traducteur IA Personnel", layout="wide")

st.title("🌍 Mon Traducteur IA Personnel")
st.write("Choisissez vos langues, tapez votre texte et traduisez instantanément.")

# Dictionnaire des paires de langues ultra-légères et rapides
# Format : (Langue Source, Langue Cible) -> Modèle Hugging Face
models_map = {
    ("Français", "Anglais"): "Helsinki-NLP/opus-mt-fr-en",
    ("Anglais", "Français"): "Helsinki-NLP/opus-mt-en-fr",
    ("Français", "Espagnol"): "Helsinki-NLP/opus-mt-fr-es",
    ("Espagnol", "Français"): "Helsinki-NLP/opus-mt-es-fr",
    ("Anglais", "Espagnol"): "Helsinki-NLP/opus-mt-en-es",
    ("Espagnol", "Anglais"): "Helsinki-NLP/opus-mt-es-en",
    ("Anglais", "Allemand"): "Helsinki-NLP/opus-mt-en-de",
    ("Allemand", "Anglais"): "Helsinki-NLP/opus-mt-de-en",
    ("Anglais", "Italien"): "Helsinki-NLP/opus-mt-en-it",
    ("Italien", "Anglais"): "Helsinki-NLP/opus-mt-it-en",
    ("Français", "Allemand"): "Helsinki-NLP/opus-mt-fr-de",
    ("Allemand", "Français"): "Helsinki-NLP/opus-mt-de-fr",
    ("Français", "Italien"): "Helsinki-NLP/opus-mt-fr-it",
    ("Italien", "Français"): "Helsinki-NLP/opus-mt-it-fr"
}

# Chargement intelligent avec cache pour la vitesse
@st.cache_resource
def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Création des deux colonnes
col_gauche, col_droite = st.columns(2)

with col_gauche:
    st.subheader("1. Langues et Texte")
    
    langue_source = st.selectbox("Langue d'origine :", ["Français", "Anglais", "Espagnol", "Allemand", "Italien"])
    langue_cible = st.selectbox("Langue de destination :", ["Français", "Anglais", "Espagnol", "Allemand", "Italien"])
    
    st.markdown("🎙️ **Entrée vocale :**")
    audio_data = st.audio_input("Enregistrer votre voix")
    if audio_data:
        st.info("Audio capturé. (Tapez votre texte ci-dessous pour lancer la traduction).")

    texte_a_traduire = st.text_area("Tapez votre texte ici :", height=130, placeholder="Écrivez votre texte...")
    
    bouton_traduire = st.button("Traduire 🚀", use_container_width=True)

with col_droite:
    st.subheader("Résultat Traduit")
    
    if bouton_traduire:
        if texte_a_traduire.strip() != "":
            if langue_source == langue_cible:
                st.warning("Veuillez choisir deux langues différentes !")
            else:
                paire = (langue_source, langue_cible)
                if paire in models_map:
                    model_name = models_map[paire]
                    with st.spinner(f"Traduction en cours..."):
                        try:
                            tokenizer, model = get_model(model_name)
                            inputs = tokenizer(texte_a_traduire, return_tensors="pt", padding=True)
                            translated_tokens = model.generate(**inputs)
                            traduction = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
                            
                            st.success(traduction)
                        except Exception as e:
                            st.error(f"Erreur technique : {e}")
                else:
                    st.warning(f"Désolé, la combinaison {langue_source} -> {langue_cible} n'est pas encore activée. Essayez Français ⇄ Anglais ou Espagnol.")
        else:
            st.warning("Veuillez d'abord entrer du texte à gauche.")
st.success(traduction)

# --- Synthèse vocale ---
from gtts import gTTS
import io

# Associer la langue cible à son code pour la voix
lang_codes = {
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Italien": "it"
}

if langue_cible in lang_codes:
    tts_lang = lang_codes[langue_cible]
    # Générer l'audio à partir du texte traduit
    tts = gTTS(text=traduction, lang=tts_lang, slow=False)
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    
    # Afficher le lecteur audio sur l'interface
    st.audio(audio_io, format='audio/mp3')
