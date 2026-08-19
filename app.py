import os
import gradio as gr
from gtts import gTTS
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import whisper

print("Chargement des modèles d'IA en cours... Veuillez patienter.")

# 1. Chargement de Whisper (Reconnaissance vocale)
whisper_model = whisper.load_model("base")

# 2. Chargement d'un modèle de traduction multilingue performant (NLLB ou M2M100)
# M2M100 gère des dizaines de langues bidirectionnelles
translator_model_name = "facebook/m2m100_418M"
from transformers import M2M100Tokenizer
tokenizer = M2M100Tokenizer.from_pretrained(translator_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_name)

print("Modèles chargés avec succès !")


def traduire_et_convertir(audio_path, texte_saisi, langue_cible):
  """Fonction principale appelée lorsqu'on clique sur le bouton"""
  texte_source = ""

  # Cas 1 : L'utilisateur a parlé (Audio présent)
  if audio_path is not None:
    # Transcription audio -> texte avec Whisper
    resultat_whisper = whisper_model.transcribe(audio_path)
    texte_source = resultat_whisper["text"]
    langue_detectee = resultat_whisper["language"]
    print(f"Voix détectée ({langue_detectee}) : {texte_source}")

  # Cas 2 : L'utilisateur a écrit du texte (Priorité au texte si rempli)
  elif texte_saisi and texte_saisi.strip() != "":
    texte_source = texte_saisi
    print(f"Texte saisi : {texte_source}")
  else:
    return (
        "Veuillez parler dans le micro ou écrire un texte à traduire.",
        None,
    )

  # 3. Traduction du texte vers la langue cible
  try:
    # Configuration de la langue cible pour M2M100 (ex: 'fr', 'en', 'es', 'de', etc.)
    tokenizer.src_lang = "en"  # Par défaut si non détecté, ou géré dynamiquement
    if audio_path is not None:
      tokenizer.src_lang = (
          langue_detectee  # Utilise la langue détectée par Whisper
      )

    encoded = tokenizer(texte_source, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(langue_cible),
    )
    texte_traduit = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]
  except Exception as e:
    texte_traduit = f"Erreur de traduction : {str(e)}"

  print(f"Traduction ({langue_cible}) : {texte_traduit}")

  # 4. Synthèse vocale (Texte vers Audio)
  # Pour l'instant, on utilise une voix d'IA propre et fluide via gTTS
  output_audio_path = "traduction_sortie.mp3"
  tts = gTTS(text=texte_traduit, lang=langue_cible, slow=False)
  tts.save(output_audio_path)

  return texte_traduit, output_audio_path


# 5. Création de l'interface graphique (Gradio)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
  gr.Markdown("# 🌍 Traducteur Universel Vocal & Texte")
  gr.Markdown(
      "Parlez dans le micro ou écrivez un texte, choisissez la langue cible et"
      " obtenez la traduction écrite et parlée !"
  )

  with gr.Row():
    with gr.Column():
      audio_input = gr.Audio(
          sources=["microphone"], type="filepath", label="1. Parlez ici (Optionnel)"
      )
      text_input = gr.Textbox(
          lines=2,
          placeholder=(
              "Ou tapez votre texte ici si vous n'avez pas parlé..."
          ),
          label="1. Ou écrivez ici",
      )
      langue_dropdown = gr.Dropdown(
          choices=["fr", "en", "es", "de", "it", "ja", "zh", "ar", "pt"],
          value="fr",
          label="2. Langue cible (fr=Français, en=Anglais, es=Espagnol, etc.)",
      )
      btn_traduire = gr.Button("Traduire et Parler 🚀", variant="primary")

    with gr.Column():
      text_output = gr.Textbox(label="Résultat Traduit (Texte)")
      audio_output = gr.Audio(label="Résultat Traduit (Voix de l'IA)")

  # Connexion du bouton à la fonction Python
  btn_traduire.click(
      fn=traduire_et_convertir,
      inputs=[audio_input, text_input, langue_dropdown],
      outputs=[text_output, audio_output],
  )

# Lancer l'application web locale
if __name__ == "__main__":
  demo.launch()