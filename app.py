import streamlit as st
import os
from groq import Groq

st.set_page_config(
    page_title="EduBot Studio",
    page_icon="📚",
    layout="wide"
)

# Récupération sécurisée de la clé depuis les secrets Streamlit
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ Clé API Groq introuvable dans les Secrets Streamlit.")
    st.stop()

client = Groq(api_key=api_key)

st.title("📚 EduBot Studio")

# Initialisation de l'historique pour garder la mémoire
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Envoi de la requête avec tout l'historique
if prompt := st.chat_input("Pose ta question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Erreur technique : {e}")
