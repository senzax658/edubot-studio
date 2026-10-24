import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="EduBot Studio",
    page_icon="📚",
    layout="wide"
)

# Remplace par ta nouvelle clé API
client = Groq(api_key="METS_TA_NOUVELLE_CLE_ICI")

st.title("📚 EduBot Studio")

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de tous les messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Gestion des nouveaux messages
if prompt := st.chat_input("Pose ta question..."):
    # Ajout du message utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # On envoie TOUT l'historique pour que le bot garde le contexte
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            
            # Ajout de la réponse à l'historique
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Erreur technique : {e}")
