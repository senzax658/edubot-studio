import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="EduBot Studio",
    page_icon="📚",
    layout="wide"
)

# Connexion directe avec ta clé pour éviter les bugs de secrets
client = Groq(api_key="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")

st.title("📚 EduBot Studio")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose ta question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Erreur technique : {e}")
