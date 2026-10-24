import streamlit as st
import os
from groq import Groq
import urllib.parse

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="EduBot Studio - Espace d'Étude",
    page_icon="📚",
    layout="wide"
)

# --- 2. CONNEXION À GROQ API ---
api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ Clé API Groq manquante ! Veuillez configurer la clé dans les secrets Streamlit Cloud.")
    st.stop()

client = Groq(api_key=api_key)

# --- 3. STYLE CSS STUDIEUX & ACADÉMIQUE ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E131F;
        color: #E2E8F0;
    }
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #1E293B;
        color: #CBD5E1;
        border: 1px solid #334155;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #334155;
        color: #F8FAFC;
        border-color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. INITIALISATION DES SESSIONS ET HISTORIQUE ---
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "Nouvelle séance d'étude"
    st.session_state.chats["Nouvelle séance d'étude"] = []

# --- 5. BARRE LATÉRALE (SIDEBAR STUDIO) ---
with st.sidebar:
    st.title("📚 EduBot Studio")
    st.caption("Espace de Révisions & Travail Guidé")
    st.divider()

    if st.button("📖 Nouvelle séance d'étude", use_container_width=True):
        count = len(st.session_state.chats) + 1
        new_title = f"Séance {count}"
        st.session_state.chats[new_title] = []
        st.session_state.current_chat_title = new_title
        st.rerun()

    st.write("### 📜 Vos fiches & conversations")
    
    titles = list(st.session_state.chats.keys())
    if titles:
        selected = st.radio(
            "Historique :",
            options=titles,
            index=titles.index(st.session_state.current_chat_title)
        )
        if selected != st.session_state.current_chat_title:
            st.session_state.current_chat_title = selected
            st.rerun()

    st.divider()
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.chats = {"Nouvelle séance d'étude": []}
        st.session_state.current_chat_title = "Nouvelle séance d'étude"
        st.rerun()

# --- 6. EN-TÊTE PRINCIPAL ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📚 EduBot — Studio d'Apprentissage</div>
    <div class="hero-subtitle">
        Bienvenue dans votre espace d'étude académique. Demandez des fiches de synthèse, des explications de cours, 
        ou des schémas scientifiques précis (ex: <i>« Fais un schéma des vecteurs »</i>).
    </div>
</div>
""", unsafe_allow_html=True)

messages = st.session_state.chats[st.session_state.current_chat_title]

if not messages:
    st.info("💡 **Séance prête.** Posez votre question ci-dessous ou demandez un résumé de cours pour démarrer.")

# --- 7. AFFICHAGE DE LA DISCUSSION ---
for msg in messages:
    avatar = "👤" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "image_url" in msg:
            st.image(msg["image_url"], caption="Schéma pédagogique généré par EduBot", use_column_width=True)

# --- 8. GÉNÉRATEUR DE SCHÉMAS ÉDUCATIFS ---
def generate_educational_schema(prompt_text):
    clean_prompt = f"academic whiteboard diagram, clean educational vector graphic, scientific math physics illustration, high resolution chart, {prompt_text}"
    encoded_prompt = urllib.parse.quote(clean_prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=500&nologo=true"

# --- 9. TRAITEMENT DE LA SAISIE ---
if prompt := st.chat_input("Posez votre question de cours ou demandez un schéma..."):
    
    if len(messages) == 0:
        clean_title = prompt.strip()[:28] + ("..." if len(prompt) > 28 else "")
        st.session_state.chats[clean_title] = st.session_state.chats.pop(st.session_state.current_chat_title)
        st.session_state.current_chat_title = clean_title
        messages = st.session_state.chats[clean_title]

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    image_keywords = ["dessine", "dessin", "schéma", "schema", "illustration", "fais une image", "montre un schéma", "graphique"]
    wants_image = any(word in prompt.lower() for word in image_keywords)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("EduBot prépare l'explication et la synthèse..."):
            
            # Préparation de l'historique pour l'API
            groq_messages = [
                {"role": "system", "content": "Tu es EduBot, un tuteur pédagogique d'élite pour lycéens. Tes réponses sont rigoureuses, structurées, claires, et rédigées dans un ton académique encourageant."}
            ] + [{"role": m["role"], "content": m["content"]} for m in messages]

            # Appel API Groq avec modèle stable
         chat_completion = client.chat.completions.create(
                model="gemma2-9b-it",
                messages=groq_messages
            )
            
            reply_text = chat_completion.choices[0].message.content
            st.markdown(reply_text)
            
            msg_data = {"role": "assistant", "content": reply_text}

            if wants_image:
                st.caption("🎨 *Génération du schéma pédagogique en cours...*")
                img_url = generate_educational_schema(prompt)
                st.image(img_url, caption="Schéma pédagogique généré par EduBot", use_column_width=True)
                msg_data["image_url"] = img_url

            messages.append(msg_data)
            st.session_state.chats[st.session_state.current_chat_title] = messages
