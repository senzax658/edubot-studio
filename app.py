import streamlit as st
import os
from groq import Groq

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="EduBot Studio - Assistant Intelligent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. STYLE CSS PERSONNALISÉ
# ==========================================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SÉCURITÉ ET API GROQ
# ==========================================
def initialiser_client_groq():
    """Récupère la clé API depuis Streamlit Secrets ou les variables d'environnement."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


client = initialiser_client_groq()

# ==========================================
# 4. BARRE LATÉRALE
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bot.png", width=80)
    st.title("Paramètres EduBot")
    st.markdown("---")

    st.markdown("### 🧠 Configuration du modèle")

    modele_selectionne = st.selectbox(
        "Modèle Groq",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b"
        ],
        index=0
    )

    temperature = st.slider(
        "Créativité (Température)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    st.markdown("---")
    st.markdown("### 🛠️ Gestion de la session")

    if st.button("Effacer la conversation", type="primary", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour ! Je suis ton tuteur EduBot. "
                    "Comment puis-je t'aider dans tes révisions aujourd'hui ?"
                )
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.info(
        "EduBot Studio est ton tuteur virtuel propulsé par "
        "l'intelligence artificielle de Groq pour t'aider à réviser "
        "et répondre à toutes tes questions."
    )

# ==========================================
# 5. INTERFACE DE CHAT
# ==========================================
st.title("📚 EduBot Studio")
st.caption("Ton assistant pédagogique personnel intelligent avec mémoire de session.")

if not client:
    st.error(
        "⚠️ **Clé API Groq introuvable !** "
        "Crée un fichier `.streamlit/secrets.toml` puis ajoute :\n\n"
        "```toml\n"
        "GROQ_API_KEY = \"gsk_...\"\n"
        "```"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Bonjour ! Je suis ton tuteur EduBot. "
                "Comment puis-je t'aider dans tes révisions aujourd'hui ?"
            )
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 6. ENTRÉE UTILISATEUR ET REQUÊTE API
# ==========================================
if prompt := st.chat_input("Écris ta question ici..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Réflexion en cours...*")

        try:
            messages_api = [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.messages
            ]

            response = client.chat.completions.create(
                model=modele_selectionne,
                messages=messages_api,
                temperature=temperature
            )

            reply = response.choices[0].message.content

            message_placeholder.markdown(reply)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )

        except Exception as e:
            message_placeholder.error(
                f"❌ Une erreur technique est survenue : {e}"
            )

# ==========================================
# 7. PIED DE PAGE
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>"
    "EduBot Studio • Propulsé par Streamlit & Groq Cloud"
    "</div>",
    unsafe_allow_html=True
)
