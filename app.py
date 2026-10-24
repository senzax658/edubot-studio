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
# 2. CHARGEMENT DU STYLE CSS PERSONNALISÉ
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
# 3. GESTION DE LA SÉCURITÉ ET DE L'API GROQ
# ==========================================
def initialiser_client_groq():
    """Récupère la clé API de manière sécurisée (Secrets Streamlit ou Environnement)"""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

client = Groq(api_key=st.secrets.get("GROQ_API_KEY")) if "GROQ_API_KEY" in st.secrets else None

# ==========================================
# 4. BARRE LATÉRALE (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bot.png", width=80)
    st.title("Paramètres EduBot")
    st.markdown("---")
    
    st.markdown("### 🧠 Configuration du Modèle")
    # Choix du modèle actif
    modele_selectionne = st.selectbox(
        "Modèle Groq",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )
    
    # Contrôle de la créativité (Température)
    temperature = st.slider(
        "Créativité (Température)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ Gestion de la session")
    
    # Bouton pour effacer l'historique
    if st.button("Effacer la conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.info(
        "EduBot Studio est ton tuteur virtuel propulsé par "
        "l'intelligence artificielle de Groq pour t'aider à réviser "
        "et répondre à toutes tes questions."
    )

# ==========================================
# 5. CORPS DE L'APPLICATION (INTERFACE DE CHAT)
# ==========================================
st.title("📚 EduBot Studio")
st.caption("Ton assistant pédagogique personnel intelligent avec mémoire de session.")

# Vérification de la clé API avant de lancer le chat
if not client and not ("GROQ_API_KEY" in st.secrets):
    st.error(
        "⚠️ **Clé API Groq introuvable !** "
        "Veuillez configurer votre clé dans les **Secrets Streamlit** "
        "(`GROQ_API_KEY = \"gsk_...\"`)."
    )
    st.stop()

# Initialisation de la session pour l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis ton tuteur EduBot. Comment puis-je t'aider dans tes révisions aujourd'hui ?"
        }
    ]

# Affichage de l'historique des messages dans l'interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 6. GESTION DES ENTRÉES UTILISATEUR ET REQUÊTES API
# ==========================================
if prompt := st.chat_input("Écris ta question ici..."):
    # Ajout du message utilisateur à l'historique de session
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Affichage immédiat du message utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération de la réponse de l'assistant
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Réflexion en cours...*")
        
        try:
            # Appel à l'API Groq avec l'historique complet pour garder le contexte
            response = client.chat.completions.create(
                model=modele_selectionne,
                messages=st.session_state.messages,
                temperature=temperature
            )
            
            # Récupération de la réponse renvoyée par le modèle
            reply = response.choices[0].message.content
            message_placeholder.markdown(reply)
            
            # Ajout de la réponse de l'assistant dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": reply})
            
        except Exception as e:
            message_placeholder.error(f"❌ Une erreur technique est survenue : {e}")

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
