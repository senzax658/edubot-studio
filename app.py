import streamlit as st
import ollama
import urllib.parse

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="EduBot Studio - Espace d'Étude",
    page_icon="📚",
    layout="wide"
)

# --- 2. STYLE CSS STUDIEUX & ACADÉMIQUE ---
st.markdown("""
<style>
    /* Fond général sombre et élégant */
    .stApp {
        background-color: #0E131F;
        color: #E2E8F0;
    }
    
    /* Panneau d'accueil studieux */
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
    
    /* Style des boutons dans la sidebar */
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

# --- 3. INITIALISATION DES SESSIONS ET HISTORIQUE ---
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "Nouvelle séance d'étude"
    st.session_state.chats["Nouvelle séance d'étude"] = []

# --- 4. BARRE LATÉRALE (SIDEBAR STUDIO) ---
with st.sidebar:
    st.title("📚 EduBot Studio")
    st.caption("Espace de Révisions & Travail Guidé")
    st.divider()

    # Bouton nouvelle séance
    if st.button("📖 Nouvelle séance d'étude", use_container_width=True):
        count = len(st.session_state.chats) + 1
        new_title = f"Séance {count}"
        st.session_state.chats[new_title] = []
        st.session_state.current_chat_title = new_title
        st.rerun()

    st.write("### 📜 Vos fiches & conversations")
    
    # Liste dynamique des conversations
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

# --- 5. EN-TÊTE PRINCIPAL "LOOK STUDIEUX" ---
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

# Message d'accueil si la session est vide
if not messages:
    st.info("💡 **Séance prête.** Posez votre question ci-dessous ou demandez un résumé de cours pour démarrer.")

# --- 6. AFFICHAGE DE LA DISCUSSION ---
for msg in messages:
    avatar = "👤" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "image_url" in msg:
            st.image(msg["image_url"], caption="Schéma pédagogique généré par EduBot", use_column_width=True)

# --- 7. GENERATEUR DE SCHÉMAS ÉDUCATIFS SÉCURISÉ ---
def generate_educational_schema(prompt_text):
    # Prompt renforcé pour forcer des graphiques/schémas scientifiques et éviter les images hors-sujet
    clean_prompt = f"academic whiteboard diagram, clean educational vector graphic, scientific math physics illustration, high resolution chart, {prompt_text}"
    encoded_prompt = urllib.parse.quote(clean_prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=500&nologo=true"

# --- 8. TRAITEMENT DE LA SAISIE UTILISATEUR ---
if prompt := st.chat_input("Posez votre question de cours ou demandez un schéma..."):
    
    # Attribution du titre automatique au 1er message
    if len(messages) == 0:
        clean_title = prompt.strip()[:28] + ("..." if len(prompt) > 28 else "")
        st.session_state.chats[clean_title] = st.session_state.chats.pop(st.session_state.current_chat_title)
        st.session_state.current_chat_title = clean_title
        messages = st.session_state.chats[clean_title]

    # Ajout du message utilisateur
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Détection stricte de demande de visuel/schéma
    image_keywords = ["dessine", "dessin", "schéma", "schema", "illustration", "fais une image", "montre un schéma", "graphique"]
    wants_image = any(word in prompt.lower() for word in image_keywords)

    # Réponse d'EduBot
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("EduBot prépare l'explication et la synthèse..."):
            
            system_prompt = {
                "role": "system",
                "content": "Tu es EduBot, un tuteur pédagogique d'élite pour lycéens. Tes réponses sont rigoureuses, structurées, claires, et rédigées dans un ton académique encourageant."
            }
            response = ollama.chat(model="llama3", messages=[system_prompt] + messages)
            reply_text = response["message"]["content"]
            st.markdown(reply_text)
            
            msg_data = {"role": "assistant", "content": reply_text}

            # Génération du schéma sécurisé si demandé
            if wants_image:
                st.caption("🎨 *Génération du schéma pédagogique en cours...*")
                img_url = generate_educational_schema(prompt)
                st.image(img_url, caption="Schéma pédagogique généré par EduBot", use_column_width=True)
                msg_data["image_url"] = img_url

            # Enregistrement
            messages.append(msg_data)
            st.session_state.chats[st.session_state.current_chat_title] = messages