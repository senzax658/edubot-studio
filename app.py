import logging
import os
import re

import groq
import streamlit as st
from groq import Groq


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EduBot Studio",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

MESSAGE_BIENVENUE = (
    "Bonjour ! Je suis EduBot, ton assistant pédagogique. "
    "Je peux t'aider à comprendre tes cours, réviser "
    "et résoudre des exercices."
)

INSTRUCTIONS_SYSTEME = r"""
Tu es EduBot, un tuteur pédagogique francophone.

OBJECTIF
- Explique les notions clairement.
- Adapte ton vocabulaire à la question de l'élève.
- Pour les exercices, détaille les étapes importantes.
- Si une information manque, demande une précision.
- Ne présente pas une supposition comme un fait.

PRÉSENTATION
- Utilise du Markdown simple : titres, paragraphes, listes et tableaux.
- Pour une formule mathématique en ligne, utilise $...$.
- Pour une équation séparée, utilise des délimiteurs $$ sur des lignes distinctes.
- N'utilise pas les délimiteurs \( \) ou \[ \].
- Pour les tableaux de signes, utilise un tableau Markdown.
- Pour les vecteurs, utilise des formules LaTeX et des explications textuelles.

IMAGES ET SCHÉMAS
- Cette application ne génère pas d'images.
- Ne prétends jamais avoir créé, affiché ou joint une image.
- Si l'utilisateur demande une image ou un dessin :
  explique brièvement que tu peux proposer une description
  ou expliquer comment le tracer, mais pas générer une image ici.
- Ne génère jamais de code Mermaid.
- Ne génère jamais de bloc de code de diagramme.
- Ne génère pas de HTML ou de SVG pour remplacer une image.
- Préfère les listes, les tableaux et les explications étape par étape.
"""

MODELES = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


# ==========================================
# 2. FONCTIONS
# ==========================================

def recuperer_cle_groq():
    """Cherche la clé dans l'environnement, puis les secrets Streamlit."""
    valeur = os.environ.get("GROQ_API_KEY")

    if valeur and valeur.strip():
        return valeur.strip()

    try:
        valeur = st.secrets.get("GROQ_API_KEY")
    except FileNotFoundError:
        return None

    if isinstance(valeur, str) and valeur.strip():
        return valeur.strip()

    return None


def nouveaux_messages():
    return [
        {
            "role": "assistant",
            "content": MESSAGE_BIENVENUE,
        }
    ]


def nettoyer_texte(texte):
    """
    Adapte les formules hors des blocs de code
    et supprime les blocs Mermaid éventuels.
    """
    if not texte:
        return "Je n'ai pas reçu de réponse. Réessaie."

    # Retire les blocs Mermaid fermés, ou jusqu'à la fin du texte
    # si le modèle oublie les délimiteurs de fermeture.
    texte = re.sub(
        r"^[ \t]*```[ \t]*mermaid[^\n]*\n"
        r".*?(?:^[ \t]*```[ \t]*(?:\n|$)|\Z)",
        "\n\n[Schéma Mermaid masqué : "
        "demande une explication ou un tableau à la place.]\n\n",
        texte,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    # Préserve les blocs de code et le code en ligne.
    parties = re.split(
        r"(```.*?```|`[^`\n]*`)",
        texte,
        flags=re.DOTALL,
    )

    for index, partie in enumerate(parties):
        if partie.startswith("`"):
            continue

        # Convertit les formules séparées \[ ... \].
        partie = re.sub(
            r"\\\[(.*?)\\\]",
            lambda resultat: (
                "\n\n$$\n"
                + resultat.group(1).strip()
                + "\n$$\n\n"
            ),
            partie,
            flags=re.DOTALL,
        )

        # Convertit les formules en ligne \( ... \).
        partie = re.sub(
            r"\\\((.*?)\\\)",
            lambda resultat: "$" + resultat.group(1).strip() + "$",
            partie,
            flags=re.DOTALL,
        )

        parties[index] = partie

    return "".join(parties).strip()


def afficher_erreur(erreur):
    """Affiche une erreur compréhensible sans montrer les détails internes."""
    if isinstance(erreur, groq.AuthenticationError):
        st.error(
            "La clé API Groq est invalide ou révoquée. "
            "Vérifie GROQ_API_KEY dans tes secrets."
        )

    elif isinstance(erreur, groq.PermissionDeniedError):
        st.error(
            "Accès refusé. Vérifie les autorisations "
            "de ton compte Groq."
        )

    elif isinstance(erreur, groq.RateLimitError):
        st.error(
            "La limite d'utilisation Groq est atteinte. "
            "Patiente avant de réessayer et vérifie tes quotas."
        )

    elif isinstance(erreur, groq.NotFoundError):
        st.error(
            "Le modèle sélectionné est introuvable ou inaccessible. "
            "Essaie l'autre modèle dans la barre latérale."
        )

    elif isinstance(erreur, groq.APITimeoutError):
        st.error(
            "Groq a mis trop de temps à répondre. "
            "Réessaie dans un moment."
        )

    elif isinstance(erreur, groq.APIConnectionError):
        st.error(
            "Connexion à Groq impossible. "
            "Vérifie la connexion réseau du serveur."
        )

    elif isinstance(erreur, groq.BadRequestError):
        st.error(
            "La requête a été refusée. Si la conversation est longue, "
            "efface l'historique puis réessaie."
        )

    else:
        st.error(
            "Une erreur technique est survenue. "
            "Consulte les journaux du serveur pour le diagnostic."
        )

    logger.error(
        "Erreur Groq : type=%s, statut=%s, request_id=%s",
        type(erreur).__name__,
        getattr(erreur, "status_code", None),
        getattr(erreur, "request_id", None),
    )


# ==========================================
# 3. STYLE
# ==========================================

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        .stChatMessage {
            border-radius: 12px;
            margin-bottom: 12px;
        }

        h1 {
            color: #1e3a8a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 4. INITIALISATION DE LA SESSION
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = nouveaux_messages()


# ==========================================
# 5. BARRE LATÉRALE
# ==========================================

with st.sidebar:
    st.title("📚 EduBot Studio")
    st.caption("Ton assistant de révision")

    st.divider()

    modele = st.selectbox(
        "Modèle Groq",
        MODELES,
        index=0,
    )

    temperature = st.slider(
        "Créativité",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
    )

    st.divider()

    if st.button(
        "🗑️ Effacer la conversation",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.messages = nouveaux_messages()
        st.rerun()

    st.info(
        "Cette version répond avec du texte, des formules "
        "et des tableaux. Elle ne génère pas d'images "
        "ni de diagrammes Mermaid."
    )


# ==========================================
# 6. INTERFACE PRINCIPALE
# ==========================================

st.title("📚 EduBot Studio")
st.caption("Comprendre, réviser et progresser.")

cle_groq = recuperer_cle_groq()

if not cle_groq:
    st.warning(
        "Ajoute ta clé GROQ_API_KEY dans les secrets Streamlit "
        "pour activer le chat."
    )

    st.code(
        'GROQ_API_KEY = "ta_cle_groq"',
        language="toml",
    )

# Nettoie aussi les anciens messages lors de leur affichage.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(nettoyer_texte(message["content"]))
        else:
            st.markdown(message["content"])


# ==========================================
# 7. QUESTION ET RÉPONSE
# ==========================================

question = st.chat_input(
    "Pose ta question ici…",
    disabled=not bool(cle_groq),
)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Le message d'accueil n'est pas envoyé au modèle.
    # Les anciennes réponses sont nettoyées avant réutilisation.
    historique = []

    for message in st.session_state.messages[1:]:
        contenu = message["content"]

        if message["role"] == "assistant":
            contenu = nettoyer_texte(contenu)

        historique.append(
            {
                "role": message["role"],
                "content": contenu,
            }
        )

    messages_api = [
        {
            "role": "system",
            "content": INSTRUCTIONS_SYSTEME,
        },
        *historique,
        {
            "role": "user",
            "content": question,
        },
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Réflexion en cours…"):
                client = Groq(
                    api_key=cle_groq,
                    timeout=90.0,
                    max_retries=0,
                )

                resultat = client.chat.completions.create(
                    model=modele,
                    messages=messages_api,
                    temperature=temperature,
                    max_completion_tokens=2048,
                )

                contenu = resultat.choices[0].message.content

                if not contenu or not contenu.strip():
                    raise ValueError("Réponse vide.")

                reponse = nettoyer_texte(contenu)

            st.markdown(reponse)

            # Mémorise uniquement les échanges réussis.
            st.session_state.messages.extend(
                [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": reponse,
                    },
                ]
            )

        except Exception as erreur:
            afficher_erreur(erreur)
            st.caption(
                "Cet échange n'a pas été ajouté à l'historique. "
                "Tu peux renvoyer ta question."
            )


# ==========================================
# 8. PIED DE PAGE
# ==========================================

st.divider()
st.caption("EduBot Studio • Streamlit & Groq")
