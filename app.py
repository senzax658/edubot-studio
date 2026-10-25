import base64
import hashlib
import hmac
import io
import logging
import os
import re
import sqlite3
import uuid
import warnings

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import groq
import streamlit as st
from groq import Groq
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduBot Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGGER = logging.getLogger(__name__)

PARIS = ZoneInfo("Europe/Paris")

DATA_DIR = Path(os.environ.get("EDUBOT_DATA_DIR", "edubot_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "edubot.sqlite3"

DAILY_IMAGE_LIMIT = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = r"""
Tu es EduBot, un assistant pédagogique francophone.

Ta mission :
- Aider l'élève à comprendre, réviser et progresser.
- Expliquer clairement les étapes importantes.
- Adapter ton vocabulaire au niveau demandé.
- Ne pas présenter une supposition comme un fait.
- Demander une précision si une information manque.

Présentation :
- Utilise des titres courts, des paragraphes et des listes.
- Utilise des tableaux Markdown quand ils sont utiles.
- Pour les mathématiques en ligne, utilise $...$.
- Pour les équations séparées, utilise $$...$$.
- Ne produis pas de diagrammes Mermaid.
- Ne produis pas de HTML ou de SVG.
- Ne prétends jamais générer une image.

Photos :
- Tu peux analyser les images jointes au message courant.
- Si une photo est illisible, demande une image plus nette.
- Ne devine pas les nombres ou les mots que tu ne peux pas lire.
- Les anciennes photos ne sont pas retransmises.
- Si tu as besoin de revoir une ancienne photo, explique-le clairement.
"""


# ============================================================
# 2. BASE DE DONNÉES
# ============================================================

def connect_db():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_db():
    with connect_db() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                title TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL
                    REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                image BLOB
            );

            CREATE TABLE IF NOT EXISTS usage (
                user_id TEXT NOT NULL REFERENCES users(id),
                day TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day)
            );
        """)


initialize_db()


def current_time():
    return datetime.now(PARIS).isoformat()


def current_day():
    return datetime.now(PARIS).date().isoformat()


def get_api_key():
    value = os.environ.get("GROQ_API_KEY", "").strip()

    if value:
        return value

    try:
        value = st.secrets.get("GROQ_API_KEY", "")
    except FileNotFoundError:
        return ""

    return value.strip() if isinstance(value, str) else ""


# ============================================================
# 3. COMPTES UTILISATEURS
# ============================================================

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600_000,
    )


def create_account(username, password):
    username = username.strip().lower()

    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        raise ValueError(
            "L'identifiant doit contenir 3 à 30 caractères : "
            "lettres sans accents, chiffres ou tiret bas _. "
            "Les espaces ne sont pas autorisés."
        )

    if not 12 <= len(password) <= 128:
        raise ValueError(
            "Le mot de passe doit contenir entre 12 et 128 caractères."
        )

    salt = os.urandom(16)

    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO users(id, username, salt, password_hash)
            VALUES (?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                username,
                salt,
                hash_password(password, salt),
            ),
        )


def authenticate(username, password):
    if len(password) > 128:
        return None

    with connect_db() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()

    salt = user["salt"] if user else b"\0" * 16
    candidate = hash_password(password, salt)

    if user and hmac.compare_digest(
        candidate,
        user["password_hash"],
    ):
        return {
            "id": user["id"],
            "username": user["username"],
        }

    return None


# ============================================================
# 4. HISTORIQUE ET QUOTAS
# ============================================================

def get_chats(user_id):
    with connect_db() as connection:
        return connection.execute(
            """
            SELECT * FROM chats
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()


def get_messages(chat_id, user_id):
    if not chat_id:
        return []

    with connect_db() as connection:
        return connection.execute(
            """
            SELECT m.*
            FROM messages m
            JOIN chats c ON c.id = m.chat_id
            WHERE c.id = ? AND c.user_id = ?
            ORDER BY m.id
            """,
            (chat_id, user_id),
        ).fetchall()


def get_image_count(user_id):
    with connect_db() as connection:
        row = connection.execute(
            """
            SELECT count FROM usage
            WHERE user_id = ? AND day = ?
            """,
            (user_id, current_day()),
        ).fetchone()

    return row["count"] if row else 0


def make_title(text):
    title = re.sub(r"\s+", " ", text).strip()
    return title[:57] + "…" if len(title) > 60 else title


def save_user_submission(user_id, chat_id, text, images):
    """
    Crée la discussion si nécessaire, réserve le quota
    et sauvegarde le message dans une seule transaction.
    """
    connection = connect_db()

    try:
        connection.execute("BEGIN IMMEDIATE")

        if chat_id:
            owned_chat = connection.execute(
                "SELECT id FROM chats WHERE id = ? AND user_id = ?",
                (chat_id, user_id),
            ).fetchone()

            if not owned_chat:
                raise ValueError("Cette discussion est introuvable.")

        day = current_day()

        row = connection.execute(
            "SELECT count FROM usage WHERE user_id = ? AND day = ?",
            (user_id, day),
        ).fetchone()

        used = row["count"] if row else 0
        remaining = max(0, DAILY_IMAGE_LIMIT - used)

        if len(images) > remaining:
            raise ValueError(
                f"Il te reste {remaining} image(s) disponible(s) "
                "aujourd'hui."
            )

        if not chat_id:
            chat_id = uuid.uuid4().hex

            connection.execute(
                """
                INSERT INTO chats(id, user_id, title, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, user_id, make_title(text), current_time()),
            )

        if images:
            connection.execute(
                """
                INSERT INTO usage(user_id, day, count)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, day)
                DO UPDATE SET count = count + excluded.count
                """,
                (user_id, day, len(images)),
            )

        connection.execute(
            """
            INSERT INTO messages(chat_id, role, content)
            VALUES (?, 'user', ?)
            """,
            (chat_id, text),
        )

        for image in images:
            connection.execute(
                """
                INSERT INTO messages(chat_id, role, content, image)
                VALUES (?, 'attachment', '', ?)
                """,
                (chat_id, image),
            )

        connection.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (current_time(), chat_id),
        )

        connection.commit()
        return chat_id

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_assistant_reply(chat_id, reply):
    with connect_db() as connection:
        connection.execute(
            """
            INSERT INTO messages(chat_id, role, content)
            VALUES (?, 'assistant', ?)
            """,
            (chat_id, reply),
        )

        connection.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (current_time(), chat_id),
        )


# ============================================================
# 5. FORMATAGE ET IMAGES
# ============================================================

def clean_text(text):
    text = re.sub(
        r"^[ \t]*```[ \t]*mermaid[^\n]*\n"
        r".*?(?:^[ \t]*```[ \t]*(?:\n|$)|\Z)",
        "\n\n[Schéma masqué : demande une explication textuelle.]\n\n",
        text or "",
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    parts = re.split(
        r"(```.*?```|`[^`\n]*`)",
        text,
        flags=re.DOTALL,
    )

    for index, part in enumerate(parts):
        if part.startswith("`"):
            continue

        part = re.sub(
            r"\\\[(.*?)\\\]",
            lambda match: (
                "\n\n$$\n"
                + match.group(1).strip()
                + "\n$$\n\n"
            ),
            part,
            flags=re.DOTALL,
        )

        part = re.sub(
            r"\\\((.*?)\\\)",
            lambda match: "$" + match.group(1).strip() + "$",
            part,
            flags=re.DOTALL,
        )

        parts[index] = part

    return "".join(parts).strip()


def prepare_image(upload):
    raw = upload.getvalue()

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Chaque image doit faire au maximum 5 Mo.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(io.BytesIO(raw)) as check:
                if check.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError(
                        "Formats autorisés : JPG, PNG et WebP."
                    )
                check.verify()

            with Image.open(io.BytesIO(raw)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((1600, 1600))

                output = io.BytesIO()
                image.save(output, format="JPEG", quality=85)

                return output.getvalue()

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError(
            "Une image est invalide, endommagée ou trop grande."
        ) from error


def display_api_error(error):
    if isinstance(error, groq.AuthenticationError):
        message = "La clé API Groq est invalide. Vérifie les secrets."

    elif isinstance(error, groq.RateLimitError):
        message = (
            "La limite d'utilisation Groq est atteinte. "
            "Vérifie les quotas du compte et réessaie plus tard."
        )

    elif isinstance(
        error,
        (groq.NotFoundError, groq.PermissionDeniedError),
    ):
        message = (
            "Le modèle est inaccessible. Vérifie sa disponibilité "
            "et les autorisations de ton compte Groq."
        )

    elif isinstance(error, groq.APITimeoutError):
        message = "Le délai de réponse est dépassé. Réessaie plus tard."

    elif isinstance(error, groq.APIConnectionError):
        message = "La connexion à Groq a échoué."

    elif isinstance(error, groq.BadRequestError):
        message = (
            "La demande a été refusée. Essaie une question plus courte "
            "ou une nouvelle discussion."
        )

    else:
        message = (
            "Une erreur est survenue. Consulte les journaux "
            "du serveur pour le diagnostic."
        )

    st.error(message)

    LOGGER.error(
        "Erreur EduBot : type=%s statut=%s",
        type(error).__name__,
        getattr(error, "status_code", None),
    )


# ============================================================
# 6. STYLE : COULEURS ET LISIBILITÉ
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                ellipse at top right,
                #282344 0%,
                #0e1220 65%
            );
        color: #f1f5f9;
        color-scheme: dark;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #111626;
        border-right: 1px solid #35405a;
    }

    /* Texte, titres, légendes et formules */

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #bac7dd !important;
    }

    [data-testid="stMarkdownContainer"] a {
        color: #d3bfff !important;
    }

    .katex {
        color: #f1f5f9 !important;
    }

    /* Messages */

    [data-testid="stChatMessage"] {
        background: #192136 !important;
        border: 1px solid #3c4863;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
    }

    [data-testid="stChatMessageContent"] {
        color: #f1f5f9 !important;
    }

    /* Tableaux */

    [data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
    }

    [data-testid="stMarkdownContainer"] th {
        background: #303958 !important;
        color: #ffffff !important;
        border: 1px solid #53617e !important;
        padding: 12px;
    }

    [data-testid="stMarkdownContainer"] td {
        background: #1b253b !important;
        color: #edf2fa !important;
        border: 1px solid #53617e !important;
        padding: 12px;
    }

    /* Formulaires */

    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="base-input"],
    [data-testid="stTextArea"] [data-baseweb="textarea"] {
        background: #222d46 !important;
        color: #f8fafc !important;
        border-color: #667596 !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #222d46 !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        caret-color: #d3bfff;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #bdc9df !important;
        -webkit-text-fill-color: #bdc9df !important;
        opacity: 1;
    }

    [data-testid="stTextInput"] button {
        color: #f1f5f9 !important;
    }

    [data-testid="stForm"] {
        background: #151d30;
        border: 1px solid #3c4863;
        border-radius: 18px;
    }

    /* Zone de saisie du chat */

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: #0e1220 !important;
    }

    [data-testid="stChatInput"] {
        background: #222d46 !important;
        border: 1px solid #7c89ae !important;
        border-radius: 18px;
        color: #f8fafc !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        caret-color: #d3bfff;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #c3cee2 !important;
        -webkit-text-fill-color: #c3cee2 !important;
        opacity: 1;
    }

    [data-testid="stChatInput"] button {
        color: #ede9fe !important;
    }

    /* Boutons */

    .stButton > button,
    .stFormSubmitButton > button {
        background: #293550;
        color: #f8fafc !important;
        border: 1px solid #617092;
        border-radius: 12px;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #394969;
        border-color: #d3bfff;
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(120deg, #6941c6, #8736a8);
        border: 1px solid #b69af1;
        color: #ffffff !important;
    }

    button:disabled {
        opacity: 0.5;
    }

    /* Onglets et panneaux */

    [data-testid="stTab"] {
        color: #d3dcee !important;
    }

    [data-testid="stTab"][aria-selected="true"] {
        color: #dfc7ff !important;
    }

    [data-testid="stExpander"] {
        background: #182136;
        border: 1px solid #3c4863;
        border-radius: 12px;
    }

    /* Bandeau d'accueil */

    .hero {
        padding: 30px;
        border-radius: 24px;
        border: 1px solid #655484;
        background: linear-gradient(120deg, #30264f, #1c3048);
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff !important;
        margin: 0 0 8px 0;
        font-size: clamp(28px, 5vw, 42px);
    }

    .hero p {
        color: #d8e1f1 !important;
        margin: 0;
        line-height: 1.6;
    }

    .badge {
        color: #e0caff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    @media (max-width: 640px) {
        .hero {
            padding: 22px;
        }

        [data-testid="stChatMessage"] {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="badge">APPRENDRE • COMPRENDRE • PROGRESSER</div>
        <h1>🎓 EduBot Studio</h1>
        <p>
            Ton espace pour réviser, comprendre tes cours
            et avancer dans tes exercices.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. CONNEXION ET INSCRIPTION
# ============================================================

if "user" not in st.session_state:
    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.subheader("Bienvenue dans ton espace ✨")
        st.caption("Connecte-toi pour retrouver tes discussions.")

        login_tab, register_tab = st.tabs(
            ["🔑 Connexion", "✨ Créer un compte"]
        )

        with login_tab:
            with st.form("login_form"):
                username = st.text_input(
                    "Identifiant",
                    placeholder="Exemple : amine_esc",
                    max_chars=30,
                )

                password = st.text_input(
                    "Mot de passe",
                    type="password",
                )

                login = st.form_submit_button(
                    "Me connecter",
                    type="primary",
                    use_container_width=True,
                )

            if login:
                user = authenticate(username, password)

                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")

        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input(
                    "Choisis un identifiant",
                    placeholder="Exemple : amine_esc",
                    max_chars=30,
                    help=(
                        "3 à 30 caractères, sans espaces. "
                        "Lettres sans accents, chiffres et _ autorisés."
                    ),
                )

                new_password = st.text_input(
                    "Choisis un mot de passe",
                    type="password",
                    help="Entre 12 et 128 caractères.",
                )

                confirm_password = st.text_input(
                    "Confirme ton mot de passe",
                    type="password",
                )

                register = st.form_submit_button(
                    "Créer mon compte",
                    type="primary",
                    use_container_width=True,
                )

            if register:
                if new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    try:
                        create_account(new_username, new_password)
                        st.success(
                            "Compte créé ! Ouvre l'onglet Connexion."
                        )
                    except ValueError as error:
                        st.error(str(error))
                    except sqlite3.IntegrityError:
                        st.error("Cet identifiant est déjà utilisé.")

        st.caption(
            "Version prototype : pas de récupération de mot de passe. "
            "N'utilise pas un mot de passe déjà utilisé ailleurs."
        )

    st.stop()


# ============================================================
# 8. BARRE LATÉRALE ET HISTORIQUE
# ============================================================

user = st.session_state.user
user_id = user["id"]
chats = get_chats(user_id)

with st.sidebar:
    st.title("🎓 Mon espace")
    st.write(f"Bonjour, {user['username']} 👋")

    if st.button(
        "＋ Nouvelle discussion",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.pop("chat_id", None)
        st.rerun()

    st.divider()

    image_count = get_image_count(user_id)
    remaining = max(0, DAILY_IMAGE_LIMIT - image_count)

    st.subheader("📷 Tes photos")
    st.progress(remaining / DAILY_IMAGE_LIMIT)
    st.caption(
        f"{remaining} / {DAILY_IMAGE_LIMIT} images restantes aujourd'hui"
    )
    st.caption("Remise à zéro à minuit, heure de Paris.")

    st.divider()
    st.subheader("💬 Mes discussions")

    search = st.text_input(
        "Rechercher une discussion",
        placeholder="Maths, français, histoire…",
    )

    visible_chats = [
        chat
        for chat in chats
        if search.casefold() in chat["title"].casefold()
    ]

    for chat in visible_chats:
        selected = st.session_state.get("chat_id") == chat["id"]

        if st.button(
            ("📌 " if selected else "💬 ") + chat["title"],
            key=f"open_{chat['id']}",
            use_container_width=True,
        ):
            st.session_state.chat_id = chat["id"]
            st.rerun()

    if not visible_chats:
        st.caption(
            "Aucune discussion trouvée."
            if search
            else "Tes discussions apparaîtront ici."
        )

    st.divider()

    if st.button(
        "🚪 Me déconnecter",
        use_container_width=True,
    ):
        st.session_state.clear()
        st.rerun()


# ============================================================
# 9. DISCUSSION ACTIVE
# ============================================================

chat_id = st.session_state.get("chat_id")
active_chat = next(
    (chat for chat in chats if chat["id"] == chat_id),
    None,
)

if active_chat is None:
    chat_id = None
    st.session_state.pop("chat_id", None)

history = get_messages(chat_id, user_id)

if active_chat:
    st.subheader(active_chat["title"])

    with st.expander("✏️ Modifier le thème de cette discussion"):
        with st.form(f"rename_{chat_id}"):
            new_title = st.text_input(
                "Titre / thème",
                value=active_chat["title"],
                max_chars=80,
            )

            rename = st.form_submit_button("Enregistrer")

        if rename:
            if not new_title.strip():
                st.warning("Le titre ne peut pas être vide.")
            else:
                with connect_db() as connection:
                    connection.execute(
                        """
                        UPDATE chats SET title = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (new_title.strip(), chat_id, user_id),
                    )
                st.rerun()

else:
    st.subheader("Que veux-tu apprendre aujourd'hui ?")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 📐 Comprendre")
            st.write("Une notion difficile ? On la reprend ensemble.")

    with col2:
        with st.container(border=True):
            st.markdown("### 📷 Analyser")
            st.write("Ajoute une photo pour expliquer ton exercice.")

    with col3:
        with st.container(border=True):
            st.markdown("### 📝 Réviser")
            st.write("Prépare ton contrôle avec des explications claires.")

st.caption(
    "📎 Photos : JPG, PNG ou WebP • 5 Mo maximum par image. "
    "Les photos envoyées sont conservées dans ton historique "
    "et transmises à Groq pour analyse."
)

for message in history:
    if message["role"] == "attachment":
        st.image(message["image"], width=320)
        continue

    avatar = "🎓" if message["role"] == "assistant" else "👤"

    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            st.markdown(clean_text(message["content"]))
        else:
            st.markdown(message["content"])


# ============================================================
# 10. ENVOI D'UN MESSAGE
# ============================================================

api_key = get_api_key()

if not api_key:
    st.warning(
        "La clé Groq est absente. Ajoute GROQ_API_KEY "
        "dans les secrets de l'application."
    )

submission = st.chat_input(
    "Écris ta question ou ajoute une photo avec +",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=5,
    disabled=not bool(api_key),
    key=f"message_{chat_id or 'new'}",
)

if submission:
    text = (submission.text or "").strip()
    files = list(submission.files or [])

    if not text and not files:
        st.stop()

    if len(files) > DAILY_IMAGE_LIMIT:
        st.error("Tu peux envoyer au maximum 3 images.")
        st.stop()

    if len(files) > max(
        0,
        DAILY_IMAGE_LIMIT - get_image_count(user_id),
    ):
        st.error(
            "Tu n'as plus assez d'images disponibles aujourd'hui. "
            "Tu peux continuer à envoyer des messages sans photo."
        )
        st.stop()

    try:
        images = [prepare_image(upload) for upload in files]
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if not text:
        text = (
            "Explique ce qui apparaît sur ces photos et "
            "aide-moi à comprendre l'exercice étape par étape."
        )

    try:
        chat_id = save_user_submission(
            user_id,
            chat_id,
            text,
            images,
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.session_state.chat_id = chat_id

    with st.chat_message("user", avatar="👤"):
        st.markdown(text)
        for image in images:
            st.image(image, width=320)

    # Les anciennes photos restent dans l'historique,
    # mais seules les nouvelles photos sont envoyées au modèle.
    previous_messages = [
        {
            "role": message["role"],
            "content": clean_text(message["content"])
            if message["role"] == "assistant"
            else message["content"],
        }
        for message in history
        if message["role"] in {"user", "assistant"}
    ][-20:]

    if images:
        content = [{"type": "text", "text": text}]

        for image in images:
            encoded = base64.b64encode(image).decode("ascii")

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}"
                    },
                }
            )

        selected_model = VISION_MODEL

    else:
        content = text
        selected_model = TEXT_MODEL

    api_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *previous_messages,
        {
            "role": "user",
            "content": content,
        },
    ]

    reply = None

    with st.chat_message("assistant", avatar="🎓"):
        try:
            with st.spinner("EduBot prépare ton explication…"):
                client = Groq(
                    api_key=api_key,
                    timeout=90.0,
                    max_retries=0,
                )

                result = client.chat.completions.create(
                    model=selected_model,
                    messages=api_messages,
                    temperature=0.4,
                    max_completion_tokens=2500,
                )

                reply = clean_text(
                    result.choices[0].message.content
                )

                if not reply:
                    raise ValueError("Réponse vide.")

                save_assistant_reply(chat_id, reply)

        except Exception as error:
            reply = None
            display_api_error(error)

            st.caption(
                "Ton message reste dans l'historique. "
                "Les photos soumises restent comptées "
                "dans le quota journalier, même si l'analyse échoue."
            )

    if reply:
        st.rerun()


# ============================================================
# 11. PIED DE PAGE
# ============================================================

st.divider()
st.caption(
    "EduBot Studio • Comprendre plutôt que recopier • "
    "Les réponses peuvent contenir des erreurs."
)
