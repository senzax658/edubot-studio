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

import streamlit as st
from groq import Groq
from PIL import Image, ImageOps, UnidentifiedImageError


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="EduBot • Ton espace de révision",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGGER = logging.getLogger(__name__)
PARIS = ZoneInfo("Europe/Paris")

DATA_DIR = Path(os.environ.get("EDUBOT_DATA_DIR", "edubot_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "edubot.sqlite3"

DAILY_LIMIT = 3
MAX_FILE_BYTES = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = r"""
Tu es EduBot, un assistant pédagogique francophone.
Explique clairement, étape par étape, en adaptant ton niveau à l'élève.
Ne présente jamais une supposition comme un fait.

Présentation :
- Utilise des titres courts, des listes et des tableaux Markdown.
- Écris les formules en ligne avec $...$.
- Écris les équations séparées avec $$...$$.
- Ne produis jamais de Mermaid, de HTML ou de SVG.
- Ne prétends pas générer des images.

Images :
- Tu peux analyser les photos jointes au message courant.
- Si le texte est illisible, demande une photo plus nette.
- Ne devine pas les valeurs que tu ne peux pas lire.
- Les anciennes photos ne sont pas retransmises : utilise les
  explications textuelles précédentes et signale toute incertitude.
"""


# ==================================================
# BASE DE DONNÉES
# ==================================================

def db():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


with db() as connection:
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
            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
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


def now():
    return datetime.now(PARIS).isoformat()


def today():
    return datetime.now(PARIS).date().isoformat()


def get_key():
    value = os.environ.get("GROQ_API_KEY")
    if value:
        return value.strip()
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except FileNotFoundError:
        return ""


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 600_000
    )


def create_account(username, password):
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        raise ValueError(
            "Identifiant : 3 à 30 caractères, lettres sans accents, "
            "chiffres ou tiret bas."
        )
    if not 12 <= len(password) <= 128:
        raise ValueError("Choisis un mot de passe de 12 à 128 caractères.")

    salt = os.urandom(16)
    with db() as connection:
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                username,
                salt,
                password_hash(password, salt),
            ),
        )


def authenticate(username, password):
    if len(password) > 128:
        return None

    with db() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()

    # Effectue aussi un calcul en cas d'identifiant inconnu.
    salt = user["salt"] if user else b"\0" * 16
    candidate = password_hash(password, salt)

    if user and hmac.compare_digest(candidate, user["password_hash"]):
        return {"id": user["id"], "username": user["username"]}
    return None


def list_chats(user_id):
    with db() as connection:
        return connection.execute(
            "SELECT * FROM chats WHERE user_id = ? "
            "ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()


def load_messages(chat_id, user_id):
    with db() as connection:
        return connection.execute(
            """
            SELECT m.* FROM messages m
            JOIN chats c ON c.id = m.chat_id
            WHERE c.id = ? AND c.user_id = ?
            ORDER BY m.id
            """,
            (chat_id, user_id),
        ).fetchall()


def used_images(user_id):
    with db() as connection:
        row = connection.execute(
            "SELECT count FROM usage WHERE user_id = ? AND day = ?",
            (user_id, today()),
        ).fetchone()
    return row["count"] if row else 0


def reserve_and_save(user_id, chat_id, text, images):
    """Réserve atomiquement le quota et sauvegarde le message utilisateur."""
    connection = db()
    try:
        connection.execute("BEGIN IMMEDIATE")

        chat = connection.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id),
        ).fetchone()
        if not chat:
            raise ValueError("Discussion introuvable.")

        day = today()
        row = connection.execute(
            "SELECT count FROM usage WHERE user_id = ? AND day = ?",
            (user_id, day),
        ).fetchone()
        count = row["count"] if row else 0

        if count + len(images) > DAILY_LIMIT:
            raise ValueError(
                f"Quota dépassé : il reste {max(0, DAILY_LIMIT - count)} "
                "image(s) aujourd'hui."
            )

        if images:
            connection.execute(
                """
                INSERT INTO usage(user_id, day, count) VALUES (?, ?, ?)
                ON CONFLICT(user_id, day)
                DO UPDATE SET count = count + excluded.count
                """,
                (user_id, day, len(images)),
            )

        connection.execute(
            "INSERT INTO messages(chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, "user", text),
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
            "UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?",
            (now(), chat_id, user_id),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ==================================================
# FORMATAGE ET VALIDATION
# ==================================================

def clean_text(text):
    text = re.sub(
        r"^[ \t]*```[ \t]*mermaid[^\n]*\n"
        r".*?(?:^[ \t]*```[ \t]*(?:\n|$)|\Z)",
        "\n\n[Schéma non affiché : demande une explication textuelle.]\n\n",
        text or "",
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.DOTALL)

    for index, part in enumerate(parts):
        if part.startswith("`"):
            continue
        part = re.sub(
            r"\\\[(.*?)\\\]",
            lambda match: "\n\n$$\n" + match.group(1).strip() + "\n$$\n\n",
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
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Chaque image doit faire au maximum 5 Mo.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(io.BytesIO(raw)) as check:
                if check.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Formats autorisés : JPG, PNG et WebP.")
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
        raise ValueError("Image invalide, trop grande ou endommagée.") from error


def automatic_title(text):
    title = re.sub(r"\s+", " ", text).strip()
    return title[:57] + "…" if len(title) > 60 else title


# ==================================================
# INTERFACE
# ==================================================

st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at top right, #222045 0%, #0e1220 60%);
    color: #edf0fa;
}
[data-testid="stSidebar"] {
    background: #111626;
    border-right: 1px solid #2b3047;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1120px;
    padding-top: 2rem;
}
.stChatMessage {
    background: #191f32;
    border: 1px solid #30364f;
    border-radius: 18px;
    margin-bottom: 14px;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(120deg, #7656f5, #a650c7);
    border: none;
    color: white;
    border-radius: 12px;
}
.hero {
    padding: 30px;
    border-radius: 24px;
    border: 1px solid #454067;
    background: linear-gradient(120deg, #252447, #19243a);
    margin-bottom: 24px;
}
.hero h1 {color: #ffffff; margin: 0 0 8px 0;}
.hero p {color: #c2c8e0; margin: 0;}
.badge {
    color: #d2bdff;
    font-size: 13px;
    letter-spacing: 2px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="badge">TON ESPACE POUR PROGRESSER</div>
    <h1>🎓 EduBot Studio</h1>
    <p>Comprends tes cours. Débloque tes exercices. Révise à ton rythme.</p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# CONNEXION
# ==================================================

if "user" not in st.session_state:
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.subheader("Bienvenue dans ton espace ✨")
        st.caption("Connecte-toi pour retrouver tes discussions.")

        login_tab, register_tab = st.tabs(["Connexion", "Créer un compte"])

        with login_tab:
            with st.form("login"):
                username = st.text_input("Identifiant", max_chars=30)
                password = st.text_input("Mot de passe", type="password")
                submit = st.form_submit_button(
                    "Entrer dans mon espace", type="primary"
                )

            if submit:
                user = authenticate(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")

        with register_tab:
            with st.form("register"):
                new_username = st.text_input(
                    "Choisis un identifiant", max_chars=30
                )
                new_password = st.text_input(
                    "Choisis un mot de passe", type="password"
                )
                confirm = st.text_input(
                    "Confirme le mot de passe", type="password"
                )
                register = st.form_submit_button("Créer mon compte")

            if register:
                if new_password != confirm:
                    st.error("Les mots de passe ne correspondent pas.")
                else:
                    try:
                        create_account(new_username, new_password)
                        st.success("Compte créé ! Connecte-toi dans l'autre onglet.")
                    except ValueError as error:
                        st.error(str(error))
                    except sqlite3.IntegrityError:
                        st.error("Cet identifiant est déjà utilisé.")

        st.caption(
            "Prototype : pas de récupération de mot de passe. "
            "N'utilise pas un mot de passe déjà utilisé ailleurs."
        )

    st.stop()


# ==================================================
# SIDEBAR ET HISTORIQUE
# ==================================================

user = st.session_state.user
user_id = user["id"]

with st.sidebar:
    st.title("🎓 EduBot")
    st.write(f"Bonjour, {user['username']} 👋")

    if st.button("＋ Nouvelle discussion", type="primary", use_container_width=True):
        st.session_state.pop("chat_id", None)
        st.rerun()

    remaining = max(0, DAILY_LIMIT - used_images(user_id))
    st.divider()
    st.caption("PHOTOS D'EXERCICES")
    st.progress(remaining / DAILY_LIMIT)
    st.caption(f"{remaining} / {DAILY_LIMIT} images restantes aujourd'hui")
    st.caption("Réinitialisation à minuit — heure de Paris.")

    st.divider()
    st.subheader("Tes discussions")
    search = st.text_input(
        "Rechercher un thème", placeholder="Vecteurs, histoire…"
    )

    chats = list_chats(user_id)
    for chat in chats:
        if search.casefold() not in chat["title"].casefold():
            continue
        if st.button(
            "💬 " + chat["title"],
            key=f"chat_{chat['id']}",
            use_container_width=True,
        ):
            st.session_state.chat_id = chat["id"]
            st.rerun()

    if not chats:
        st.caption("Tes discussions apparaîtront ici.")

    st.divider()
    if st.button("Se déconnecter", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ==================================================
# DISCUSSION ACTIVE
# ==================================================

chat_id = st.session_state.get("chat_id")
active = next((chat for chat in chats if chat["id"] == chat_id), None)

if not active:
    chat_id = None
    st.session_state.pop("chat_id", None)

history = load_messages(chat_id, user_id) if chat_id else []

if active:
    st.subheader(active["title"])
    with st.expander("✏️ Modifier le thème de cette discussion"):
        with st.form(f"rename_{chat_id}"):
            title = st.text_input(
                "Titre / thème", value=active["title"], max_chars=80
            )
            if st.form_submit_button("Enregistrer") and title.strip():
                with db() as connection:
                    connection.execute(
                        "UPDATE chats SET title = ? WHERE id = ? AND user_id = ?",
                        (title.strip(), chat_id, user_id),
                    )
                st.rerun()
else:
    st.subheader("Sur quoi veux-tu travailler aujourd'hui ?")
    c1, c2, c3 = st.columns(3)
    c1.info("📐 Comprendre une notion")
    c2.info("📷 Étudier une photo d'exercice")
    c3.info("📝 Préparer une évaluation")

st.caption(
    "📎 Ajoute une photo avec le bouton de pièce jointe. "
    "JPG, PNG ou WebP • 5 Mo maximum par image."
)

for message in history:
    if message["role"] == "attachment":
        st.image(message["image"], width=300)
    else:
        avatar = "🎓" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(clean_text(message["content"]))

key = get_key()
if not key:
    st.warning("Configure GROQ_API_KEY dans les secrets pour activer EduBot.")

submission = st.chat_input(
    "Pose ta question ou joins une photo d'exercice…",
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=5,
    disabled=not bool(key),
    key=f"input_{chat_id or 'new'}",
)

if submission:
    text = submission.text.strip()
    files = list(submission.files or [])

    if not text and not files:
        st.stop()

    if len(files) > DAILY_LIMIT:
        st.error("Tu peux envoyer au maximum 3 images.")
        st.stop()

    try:
        images = [prepare_image(file) for file in files]
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if len(images) > max(0, DAILY_LIMIT - used_images(user_id)):
        st.error("Tu n'as plus assez d'images disponibles aujourd'hui.")
        st.stop()

    if not text:
        text = "Explique cet exercice et aide-moi à le résoudre étape par étape."

    if chat_id is None:
        chat_id = uuid.uuid4().hex
        with db() as connection:
            connection.execute(
                "INSERT INTO chats VALUES (?, ?, ?, ?)",
                (chat_id, user_id, automatic_title(text), now()),
            )
        st.session_state.chat_id = chat_id

    try:
        reserve_and_save(user_id, chat_id, text, images)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    with st.chat_message("user", avatar="👤"):
        st.markdown(text)
        for image in images:
            st.image(image, width=300)

    # Garde les derniers échanges textuels.
    # Les anciennes images restent visibles mais ne sont pas retransmises.
    previous = [
        {"role": message["role"], "content": message["content"]}
        for message in history
        if message["role"] in {"user", "assistant"}
    ][-20:]

    content = [{"type": "text", "text": text}]
    for image in images:
        encoded = base64.b64encode(image).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        })

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *previous,
        {"role": "user", "content": content if images else text},
    ]

    with st.chat_message("assistant", avatar="🎓"):
        try:
            with st.spinner("EduBot prépare ton explication…"):
                client = Groq(api_key=key, timeout=90, max_retries=0)
                response = client.chat.completions.create(
                    model=VISION_MODEL if images else TEXT_MODEL,
                    messages=api_messages,
                    temperature=0.4,
                    max_completion_tokens=2500,
                )
                reply = clean_text(response.choices[0].message.content)
                if not reply:
                    raise ValueError("Réponse vide.")

                with db() as connection:
                    connection.execute(
                        """
                        INSERT INTO messages(chat_id, role, content)
                        VALUES (?, 'assistant', ?)
                        """,
                        (chat_id, reply),
                    )
                    connection.execute(
                        "UPDATE chats SET updated_at = ? WHERE id = ?",
                        (now(), chat_id),
                    )

            st.rerun()

        except Exception as error:
            LOGGER.error(
                "Erreur EduBot : type=%s statut=%s",
                type(error).__name__,
                getattr(error, "status_code", None),
            )
            st.error(
                "EduBot n'a pas pu répondre. Vérifie la clé Groq, "
                "les quotas et l'accès au modèle utilisé."
            )
            st.caption(
                "Le message reste sauvegardé. Les images soumises "
                "restent comptées dans le quota journalier."
            )

st.divider()
st.caption("EduBot Studio • Un espace pour apprendre, pas simplement recopier.")
