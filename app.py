import base64
import io
import logging
import os
import re
import uuid
import warnings

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import groq
import psycopg
import streamlit as st
from groq import Groq
from PIL import Image, ImageOps, UnidentifiedImageError
from dotenv import load_dotenv


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="EduBot Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGGER = logging.getLogger(__name__)
PARIS = ZoneInfo("Europe/Paris")

DATABASE_URL = (
    os.environ.get("DATABASE_URL", "")
    or st.secrets.get("DATABASE_URL", "")
).strip()

DAILY_IMAGE_LIMIT = 3
CHAT_MESSAGE_LIMIT = 20
MAX_IMAGE_BYTES = 5 * 1024 * 1024

Image.MAX_IMAGE_PIXELS = 20_000_000

TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"

MODES = {
    "course": "📖 Comprendre mon cours",
    "exercise": "🧩 M’aider sur un exercice",
    "quiz": "🎯 Me tester",
}

MODE_DESCRIPTIONS = {
    "course": "Une explication claire, un exemple et les points essentiels.",
    "exercise": "Des indices progressifs pour avancer dans ton exercice.",
    "quiz": "Une question à la fois, puis une correction expliquée.",
}

PLACEHOLDERS = {
    "course": "Explique-moi les fonctions affines, niveau seconde…",
    "exercise": "Écris l’exercice ou ajoute une photo avec +…",
    "quiz": "Un quiz de 5 questions sur les vecteurs, niveau seconde…",
}

PHOTO_REQUESTS = {
    "course": (
        "Explique les notions lisibles sur ces photos "
        "avec un exemple et un résumé."
    ),
    "exercise": (
        "Aide-moi sur cet exercice. Commence par un indice "
        "sans donner toute la correction."
    ),
    "quiz": (
        "Crée un quiz de 5 questions sur le contenu lisible "
        "de ces photos. Pose seulement la première question."
    ),
}

SYSTEM_PROMPT = r"""
Tu es EduBot, un assistant pédagogique francophone.

- Aide l'élève à comprendre et à progresser.
- Adapte tes explications au niveau indiqué.
- Explique les étapes importantes.
- Demande une précision si nécessaire.
- Signale tes incertitudes et n'invente pas de faits.
- Utilise du Markdown simple.
- Écris les formules en ligne avec $...$.
- Écris les équations séparées avec $$...$$.
- Ne produis pas de Mermaid, HTML ou SVG.
- Ne prétends pas générer des images.

Photos :
- Analyse uniquement les photos jointes au message courant.
- Ne devine pas les mots ou nombres illisibles.
- Demande une photo plus nette si nécessaire.
- Les anciennes photos ne sont pas retransmises.
- Si tu dois revoir une ancienne photo, demande de la joindre à nouveau.
"""

MODE_PROMPTS = {
    "course": """
MODE : COMPRENDRE MON COURS
Présente l'idée essentielle, explique progressivement,
donne un exemple et termine par un résumé court.
Tu peux poser une question de compréhension.
Ne répète pas toutes ces rubriques pour une simple question de suivi.
""",
    "exercise": """
MODE : M'AIDER SUR UN EXERCICE
Vérifie que l'énoncé est complet et lisible.
Prends en compte la tentative de l'élève.
Commence par un indice ou une première étape.
Invite l'élève à essayer la suite.
Corrige les erreurs précisément, sans le décourager.
Ne donne toute la correction que si l'élève la demande explicitement.
""",
    "quiz": """
MODE : ME TESTER
Si le thème est absent, demande-le.
Utilise le niveau indiqué.
Propose 5 questions par défaut, 10 maximum par série.
Pose UNE question à la fois et numérote-la.
Par défaut, utilise un QCM avec les choix A, B, C et D.
Ne révèle pas la réponse avant la réponse de l'élève.
Attends sa réponse avant de continuer.
Si elle est ambiguë, demande une précision sans la noter.
Après sa réponse, indique si elle est correcte et explique pourquoi.
Pose ensuite la question suivante.
Compte seulement les réponses réellement données.
À la fin, donne le score et les notions à retravailler.
Si tu ne peux pas compter de manière fiable, dis-le.
Ne relance pas automatiquement une nouvelle série.
""",
}


# ============================================================
# 2. STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(
            ellipse at top right, #292345 0%, #0e1220 65%
        );
        color: #f1f5f9;
        color-scheme: dark;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #111626;
        border-right: 1px solid #35405a;
    }

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
    [data-testid="stWidgetLabel"] p,
    [data-testid="stRadio"] label p,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #bac7dd !important;
    }
    [data-testid="stMarkdownContainer"] a {
        color: #d3bfff !important;
    }
    .katex { color: #f1f5f9 !important; }

    [data-testid="stChatMessage"] {
        background: #192136 !important;
        border: 1px solid #3c4863;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
    }

    [data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
    }
    [data-testid="stMarkdownContainer"] th {
        background: #303958 !important;
        color: #fff !important;
        border: 1px solid #53617e !important;
        padding: 12px;
    }
    [data-testid="stMarkdownContainer"] td {
        background: #1b253b !important;
        color: #edf2fa !important;
        border: 1px solid #53617e !important;
        padding: 12px;
    }

    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="base-input"] {
        background: #222d46 !important;
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

    [data-testid="stForm"],
    [data-testid="stExpander"] {
        background: #182136;
        border: 1px solid #3c4863;
        border-radius: 14px;
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background: #0e1220 !important;
    }
    [data-testid="stChatInput"] {
        background: #222d46 !important;
        border: 1px solid #7c89ae !important;
        border-radius: 18px;
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
    }
    button:disabled { opacity: 0.5; }

    .hero {
        padding: 30px;
        border-radius: 24px;
        border: 1px solid #655484;
        background: linear-gradient(120deg, #30264f, #1c3048);
        margin-bottom: 24px;
    }
    .hero h1 {
        color: #fff !important;
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
        .hero { padding: 22px; }
        [data-testid="stChatMessage"] { padding: 14px; }
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
            Comprends tes cours, avance dans tes exercices
            et teste tes connaissances.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. GOOGLE ET IDENTITÉ ADMINISTRATEUR
# ============================================================

try:
    secrets = st.secrets.to_dict()
except FileNotFoundError:
    secrets = {}

auth = secrets.get("auth", {})
required_fields = [
    "redirect_uri",
    "cookie_secret",
    "client_id",
    "client_secret",
    "server_metadata_url",
]

if not all(auth.get(field) for field in required_fields):
    st.warning(
        "Complète la section [auth] dans les Secrets Streamlit "
        "pour activer Google."
    )
    st.stop()

if not st.user.get("is_logged_in", False):
    st.session_state.clear()

    _, center, _ = st.columns([1, 2, 1])
    with center:
        with st.container(border=True):
            st.subheader("Bienvenue dans ton espace ✨")
            st.write(
                "Connecte-toi pour retrouver tes discussions."
            )
            st.button(
                "Continuer avec Google",
                on_click=st.login,
                type="primary",
                use_container_width=True,
            )
            st.caption(
                "EduBot ne reçoit pas ton mot de passe Google."
            )
    st.stop()

google_sub = str(st.user.get("sub", "")).strip()

if not google_sub:
    st.error("L'identifiant Google est absent.")
    st.button("Se déconnecter", on_click=st.logout)
    st.stop()

display_name = st.user.get("name") or st.user.get("email") or "Élève"
email = st.user.get("email", "")

api_key = (
    os.environ.get("GROQ_API_KEY", "")
    or secrets.get("GROQ_API_KEY", "")
).strip()


# ============================================================
# 4. BASE SUPABASE / POSTGRESQL
# ============================================================

if not DATABASE_URL:
    st.error(
        "DATABASE_URL est absent. Ajoute-le dans le fichier .env "
        "ou dans les Secrets Streamlit."
    )
    st.stop()


@contextmanager
def database():
    with psycopg.connect(
        DATABASE_URL,
        row_factory=psycopg.rows.dict_row,
        autocommit=False,
    ) as connection:
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def now():
    return datetime.now(PARIS)


def today():
    return datetime.now(PARIS).date()


def get_or_create_user():
    with database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM edubot.users
                WHERE google_sub = %s
                """,
                (google_sub,),
            )
            user = cursor.fetchone()

            if user:
                cursor.execute(
                    """
                    UPDATE edubot.users
                    SET display_name = %s,
                        email = %s,
                        last_seen_at = NOW()
                    WHERE id = %s
                    RETURNING id
                    """,
                    (display_name, email, user["id"]),
                )
                return cursor.fetchone()["id"]

            user_id = str(uuid.uuid4())

            cursor.execute(
                """
                INSERT INTO edubot.users
                    (id, google_sub, display_name, email)
                VALUES
                    (%s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, google_sub, display_name, email),
            )
            return cursor.fetchone()["id"]


user_id = get_or_create_user()

if st.session_state.get("identity") != user_id:
    st.session_state.clear()
    st.session_state.identity = user_id


# ============================================================
# 5. DISCUSSIONS ET QUOTAS
# ============================================================

def get_chats():
    with database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, user_id, title, mode, created_at, updated_at
                FROM edubot.chats
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id,),
            )
            return cursor.fetchall()


def get_messages(chat_id):
    if not chat_id:
        return []

    with database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT m.id, m.chat_id, m.role, m.content, m.image,
                       m.created_at
                FROM edubot.messages m
                JOIN edubot.chats c ON c.id = m.chat_id
                WHERE m.chat_id = %s
                  AND c.user_id = %s
                ORDER BY m.id
                """,
                (chat_id, user_id),
            )
            return cursor.fetchall()


def image_count():
    with database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT image_count
                FROM edubot.daily_usage
                WHERE user_id = %s
                  AND day = %s
                """,
                (user_id, today()),
            )
            row = cursor.fetchone()

    return int(row["image_count"]) if row else 0


def save_mode(chat_id, mode):
    if mode not in MODES:
        raise ValueError("Mode invalide.")

    with database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM edubot.chats
                WHERE id = %s
                  AND user_id = %s
                """,
                (chat_id, user_id),
            )

            if not cursor.fetchone():
                raise ValueError("Discussion introuvable.")

            cursor.execute(
                """
                UPDATE edubot.chats
                SET mode = %s
                WHERE id = %s
                  AND user_id = %s
                """,
                (mode, chat_id, user_id),
            )


def save_submission(chat_id, text, images, mode):
    if mode not in MODES:
        raise ValueError("Mode invalide.")

    with database() as connection:
        with connection.cursor() as cursor:
            if chat_id:
                cursor.execute(
                    """
                    SELECT id
                    FROM edubot.chats
                    WHERE id = %s
                      AND user_id = %s
                    """,
                    (chat_id, user_id),
                )

                if not cursor.fetchone():
                    raise ValueError("Discussion introuvable.")

                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM edubot.messages
                    WHERE chat_id = %s
                      AND role = 'user'
                    """,
                    (chat_id,),
                )
                sent = cursor.fetchone()["count"]

                if sent >= CHAT_MESSAGE_LIMIT:
                    raise ValueError(
                        "Cette discussion a atteint 20 messages. "
                        "Ouvre une nouvelle discussion gratuitement."
                    )

            day = today()

            cursor.execute(
                """
                SELECT image_count
                FROM edubot.daily_usage
                WHERE user_id = %s
                  AND day = %s
                """,
                (user_id, day),
            )
            row = cursor.fetchone()

            used = int(row["image_count"]) if row else 0
            remaining = max(0, DAILY_IMAGE_LIMIT - used)

            if len(images) > remaining:
                raise ValueError(
                    f"Il te reste {remaining} image(s) aujourd'hui."
                )

            if not chat_id:
                chat_id = str(uuid.uuid4())
                title = re.sub(r"\s+", " ", text).strip()
                title = title[:57] + "…" if len(title) > 60 else title

                cursor.execute(
                    """
                    INSERT INTO edubot.chats
                        (id, user_id, title, mode)
                    VALUES
                        (%s, %s, %s, %s)
                    """,
                    (chat_id, user_id, title, mode),
                )
            else:
                cursor.execute(
                    """
                    UPDATE edubot.chats
                    SET mode = %s
                    WHERE id = %s
                      AND user_id = %s
                    """,
                    (mode, chat_id, user_id),
                )

            if images:
                cursor.execute(
                    """
                    INSERT INTO edubot.daily_usage
                        (user_id, day, image_count)
                    VALUES
                        (%s, %s, %s)
                    ON CONFLICT (user_id, day)
                    DO UPDATE SET
                        image_count =
                            edubot.daily_usage.image_count
                            + EXCLUDED.image_count
                    """,
                    (user_id, day, len(images)),
                )

            cursor.execute(
                """
                INSERT INTO edubot.messages
                    (chat_id, role, content)
                VALUES
                    (%s, 'user', %s)
                """,
                (chat_id, text),
            )

            for image in images:
                cursor.execute(
                    """
                    INSERT INTO edubot.messages
                        (chat_id, role, content, image)
                    VALUES
                        (%s, 'attachment', '', %s)
                    """,
                    (chat_id, psycopg.Binary(image)),
                )

            cursor.execute(
                """
                UPDATE edubot.chats
                SET updated_at = NOW()
                WHERE id = %s
                """,
                (chat_id,),
            )

    return chat_id


def save_reply(chat_id, reply):
    with database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM edubot.chats
                WHERE id = %s
                  AND user_id = %s
                """,
                (chat_id, user_id),
            )

            if not cursor.fetchone():
                raise ValueError("Discussion introuvable.")

            cursor.execute(
                """
                INSERT INTO edubot.messages
                    (chat_id, role, content)
                VALUES
                    (%s, 'assistant', %s)
                """,
                (chat_id, reply),
            )

            cursor.execute(
                """
                UPDATE edubot.chats
                SET updated_at = NOW()
                WHERE id = %s
                """,
                (chat_id,),
            )


def rename_chat(chat_id, new_title):
    with database() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE edubot.chats
                SET title = %s
                WHERE id = %s
                  AND user_id = %s
                """,
                (new_title, chat_id, user_id),
            )


# ============================================================
# 6. FORMATAGE ET PHOTOS
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
        r"(```.*?```|`[^`\n]*`)", text, flags=re.DOTALL
    )

    for index, part in enumerate(parts):
        if part.startswith("`"):
            continue

        part = re.sub(
            r"\\\[(.*?)\\\]",
            lambda m: "\n\n$$\n" + m.group(1).strip() + "\n$$\n\n",
            part,
            flags=re.DOTALL,
        )
        part = re.sub(
            r"\\\((.*?)\\\)",
            lambda m: "$" + m.group(1).strip() + "$",
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
                "error", Image.DecompressionBombWarning
            )

            with Image.open(io.BytesIO(raw)) as check:
                if check.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Utilise une image JPG, PNG ou WebP.")
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
            "Image invalide, endommagée ou trop grande."
        ) from error


# ============================================================
# 7. ERREURS GROQ
# ============================================================

def api_error_message(error):
    if isinstance(error, groq.AuthenticationError):
        return "La clé API Groq est invalide."
    if isinstance(error, groq.RateLimitError):
        return "La limite Groq est atteinte. Réessaie plus tard."
    if isinstance(
        error, (groq.NotFoundError, groq.PermissionDeniedError)
    ):
        return "Le modèle est inaccessible sur ton compte Groq."
    if isinstance(error, groq.APITimeoutError):
        return "Le délai de réponse est dépassé."
    if isinstance(error, groq.APIConnectionError):
        return "La connexion à Groq a échoué."
    if isinstance(error, groq.BadRequestError):
        return (
            "La demande a été refusée. Essaie une question plus courte "
            "ou une nouvelle discussion."
        )
    return "Une erreur est survenue. Consulte les journaux du serveur."


# ============================================================
# 8. DIAGNOSTIC SUPABASE
# ============================================================

def show_supabase_test():
    if not is_admin():
        st.error("Accès non autorisé.")
        return

    st.subheader("🔧 Connexion à Supabase")
    st.write(
        "Ce test vérifie la connexion PostgreSQL et les tables EduBot."
    )

    if not st.button(
        "Tester la connexion à Supabase",
        type="primary",
    ):
        return

    try:
        with st.spinner("Vérification de la connexion…"):
            with database() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()

                    cursor.execute(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'edubot'
                          AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                        """
                    )
                    tables = [
                        row["table_name"]
                        for row in cursor.fetchall()
                    ]

        if result and result["?column?"] == 1:
            st.success(
                "Connexion réussie : EduBot peut accéder à Supabase !"
            )
            st.write("Tables trouvées :")
            st.code("\n".join(tables))
        else:
            st.error("La base a renvoyé une réponse inattendue.")

    except Exception as error:
        LOGGER.error(
            "Test Supabase : %s",
            type(error).__name__,
        )
        st.error(
            "Connexion Supabase impossible. "
            "Vérifie DATABASE_URL dans .env ou les Secrets Streamlit."
        )
        st.caption(f"Type d’erreur : {type(error).__name__}")


# ============================================================
# 9. STATISTIQUES PRIVÉES
# ============================================================

def show_statistics():
    if not is_admin():
        st.error("Accès non autorisé.")
        return

    st.subheader("📊 Statistiques EduBot")
    st.caption(
        "Compteurs uniquement : aucun contenu de conversation, "
        "aucune photo et aucune liste d'e-mails."
    )

    if st.button("Actualiser les statistiques"):
        st.rerun()

    end_day = today()
    start_day = end_day - timedelta(days=29)

    try:
        with database() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM edubot.users")
                accounts = cursor.fetchone()["total"]

                cursor.execute(
                    """
                    SELECT COUNT(DISTINCT user_id) AS total
                    FROM edubot.chats
                    """
                )
                participating = cursor.fetchone()["total"]

                cursor.execute("SELECT COUNT(*) AS total FROM edubot.chats")
                chats_count = cursor.fetchone()["total"]

                cursor.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM edubot.messages
                    WHERE role = 'user'
                    """
                )
                messages_count = cursor.fetchone()["total"]

                cursor.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM edubot.messages
                    WHERE role = 'attachment'
                      AND image IS NOT NULL
                    """
                )
                photos_count = cursor.fetchone()["total"]

                cursor.execute(
                    """
                    SELECT mode, COUNT(*) AS total
                    FROM edubot.chats
                    GROUP BY mode
                    """
                )
                modes_rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT day, SUM(image_count) AS total
                    FROM edubot.daily_usage
                    WHERE day BETWEEN %s AND %s
                    GROUP BY day
                    ORDER BY day
                    """,
                    (start_day, end_day),
                )
                photo_rows = cursor.fetchall()

        a, b, c = st.columns(3)
        a.metric("Comptes enregistrés", accounts)
        b.metric("Comptes avec une discussion", participating)
        c.metric("Discussions", chats_count)

        d, e = st.columns(2)
        d.metric("Messages utilisateurs", messages_count)
        e.metric("Photos enregistrées", photos_count)

        counts = {"course": 0, "exercise": 0, "quiz": 0}
        for row in modes_rows:
            if row["mode"] in counts:
                counts[row["mode"]] += int(row["total"])

        st.subheader("Répartition des discussions")

        if chats_count:
            st.bar_chart(
                {
                    "Mode": ["Cours", "Exercices", "Quiz"],
                    "Discussions": [
                        counts["course"],
                        counts["exercise"],
                        counts["quiz"],
                    ],
                },
                x="Mode",
                y="Discussions",
            )
        else:
            st.info("Aucune discussion enregistrée.")

        per_day = {
            row["day"].isoformat(): int(row["total"])
            for row in photo_rows
        }
        days = [start_day + timedelta(days=n) for n in range(30)]

        st.subheader("Photos soumises — 30 derniers jours")
        st.bar_chart(
            {
                "Date": [day.isoformat() for day in days],
                "Photos": [
                    per_day.get(day.isoformat(), 0)
                    for day in days
                ],
            },
            x="Date",
            y="Photos",
        )

        st.info(
            "Ces chiffres concernent les données présentes dans Supabase."
        )

    except Exception as error:
        LOGGER.error("Statistiques : %s", type(error).__name__)
        st.error("Impossible de charger les statistiques.")


# ============================================================
# 10. ADMINISTRATEUR
# ============================================================

def is_admin():
    allowed = str(
        secrets.get("admin", {}).get("google_sub", "")
    ).strip()

    return (
        bool(st.user.get("is_logged_in", False))
        and bool(allowed)
        and google_sub == allowed
    )


# ============================================================
# 11. NAVIGATION
# ============================================================

chats = get_chats()

with st.sidebar:
    st.title("🎓 Mon espace")
    st.write(f"Bonjour, {display_name} 👋")

    if email:
        st.caption(email)

    with st.expander("⚙️ Mon compte"):
        st.caption("Identifiant Google de ton propre compte :")
        st.code(google_sub, language=None)

    if is_admin():
        page = st.radio(
            "Navigation",
            [
                "💬 Assistant",
                "📊 Statistiques",
                "🔧 Test Supabase",
            ],
            key="navigation",
        )
    else:
        page = "💬 Assistant"

    if page == "💬 Assistant":
        if st.button(
            "＋ Nouvelle discussion",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.pop("chat_id", None)
            st.session_state.pop("mode_new", None)
            st.rerun()

        st.divider()
        st.subheader("📷 Mes photos")

        remaining = max(0, DAILY_IMAGE_LIMIT - image_count())
        st.progress(remaining / DAILY_IMAGE_LIMIT)
        st.caption(f"{remaining} / {DAILY_IMAGE_LIMIT} images restantes")
        st.caption("Remise à zéro à minuit, heure de Paris.")

        st.divider()
        st.subheader("💬 Mes discussions")

        search = st.text_input(
            "Rechercher un thème",
            placeholder="Maths, français, histoire…",
        )

        visible_chats = [
            chat for chat in chats
            if search.casefold() in chat["title"].casefold()
        ]

        icons = {"course": "📖", "exercise": "🧩", "quiz": "🎯"}

        for chat in visible_chats:
            icon = icons.get(chat["mode"], "💬")

            if st.session_state.get("chat_id") == chat["id"]:
                icon = "📌"

            if st.button(
                f"{icon} {chat['title']}",
                key=f"open_{chat['id']}",
                use_container_width=True,
            ):
                st.session_state.chat_id = chat["id"]
                st.rerun()

        if not visible_chats:
            st.caption(
                "Aucun résultat."
                if search
                else "Tes discussions apparaîtront ici."
            )

    st.divider()

    if st.button("🚪 Me déconnecter", use_container_width=True):
        st.session_state.clear()
        st.logout()


# ============================================================
# 12. PAGES ADMINISTRATEUR
# ============================================================

if page == "🔧 Test Supabase":
    show_supabase_test()
    st.stop()

if page == "📊 Statistiques":
    show_statistics()
    st.stop()


# ============================================================
# 13. PAGE ASSISTANT
# ============================================================

chat_id = st.session_state.get("chat_id")

active_chat = next(
    (chat for chat in chats if chat["id"] == chat_id),
    None,
)

if not active_chat:
    chat_id = None
    st.session_state.pop("chat_id", None)

history = get_messages(chat_id)

notice = st.session_state.pop("notice", None)
if notice:
    st.error(notice)

st.subheader("Comment veux-tu travailler ?")

saved_mode = active_chat["mode"] if active_chat else "course"
if saved_mode not in MODES:
    saved_mode = "course"

mode_keys = list(MODES)

selected_mode = st.radio(
    "Choisis ton mode",
    options=mode_keys,
    index=mode_keys.index(saved_mode),
    format_func=lambda mode: MODES[mode],
    horizontal=True,
    key=f"mode_{chat_id or 'new'}",
)

st.info(MODE_DESCRIPTIONS[selected_mode])

if chat_id and selected_mode != saved_mode:
    save_mode(chat_id, selected_mode)
    st.rerun()

if active_chat:
    st.subheader(active_chat["title"])

    with st.expander("✏️ Modifier le thème"):
        with st.form(f"rename_{chat_id}"):
            new_title = st.text_input(
                "Titre",
                value=active_chat["title"],
                max_chars=80,
            )
            rename = st.form_submit_button("Enregistrer")

        if rename:
            if not new_title.strip():
                st.warning("Le titre ne peut pas être vide.")
            else:
                rename_chat(chat_id, new_title.strip())
                st.rerun()

else:
    welcome = {
        "course": (
            "📖 Une notion à comprendre ?",
            "Indique le sujet et ton niveau.",
        ),
        "exercise": (
            "🧩 On avance ensemble",
            "Ajoute l'énoncé et ta tentative si tu as commencé.",
        ),
        "quiz": (
            "🎯 À toi de jouer",
            "Indique le chapitre et ton niveau.",
        ),
    }
    title, description = welcome[selected_mode]
    st.markdown(f"### {title}")
    st.write(description)

if selected_mode == "quiz":
    st.caption(
        "Le score est suivi par l'IA dans la conversation : "
        "c'est un entraînement, pas une notation officielle."
    )

st.caption(
    "📎 JPG, PNG ou WebP • 5 Mo par image. "
    "Messages et photos sont enregistrés pour ton historique. "
    "Les éléments nécessaires à la réponse sont transmis à Groq."
)

for message in history:
    if message["role"] == "attachment":
        if message["image"]:
            st.image(bytes(message["image"]), width=320)
    else:
        avatar = "🎓" if message["role"] == "assistant" else "👤"

        with st.chat_message(message["role"], avatar=avatar):
            content = message["content"]

            if message["role"] == "assistant":
                content = clean_text(content)

            st.markdown(content)

sent_count = sum(
    1 for message in history
    if message["role"] == "user"
)
limit_reached = sent_count >= CHAT_MESSAGE_LIMIT

st.caption(
    f"{sent_count} / {CHAT_MESSAGE_LIMIT} messages "
    "envoyés dans cette discussion."
)

if limit_reached:
    st.info(
        "Tu as atteint les 20 messages de cette discussion. "
        "Ouvre une nouvelle discussion gratuitement. "
        "Ton historique reste consultable."
    )
    if st.button("＋ Commencer une nouvelle discussion"):
        st.session_state.pop("chat_id", None)
        st.session_state.pop("mode_new", None)
        st.rerun()

if not api_key:
    st.warning("Ajoute GROQ_API_KEY dans les Secrets Streamlit.")


# ============================================================
# 14. APPEL GROQ
# ============================================================

submission = st.chat_input(
    PLACEHOLDERS[selected_mode],
    accept_file="multiple",
    file_type=["jpg", "jpeg", "png", "webp"],
    max_upload_size=5,
    disabled=not bool(api_key) or limit_reached,
    key=f"input_{chat_id or 'new'}",
)

if submission:
    text = (submission.text or "").strip()
    files = list(submission.files or [])

    if not text and not files:
        st.stop()

    available = max(0, DAILY_IMAGE_LIMIT - image_count())

    if len(files) > DAILY_IMAGE_LIMIT or len(files) > available:
        st.error(
            f"Il te reste {available} image(s) aujourd'hui. "
            "Tu peux continuer sans photo."
        )
        st.stop()

    try:
        images = [prepare_image(file) for file in files]
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if not text:
        text = PHOTO_REQUESTS[selected_mode]

    try:
        chat_id = save_submission(
            chat_id, text, images, selected_mode
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.session_state.chat_id = chat_id

    with st.chat_message("user", avatar="👤"):
        st.markdown(text)
        for image in images:
            st.image(image, width=320)

    context_limit = 50 if selected_mode == "quiz" else 20

    previous = [
        {
            "role": message["role"],
            "content": (
                clean_text(message["content"])
                if message["role"] == "assistant"
                else message["content"]
            ),
        }
        for message in history
        if message["role"] in {"user", "assistant"}
    ][-context_limit:]

    if images:
        content = [{"type": "text", "text": text}]

        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}"
                },
            })

        model = VISION_MODEL
    else:
        content = text
        model = TEXT_MODEL

    messages_api = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\n" + MODE_PROMPTS[selected_mode],
        },
        *previous,
        {"role": "user", "content": content},
    ]

    try:
        with st.spinner("EduBot prépare sa réponse…"):
            client = Groq(
                api_key=api_key,
                timeout=90.0,
                max_retries=0,
            )
            result = client.chat.completions.create(
                model=model,
                messages=messages_api,
                temperature=0.3,
                max_completion_tokens=2500,
            )

            reply = clean_text(result.choices[0].message.content)

            if not reply:
                raise ValueError("Réponse vide.")

            save_reply(chat_id, reply)

    except Exception as error:
        LOGGER.error(
            "Erreur Groq : type=%s statut=%s",
            type(error).__name__,
            getattr(error, "status_code", None),
        )

        st.session_state.notice = (
            api_error_message(error)
            + " Les messages et photos soumis restent enregistrés "
            "et comptés dans les limites."
        )

    st.rerun()

st.divider()
st.caption(
    "EduBot Studio • Comprendre plutôt que recopier • "
    "Les réponses peuvent contenir des erreurs."
)
