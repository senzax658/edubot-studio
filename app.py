import base64
import io
import logging
import os
import re
import sqlite3
import uuid
import warnings

from contextlib import contextmanager
from datetime import date, datetime, timedelta
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
DB_PATH = DATA_DIR / "edubot.sqlite3"

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
    "course": (
        "Comprends une notion avec une explication claire, "
        "un exemple et les points essentiels."
    ),
    "exercise": (
        "Avance avec des indices progressifs. "
        "Demande la correction complète quand tu en as besoin."
    ),
    "quiz": (
        "Réponds à une question à la fois, "
        "puis découvre une correction expliquée."
    ),
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
        "Aide-moi à comprendre cet exercice. "
        "Commence par un indice sans donner toute la correction."
    ),
    "quiz": (
        "Crée un quiz de 5 questions à partir du contenu lisible "
        "sur ces photos. Pose seulement la première question."
    ),
}

SYSTEM_PROMPT = r"""
Tu es EduBot, un assistant pédagogique francophone.

- Aide l'élève à comprendre et à progresser.
- Adapte les explications au niveau indiqué.
- Explique les étapes importantes.
- Demande une précision si nécessaire.
- Signale tes incertitudes et n'invente pas de faits.
- Utilise du Markdown simple.
- Pour les formules en ligne, utilise $...$.
- Pour les équations séparées, utilise $$...$$.
- Ne produis pas de Mermaid, HTML ou SVG.
- Ne prétends pas générer des images.

Photos :
- Tu peux analyser les photos jointes au message courant.
- Ne devine pas les mots ou nombres illisibles.
- Demande une photo plus nette si nécessaire.
- Les anciennes photos ne sont pas retransmises.
- Si tu dois revoir une ancienne photo, demande de la joindre à nouveau.
"""

MODE_PROMPTS = {
    "course": """
MODE : COMPRENDRE MON COURS

Pour une nouvelle explication :
1. Présente simplement l'idée essentielle.
2. Explique progressivement les notions importantes.
3. Donne un exemple concret ou résolu.
4. Signale une erreur fréquente si utile.
5. Termine par un résumé court.

Tu peux poser une question de compréhension.
Ne répète pas toutes les rubriques pour une simple question de suivi.
""",
    "exercise": """
MODE : M'AIDER SUR UN EXERCICE

- Vérifie que l'énoncé est complet et lisible.
- Prends en compte la tentative de l'élève.
- Commence par un indice ou une première étape.
- Invite l'élève à essayer la suite.
- Corrige les erreurs précisément et sans le décourager.
- Ne donne pas immédiatement toute la solution.
- Si l'élève demande explicitement la correction complète,
  donne-la avec les étapes.
""",
    "quiz": """
MODE : ME TESTER

- Si le thème est absent, demande-le.
- Utilise le niveau indiqué.
- Propose 5 questions par défaut, 10 maximum par série.
- Pose UNE question à la fois.
- Numérote les questions : Question 1/5, etc.
- Par défaut, propose un QCM avec les choix A, B, C et D.
- Ne révèle pas la bonne réponse avant la réponse de l'élève.
- Attends sa réponse.
- Si elle est ambiguë, demande une précision sans la noter.
- Après sa réponse, indique si elle est correcte et explique pourquoi.
- Pose ensuite la question suivante.
- Compte seulement les réponses réellement données.
- Si l'élève veut passer, explique la réponse et compte la question
  comme non réussie.
- À la fin, donne le score et les notions à retravailler.
- Ne recommence pas automatiquement.
- Si le contexte ne permet pas un score fiable, dis-le.
""",
}


# ============================================================
# 2. INTERFACE ET COULEURS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(
            ellipse at top right, #292345 0%, #0e1220 65%
        );
        color: #f1f5f9;
