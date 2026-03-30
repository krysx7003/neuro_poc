"""Konfiguracja aplikacji UODO RAG — stałe i zmienne środowiskowe."""

import os
from pathlib import Path

# Wczytaj .env z katalogu tego pliku (niezależnie od CWD przy uruchamianiu)
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent / "../.env"
    _ = load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)
except ImportError:
    pass

# ── Qdrant ────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "uodo_decisions"
GRAPH_PATH = os.getenv("UODO_GRAPH_PATH", "./uodo_graph.pkl")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sdadas/mmlw-retrieval-roberta-large")
