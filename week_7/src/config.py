"""Central configuration for the RAG system.

Everything tunable lives here (or can be overridden via environment
variables in a ``.env`` file) so that experiments are reproducible and the
pipeline is *config-driven* rather than hard-coded. The ablation studies in
the analysis notebook simply sweep the ``*_TO_COMPARE`` lists below.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (if present) exactly once, on import.
load_dotenv()

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Make sure the directories we write to exist.
for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Secrets -------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def require_api_key() -> str:
    """Return the Gemini API key or raise a friendly error if it is missing."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_gemini_api_key_here":
        raise RuntimeError(
            "GOOGLE_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Paste your free key from https://aistudio.google.com/app/apikey\n"
        )
    return GOOGLE_API_KEY


# --- Reproducibility -----------------------------------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# --- Generation (Gemini) -------------------------------------------------
# Model selection (verified against this key's free tier, 2026-06):
#   - gemini-1.5-*            : retired (404)
#   - gemini-2.0-flash/-lite  : no free quota on this key (429, limit 0)
#   - gemini-2.5-flash / flash-latest : free, but are "thinking" models that can
#                              return empty text unless max_output_tokens is large
#   - gemini-2.5-flash-lite   : free, fast, returns text reliably -> our default
# Override anytime via .env, e.g. GEMINI_MODEL=gemini-2.5-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GENERATION_TEMPERATURE = 0.2          # low -> more grounded / less creative
GENERATION_MAX_TOKENS = 1024
# Free-tier requests-per-minute cap. The limiter in src/ratelimit.py throttles
# ALL Gemini calls (answers + LLM judge) to stay under quota.
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "15"))

# --- Embeddings ----------------------------------------------------------
# Default model plus the roster benchmarked in the embedding ablation.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_MODELS_TO_COMPARE = [
    "sentence-transformers/all-MiniLM-L6-v2",   # 384-dim, fast baseline
    "BAAI/bge-small-en-v1.5",                   # 384-dim, stronger retrieval
    # "models/text-embedding-004",              # Gemini embeddings (API) — optional
]

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 512                       # ~words per chunk
CHUNK_OVERLAP = 64
CHUNK_STRATEGY = "recursive"           # "fixed" | "recursive"
CHUNK_SIZES_TO_COMPARE = [128, 256, 512, 1024]

# --- Retrieval -----------------------------------------------------------
TOP_K = 5
TOP_K_VALUES_TO_COMPARE = [1, 3, 5, 10]
HYBRID_ALPHA = 0.5                     # 1.0 = pure dense, 0.0 = pure BM25
HYBRID_ALPHAS_TO_COMPARE = [0.0, 0.25, 0.5, 0.75, 1.0]
RERANK_CANDIDATES = 20                 # base pool size fed to the cross-encoder
RERANK_TOP_N = 5                       # chunks kept after cross-encoder rerank
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_REWRITE_N = 3                    # number of query reformulations (multi-query)

# --- Evaluation ----------------------------------------------------------
# Fixed, seeded subsample keeps us inside the Gemini free-tier rate limits
# (~15 req/min, ~1500 req/day) while remaining statistically meaningful.
EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "120"))

# --- Dataset -------------------------------------------------------------
HF_DATASET_REPO = "vectara/open_ragbench"
