import sqlite3
from pathlib import Path

from style_and_sense.config import DB_PATH, INDEXES_DIR, STYLE_RULES_DIR, UPLOADS_DIR


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS garments (
    id TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    original_filename TEXT,
    category TEXT NOT NULL,
    subcategory TEXT,
    colors TEXT NOT NULL,
    style_tags TEXT,
    season_tags TEXT,
    formality TEXT,
    caption TEXT,
    laundry_status TEXT NOT NULL DEFAULT 'available',
    favorite INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS garment_embeddings (
    garment_id TEXT PRIMARY KEY,
    faiss_index INTEGER NOT NULL UNIQUE,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (garment_id) REFERENCES garments(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendation_requests (
    id TEXT PRIMARY KEY,
    user_prompt TEXT NOT NULL,
    weather_text TEXT,
    occasion TEXT,
    raw_context_json TEXT,
    retrieved_garment_ids TEXT,
    retrieved_rule_ids TEXT,
    latency_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outfits (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    title TEXT NOT NULL,
    item_ids TEXT NOT NULL,
    explanation TEXT NOT NULL,
    score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES recommendation_requests(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS outfit_feedback (
    id TEXT PRIMARY KEY,
    outfit_id TEXT NOT NULL,
    saved INTEGER NOT NULL DEFAULT 0,
    liked INTEGER,
    rating INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (outfit_id) REFERENCES outfits(id) ON DELETE CASCADE
);
"""


def init_storage() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_RULES_DIR.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    init_storage()
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)

