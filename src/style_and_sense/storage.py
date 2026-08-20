import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from style_and_sense.config import (
    DB_PATH,
    INDEXES_DIR,
    MODELS_DIR,
    STYLE_RULES_DIR,
    UPLOADS_DIR,
)


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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_list(values: list[str] | None) -> str:
    return json.dumps(values or [])


def decode_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def row_to_garment(row: sqlite3.Row) -> dict:
    garment = dict(row)
    garment["colors"] = decode_list(garment.get("colors"))
    garment["style_tags"] = decode_list(garment.get("style_tags"))
    garment["season_tags"] = decode_list(garment.get("season_tags"))
    garment["favorite"] = bool(garment.get("favorite"))
    return garment


def list_garments(db_path: Path = DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM garments
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [row_to_garment(row) for row in rows]


def get_garment(garment_id: str, db_path: Path = DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM garments WHERE id = ?",
            (garment_id,),
        ).fetchone()
    return row_to_garment(row) if row else None


def create_garment(
    *,
    image_path: str,
    original_filename: str,
    category: str,
    subcategory: str | None,
    colors: list[str],
    style_tags: list[str],
    season_tags: list[str],
    formality: str | None,
    caption: str | None,
    laundry_status: str = "available",
    favorite: bool = False,
    db_path: Path = DB_PATH,
) -> str:
    garment_id = f"garment_{uuid.uuid4().hex[:12]}"
    timestamp = now_iso()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO garments (
                id,
                image_path,
                original_filename,
                category,
                subcategory,
                colors,
                style_tags,
                season_tags,
                formality,
                caption,
                laundry_status,
                favorite,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                garment_id,
                image_path,
                original_filename,
                category,
                subcategory,
                encode_list(colors),
                encode_list(style_tags),
                encode_list(season_tags),
                formality,
                caption,
                laundry_status,
                int(favorite),
                timestamp,
                timestamp,
            ),
        )
    return garment_id


def update_garment(
    garment_id: str,
    *,
    category: str,
    subcategory: str | None,
    colors: list[str],
    style_tags: list[str],
    season_tags: list[str],
    formality: str | None,
    caption: str | None,
    laundry_status: str,
    favorite: bool,
    db_path: Path = DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE garments
            SET category = ?,
                subcategory = ?,
                colors = ?,
                style_tags = ?,
                season_tags = ?,
                formality = ?,
                caption = ?,
                laundry_status = ?,
                favorite = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                category,
                subcategory,
                encode_list(colors),
                encode_list(style_tags),
                encode_list(season_tags),
                formality,
                caption,
                laundry_status,
                int(favorite),
                now_iso(),
                garment_id,
            ),
        )


def delete_garment(garment_id: str, db_path: Path = DB_PATH) -> None:
    garment = get_garment(garment_id, db_path=db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM garments WHERE id = ?", (garment_id,))

    if garment:
        image_path = Path(garment["image_path"])
        if image_path.exists() and UPLOADS_DIR in image_path.parents:
            image_path.unlink()


def save_uploaded_image_bytes(
    image_bytes: bytes,
    *,
    original_filename: str,
    uploads_dir: Path = UPLOADS_DIR,
) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(original_filename).suffix.lower() or ".jpg"
    image_path = uploads_dir / f"garment_{uuid.uuid4().hex[:12]}{extension}"
    image_path.write_bytes(image_bytes)
    return image_path


def save_uploaded_image(
    source_path: Path,
    *,
    original_filename: str,
    uploads_dir: Path = UPLOADS_DIR,
) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(original_filename).suffix.lower() or ".jpg"
    image_path = uploads_dir / f"garment_{uuid.uuid4().hex[:12]}{extension}"
    shutil.copyfile(source_path, image_path)
    return image_path
