import sqlite3

from style_and_sense.storage import init_db


def table_names(db_path):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "app.db"

    init_db(db_path)

    assert table_names(db_path) >= {
        "garments",
        "garment_embeddings",
        "recommendation_requests",
        "outfits",
        "outfit_feedback",
    }

