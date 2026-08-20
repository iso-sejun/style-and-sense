import sqlite3

from style_and_sense.metadata import (
    category_for_subcategory,
    fallback_metadata_suggestion,
    nearest_palette_color,
)
from style_and_sense.storage import (
    create_garment,
    get_garment,
    init_db,
    list_garments,
    save_uploaded_image_bytes,
    update_garment,
)


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


def test_create_and_update_garment_round_trip(tmp_path):
    db_path = tmp_path / "app.db"
    init_db(db_path)

    garment_id = create_garment(
        image_path=str(tmp_path / "shirt.jpg"),
        original_filename="shirt.jpg",
        category="top",
        subcategory="t-shirt",
        colors=["white"],
        style_tags=["casual", "basic"],
        season_tags=["summer"],
        formality="casual",
        caption="white tee",
        db_path=db_path,
    )

    garment = get_garment(garment_id, db_path=db_path)

    assert garment is not None
    assert garment["colors"] == ["white"]
    assert garment["style_tags"] == ["casual", "basic"]
    assert garment["laundry_status"] == "available"
    assert list_garments(db_path=db_path)[0]["id"] == garment_id

    update_garment(
        garment_id,
        category="top",
        subcategory="button-down",
        colors=["blue"],
        style_tags=["preppy"],
        season_tags=["spring", "fall"],
        formality="smart casual",
        caption="blue button-down",
        laundry_status="in_laundry",
        favorite=True,
        db_path=db_path,
    )

    updated = get_garment(garment_id, db_path=db_path)

    assert updated["subcategory"] == "button-down"
    assert updated["colors"] == ["blue"]
    assert updated["laundry_status"] == "in_laundry"
    assert updated["favorite"] is True


def test_save_uploaded_image_bytes_uses_upload_directory(tmp_path):
    image_path = save_uploaded_image_bytes(
        b"fake image bytes",
        original_filename="shirt.png",
        uploads_dir=tmp_path,
    )

    assert image_path.parent == tmp_path
    assert image_path.suffix == ".png"
    assert image_path.read_bytes() == b"fake image bytes"


def test_palette_color_mapping():
    assert nearest_palette_color((245, 245, 245)) == "white"
    assert nearest_palette_color((15, 20, 30)) == "black"
    assert nearest_palette_color((70, 105, 150)) == "denim"


def test_fallback_metadata_uses_filename_and_color(tmp_path):
    from PIL import Image

    image_path = tmp_path / "blue_jeans.png"
    Image.new("RGB", (20, 20), (70, 105, 150)).save(image_path)

    suggestion = fallback_metadata_suggestion(
        image_path.read_bytes(),
        filename=image_path.name,
    )

    assert suggestion.subcategory == "jeans"
    assert suggestion.category == "bottom"
    assert "denim" in suggestion.colors


def test_category_for_subcategory():
    assert category_for_subcategory("sneakers") == "shoes"
    assert category_for_subcategory("cardigan") == "outerwear"
