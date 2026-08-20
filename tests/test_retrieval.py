import numpy as np

from style_and_sense.retrieval import (
    GarmentIndex,
    build_garment_index,
    garment_search_text,
    load_garment_index,
    metadata_search_garments,
    normalize_embedding,
    save_garment_index,
    search_garments,
)


def test_normalize_embedding_returns_unit_vector():
    vector = normalize_embedding(np.array([3.0, 4.0]))

    assert np.allclose(vector, np.array([0.6, 0.8], dtype=np.float32))


def test_save_and_load_garment_index_round_trip(tmp_path):
    ids_path = tmp_path / "ids.json"
    embeddings_path = tmp_path / "embeddings.npy"
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    save_garment_index(
        ["garment_1", "garment_2"],
        embeddings,
        ids_path=ids_path,
        embeddings_path=embeddings_path,
        faiss_path=tmp_path / "garments.faiss",
    )
    index = load_garment_index(ids_path=ids_path, embeddings_path=embeddings_path)

    assert index is not None
    assert index.garment_ids == ["garment_1", "garment_2"]
    assert np.array_equal(index.embeddings, embeddings)


def test_search_garments_uses_vector_scores():
    garments = [
        {
            "id": "garment_1",
            "category": "top",
            "subcategory": "t-shirt",
            "colors": ["white"],
            "style_tags": ["basic"],
            "season_tags": [],
            "formality": "casual",
            "caption": "white tee",
            "laundry_status": "available",
        },
        {
            "id": "garment_2",
            "category": "bottom",
            "subcategory": "jeans",
            "colors": ["denim"],
            "style_tags": ["casual"],
            "season_tags": [],
            "formality": "casual",
            "caption": "blue jeans",
            "laundry_status": "available",
        },
    ]
    index = GarmentIndex(
        garment_ids=["garment_1", "garment_2"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    results = search_garments(
        "white shirt",
        garments=garments,
        embed_text=lambda _: np.array([0.9, 0.1], dtype=np.float32),
        index=index,
        top_k=2,
    )

    assert [result.garment["id"] for result in results] == ["garment_1", "garment_2"]


def test_search_garments_filters_laundry_items():
    garments = [
        {
            "id": "garment_1",
            "category": "top",
            "subcategory": "t-shirt",
            "colors": ["white"],
            "style_tags": ["basic"],
            "season_tags": [],
            "formality": "casual",
            "caption": "white tee",
            "laundry_status": "in_laundry",
        },
        {
            "id": "garment_2",
            "category": "bottom",
            "subcategory": "jeans",
            "colors": ["denim"],
            "style_tags": ["casual"],
            "season_tags": [],
            "formality": "casual",
            "caption": "blue jeans",
            "laundry_status": "available",
        },
    ]
    index = GarmentIndex(
        garment_ids=["garment_1", "garment_2"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    results = search_garments(
        "shirt",
        garments=garments,
        embed_text=lambda _: np.array([1.0, 0.0], dtype=np.float32),
        index=index,
        top_k=2,
    )

    assert [result.garment["id"] for result in results] == ["garment_2"]


def test_metadata_search_works_without_vector_index():
    garments = [
        {
            "id": "garment_1",
            "category": "outerwear",
            "subcategory": "cardigan",
            "colors": ["cream"],
            "style_tags": ["cozy"],
            "season_tags": ["cool weather"],
            "formality": "casual",
            "caption": "cream cardigan",
            "laundry_status": "available",
        }
    ]

    results = metadata_search_garments(
        "cool weather cozy cardigan",
        garments=garments,
    )

    assert results[0].garment["id"] == "garment_1"


def test_garment_search_text_includes_tags():
    text = garment_search_text(
        {
            "category": "top",
            "subcategory": "sweater",
            "colors": ["cream"],
            "style_tags": ["cozy"],
            "season_tags": ["winter"],
            "formality": "casual",
            "caption": "cream sweater",
        }
    )

    assert "cream" in text
    assert "cozy" in text
    assert "sweater" in text
