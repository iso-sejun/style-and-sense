from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from style_and_sense.config import (
    GARMENT_EMBEDDINGS_PATH,
    GARMENT_FAISS_PATH,
    GARMENT_IDS_PATH,
)
from style_and_sense.metadata import FASHION_CLIP_MODEL, FashionClipTagger
from style_and_sense.storage import (
    list_garments,
    replace_garment_embedding_records,
)


EmbedImageFn = Callable[[bytes], np.ndarray]
EmbedTextFn = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class RetrievalResult:
    garment: dict
    score: float


@dataclass(frozen=True)
class GarmentIndex:
    garment_ids: list[str]
    embeddings: np.ndarray


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def save_garment_index(
    garment_ids: list[str],
    embeddings: np.ndarray,
    *,
    ids_path: Path = GARMENT_IDS_PATH,
    embeddings_path: Path = GARMENT_EMBEDDINGS_PATH,
    faiss_path: Path = GARMENT_FAISS_PATH,
) -> None:
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids_path.write_text(json.dumps(garment_ids, indent=2), encoding="utf-8")
    np.save(embeddings_path, embeddings)
    maybe_save_faiss_index(embeddings, faiss_path=faiss_path)


def load_garment_index(
    *,
    ids_path: Path = GARMENT_IDS_PATH,
    embeddings_path: Path = GARMENT_EMBEDDINGS_PATH,
) -> GarmentIndex | None:
    if not ids_path.exists() or not embeddings_path.exists():
        return None
    garment_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path).astype(np.float32)
    if len(garment_ids) != len(embeddings):
        return None
    return GarmentIndex(garment_ids=garment_ids, embeddings=embeddings)


def maybe_save_faiss_index(
    embeddings: np.ndarray,
    *,
    faiss_path: Path = GARMENT_FAISS_PATH,
) -> None:
    try:
        import faiss
    except ImportError:
        return

    if embeddings.size == 0:
        return

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(faiss_path))


def build_garment_index(
    *,
    garments: list[dict] | None = None,
    embed_image: EmbedImageFn | None = None,
    embedding_model: str = FASHION_CLIP_MODEL,
) -> GarmentIndex:
    garments = garments if garments is not None else list_garments()
    tagger = None
    if embed_image is None:
        tagger = FashionClipTagger(embedding_model)
        embed_image = tagger.encode_image

    indexed_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    for garment in garments:
        image_path = Path(garment["image_path"])
        if not image_path.exists():
            continue
        embedding = normalize_embedding(embed_image(image_path.read_bytes()))
        indexed_ids.append(garment["id"])
        embeddings.append(embedding)

    matrix = (
        np.vstack(embeddings).astype(np.float32)
        if embeddings
        else np.empty((0, 0), dtype=np.float32)
    )
    save_garment_index(indexed_ids, matrix)
    if matrix.size:
        replace_garment_embedding_records(
            indexed_ids,
            embedding_model=embedding_model,
            embedding_dim=matrix.shape[1],
        )
    return GarmentIndex(garment_ids=indexed_ids, embeddings=matrix)


def search_garments(
    query: str,
    *,
    garments: list[dict] | None = None,
    embed_text: EmbedTextFn | None = None,
    index: GarmentIndex | None = None,
    top_k: int = 12,
) -> list[RetrievalResult]:
    garments = garments if garments is not None else list_garments()
    garment_by_id = {garment["id"]: garment for garment in garments}
    index = index if index is not None else load_garment_index()
    if index is None or index.embeddings.size == 0:
        return metadata_search_garments(query, garments=garments, top_k=top_k)

    if embed_text is None:
        embed_text = FashionClipTagger().encode_text
    query_embedding = normalize_embedding(embed_text(query))
    scores = index.embeddings @ query_embedding
    ranked_indexes = np.argsort(-scores)[:top_k]

    results: list[RetrievalResult] = []
    for row_index in ranked_indexes:
        garment_id = index.garment_ids[int(row_index)]
        garment = garment_by_id.get(garment_id)
        if garment and garment.get("laundry_status") == "available":
            results.append(
                RetrievalResult(
                    garment=garment,
                    score=float(scores[int(row_index)]),
                )
            )
    return results


def metadata_search_garments(
    query: str,
    *,
    garments: list[dict],
    top_k: int = 12,
) -> list[RetrievalResult]:
    terms = [term for term in query.lower().replace(",", " ").split() if term]
    scored: list[RetrievalResult] = []
    for garment in garments:
        if garment.get("laundry_status") != "available":
            continue
        text = garment_search_text(garment)
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append(RetrievalResult(garment=garment, score=float(score)))

    scored.sort(key=lambda result: (-result.score, result.garment["id"]))
    return scored[:top_k]


def garment_search_text(garment: dict) -> str:
    parts = [
        garment.get("category") or "",
        garment.get("subcategory") or "",
        garment.get("formality") or "",
        garment.get("caption") or "",
        " ".join(garment.get("colors") or []),
        " ".join(garment.get("style_tags") or []),
        " ".join(garment.get("season_tags") or []),
    ]
    return " ".join(parts).lower()

