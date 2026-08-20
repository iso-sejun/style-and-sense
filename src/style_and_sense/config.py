from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEXES_DIR = DATA_DIR / "indexes"
MODELS_DIR = DATA_DIR / "models"
STYLE_RULES_DIR = DATA_DIR / "style_rules"
DB_PATH = DATA_DIR / "app.db"
GARMENT_IDS_PATH = INDEXES_DIR / "garment_ids.json"
GARMENT_EMBEDDINGS_PATH = INDEXES_DIR / "garment_embeddings.npy"
GARMENT_FAISS_PATH = INDEXES_DIR / "garments.faiss"
