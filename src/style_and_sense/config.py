from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEXES_DIR = DATA_DIR / "indexes"
STYLE_RULES_DIR = DATA_DIR / "style_rules"
DB_PATH = DATA_DIR / "app.db"

