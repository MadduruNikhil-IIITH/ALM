from pathlib import Path

# ────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
TEXTBOOKS_DIR = RAW_DATA_DIR / "textbooks"
TEXTBOOKS_DB_DIR = VECTOR_DB_DIR / "textbooks"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ────────────────────────────────────────────────────────────────
# RAG CONFIG
# ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RAG_TOP_K = 3

# ────────────────────────────────────────────────────────────────
# SFT CONFIG
# ────────────────────────────────────────────────────────────────
VISION_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
LOGIC_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
VISION_PAIRS_FILE = PROCESSED_DATA_DIR / "ai2d_vision_sft_pairs.json"
SCIENCEQA_TRAIN = PROCESSED_DATA_DIR / "scienceqa_physics_train.json"
SCIENCEQA_VAL = PROCESSED_DATA_DIR / "scienceqa_physics_val.json"

# ────────────────────────────────────────────────────────────────
# SIMULATION CONFIG
# ────────────────────────────────────────────────────────────────
SCENE_TEMPLATE_PATH = PROJECT_ROOT / "src" / "simulation" / "templates" / "scene_template.html"
