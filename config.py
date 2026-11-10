from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
STORAGE_DIR = PROJECT_DIR / "storage"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4

