from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    chroma_persist_dir: str = "./chroma_db"
    data_processed_dir: str = "./data/processed"

    text_embedding_model: str = "BAAI/bge-small-en-v1.5"
    image_embedding_model: str = "clip-ViT-B-32"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    claude_model: str = "claude-sonnet-4-6"

    chunk_size: int = 800          # characters
    chunk_overlap: int = 120       # characters

    top_k_bm25: int = 20
    top_k_vector: int = 20
    top_k_reranked: int = 5
    top_m_images: int = 2

settings = Settings()

def processed_dir() -> Path:
    p = Path(settings.data_processed_dir)
    (p / "images").mkdir(parents=True, exist_ok=True)
    return p