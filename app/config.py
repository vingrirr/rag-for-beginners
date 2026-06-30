from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "llama3.1:8b"

    ghost_url: str = ""
    ghost_admin_api_key: str = ""

    chunk_size: int = 200
    chunk_overlap: int = 50
    top_k: int = 5

    retrieve_k: int = 20
    use_hybrid: bool = True
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
