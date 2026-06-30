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

    # Multi-query expansion: LLM rewrites the question into N paraphrases,
    # we retrieve for each, fuse with RRF.
    use_multi_query: bool = False
    multi_query_count: int = 3

    # Cosine-similarity floor for vector candidates. 0 ≈ off (allows any
    # non-anti-aligned chunk through). 0.3 is a sane threshold for filtering
    # obvious garbage when running pure vector / vector-then-fuse.
    min_score: float = 0.0

    # Maximum Marginal Relevance: trade pure relevance for diversity across
    # the final top-k. lambda=1 → pure relevance (off-ish), 0 → max diversity.
    use_mmr: bool = False
    mmr_lambda: float = 0.5
