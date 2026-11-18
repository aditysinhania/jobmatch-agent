import os

from dotenv import load_dotenv

load_dotenv()


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", "postgresql+psycopg://jobs:jobs@localhost:5432/jobs")
    # Managed providers (Neon, Supabase, Render) hand out postgres:// or postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


class Settings:
    database_url: str = _db_url()

    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    embed_cache: str = os.getenv("EMBED_CACHE", "/tmp/fastembed")

    admin_token: str = os.getenv("ADMIN_TOKEN", "")
    auto_ingest: bool = os.getenv("AUTO_INGEST", "1") == "1"
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "30"))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "6"))


settings = Settings()
