from sqlalchemy import create_engine, text

from .config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)

DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    f"""
    CREATE TABLE IF NOT EXISTS jobs (
        id            BIGINT PRIMARY KEY,
        source        TEXT NOT NULL DEFAULT 'remotive',
        title         TEXT NOT NULL,
        company       TEXT,
        category      TEXT,
        tags          TEXT DEFAULT '',
        job_type      TEXT,
        location      TEXT,
        salary        TEXT,
        url           TEXT,
        description   TEXT,
        published_at  TIMESTAMPTZ,
        content_hash  TEXT,
        embedding     vector({settings.embed_dim}),
        active        BOOLEAN NOT NULL DEFAULT TRUE,
        last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        tsv           tsvector GENERATED ALWAYS AS (
            to_tsvector('english',
                coalesce(title, '') || ' ' || coalesce(company, '') || ' ' ||
                coalesce(category, '') || ' ' || coalesce(tags, '') || ' ' ||
                coalesce(description, ''))
        ) STORED
    )
    """,
    "CREATE INDEX IF NOT EXISTS jobs_tsv_idx ON jobs USING gin (tsv)",
    "CREATE INDEX IF NOT EXISTS jobs_embedding_idx ON jobs USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS jobs_active_idx ON jobs (active)",
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id          BIGSERIAL PRIMARY KEY,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, id)",
    """
    CREATE TABLE IF NOT EXISTS session_profiles (
        session_id  TEXT PRIMARY KEY,
        summary     TEXT NOT NULL DEFAULT '',
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS saved_jobs (
        session_id  TEXT NOT NULL,
        job_id      BIGINT NOT NULL,
        saved_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (session_id, job_id)
    )
    """,
]


def init_db() -> None:
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
