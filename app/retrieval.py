"""Vector, keyword and hybrid (reciprocal rank fusion) retrieval over the jobs table."""
import re

from sqlalchemy import text

from .db import engine
from .embeddings import embed_one, to_pgvector

COLS = (
    "id, title, company, category, tags, job_type, location, salary, url, "
    "left(coalesce(description, ''), 400) AS snippet"
)
CANDIDATES = 50  # candidates pulled from each retriever before fusion
RRF_K = 60       # standard RRF damping constant

_STOP = {
    "a", "an", "and", "or", "the", "for", "with", "in", "on", "at", "to", "of", "job", "jobs",
    "role", "roles", "position", "positions", "looking", "want", "need", "someone", "who", "can",
    "that", "is", "are", "me", "my", "i", "work", "working", "build", "building",
}


def build_tsquery(query: str) -> str:
    """Natural-language query -> OR-joined tsquery string (empty string if nothing usable)."""
    seen: list[str] = []
    for tok in re.findall(r"[a-z0-9]+", query.lower()):
        if len(tok) > 1 and tok not in _STOP and tok not in seen:
            seen.append(tok)
    return " | ".join(seen[:12])


def _norm_job_type(job_type: str | None) -> str | None:
    return re.sub(r"[\s-]+", "_", job_type.strip().lower()) if job_type and job_type.strip() else None


def _where(location: str | None, job_type: str | None, params: dict) -> str:
    clauses = ["active = TRUE"]
    if location and location.strip():
        clauses.append("location ILIKE :loc")
        params["loc"] = f"%{location.strip()}%"
    jt = _norm_job_type(job_type)
    if jt:
        clauses.append("lower(job_type) = :jt")
        params["jt"] = jt
    return " AND ".join(clauses)


def vector_search(query: str, k: int = 10, location: str | None = None, job_type: str | None = None) -> list[dict]:
    params: dict = {"emb": to_pgvector(embed_one(query)), "n": k}
    where = _where(location, job_type, params)
    sql = text(
        f"""
        SELECT {COLS}, 1 - (embedding <=> CAST(:emb AS vector)) AS score
        FROM jobs
        WHERE {where} AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:emb AS vector)
        LIMIT :n
        """
    )
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).mappings()]


def keyword_search(query: str, k: int = 10, location: str | None = None, job_type: str | None = None) -> list[dict]:
    tsq = build_tsquery(query)
    if not tsq:
        return []
    params: dict = {"tsq": tsq, "n": k}
    where = _where(location, job_type, params)
    sql = text(
        f"""
        SELECT {COLS}, ts_rank_cd(tsv, q) AS score
        FROM jobs, to_tsquery('english', :tsq) AS q
        WHERE {where} AND tsv @@ q
        ORDER BY score DESC
        LIMIT :n
        """
    )
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).mappings()]


def rrf(rankings: list[list[dict]], top_k: int = 10, k_const: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    docs: dict[int, dict] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0.0) + 1.0 / (k_const + rank)
            docs.setdefault(doc["id"], doc)
    fused = sorted(docs.values(), key=lambda d: scores[d["id"]], reverse=True)[:top_k]
    return [{**d, "score": round(scores[d["id"]], 5)} for d in fused]


def search(
    query: str,
    mode: str = "hybrid",
    k: int = 10,
    location: str | None = None,
    job_type: str | None = None,
) -> list[dict]:
    if mode == "vector":
        return vector_search(query, k, location, job_type)
    if mode == "keyword":
        return keyword_search(query, k, location, job_type)
    if mode == "hybrid":
        vec = vector_search(query, CANDIDATES, location, job_type)
        kw = keyword_search(query, CANDIDATES, location, job_type)
        return rrf([vec, kw], top_k=k)
    raise ValueError(f"unknown mode: {mode}")


def get_job(job_id: int) -> dict | None:
    sql = text(
        """
        SELECT id, title, company, category, tags, job_type, location, salary, url,
               description, published_at, active
        FROM jobs WHERE id = :id
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"id": job_id}).mappings().first()
    return dict(row) if row else None


def get_jobs(ids: list[int]) -> list[dict]:
    """Card-sized records for the given ids, in the order given."""
    if not ids:
        return []
    sql = text(f"SELECT {COLS} FROM jobs WHERE id = ANY(:ids)")
    with engine.connect() as conn:
        by_id = {r["id"]: dict(r) for r in conn.execute(sql, {"ids": list(ids)}).mappings()}
    return [by_id[i] for i in ids if i in by_id]


def job_count(active_only: bool = True) -> int:
    sql = "SELECT count(*) FROM jobs WHERE active = TRUE" if active_only else "SELECT count(*) FROM jobs"
    with engine.connect() as conn:
        return int(conn.execute(text(sql)).scalar() or 0)
