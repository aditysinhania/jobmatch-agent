"""Incremental ingestion of Remotive jobs.

Per run:
  * new or changed postings (by content hash) are embedded and upserted
  * unchanged postings are only touched (last_seen_at) so they are not re-embedded
  * postings missing from the latest full fetch are marked inactive (stale-job removal)

Usage:
  python -m app.ingest                       # fetch from the Remotive API
  python -m app.ingest --save-raw data/r.json
  python -m app.ingest --file data/r.json    # reuse a saved copy (Remotive asks for few API calls per day)
"""
import argparse
import hashlib
import html
import json
import logging
import re
import threading
import time

import httpx
from sqlalchemy import text

from .db import engine, init_db
from .embeddings import embed, to_pgvector

log = logging.getLogger(__name__)

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
BATCH = 64

_run_lock = threading.Lock()
_state: dict = {"running": False, "last": None}


def clean_html(raw: str | None) -> str:
    s = re.sub(r"<(script|style).*?</\1>", " ", raw or "", flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def normalize(raw: dict) -> dict | None:
    title = (raw.get("title") or "").strip()
    if raw.get("id") is None or not title:
        return None
    tags = ", ".join(raw.get("tags") or [])
    desc = clean_html(raw.get("description"))
    company = raw.get("company_name")
    row = {
        "id": int(raw["id"]),
        "title": title,
        "company": company,
        "category": raw.get("category"),
        "tags": tags,
        "job_type": (raw.get("job_type") or "").lower() or None,
        "location": raw.get("candidate_required_location"),
        "salary": raw.get("salary") or None,
        "url": raw.get("url"),
        "description": desc,
        "published_at": raw.get("publication_date") or None,
    }
    row["content_hash"] = hashlib.sha256("|".join([title, company or "", tags, desc]).encode()).hexdigest()
    return row


def embed_text(r: dict) -> str:
    # Title first: MiniLM truncates around 256 word pieces, so the most informative part goes up front.
    return f"{r['title']}. {r['company'] or ''}. {r['category'] or ''}. {r['tags']}. {r['description'][:1500]}"


def fetch_remotive(limit: int | None = None) -> list[dict]:
    params = {"limit": limit} if limit else {}
    resp = httpx.get(REMOTIVE_URL, params=params, timeout=30, headers={"User-Agent": "jobmatch-agent/0.1"})
    resp.raise_for_status()
    return resp.json().get("jobs", [])


UPSERT = text(
    """
    INSERT INTO jobs (id, source, title, company, category, tags, job_type, location, salary, url,
                      description, published_at, content_hash, embedding, active, last_seen_at)
    VALUES (:id, 'remotive', :title, :company, :category, :tags, :job_type, :location, :salary, :url,
            :description, CAST(:published_at AS timestamptz), :content_hash,
            CAST(:embedding AS vector), TRUE, now())
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title, company = EXCLUDED.company, category = EXCLUDED.category,
        tags = EXCLUDED.tags, job_type = EXCLUDED.job_type, location = EXCLUDED.location,
        salary = EXCLUDED.salary, url = EXCLUDED.url, description = EXCLUDED.description,
        published_at = EXCLUDED.published_at, content_hash = EXCLUDED.content_hash,
        embedding = EXCLUDED.embedding, active = TRUE, last_seen_at = now()
    """
)


def run(limit: int | None = None, file: str | None = None, save_raw: str | None = None) -> dict:
    if not _run_lock.acquire(blocking=False):
        return {"status": "already_running"}
    _state["running"] = True
    started = time.time()
    try:
        init_db()
        if file:
            with open(file, encoding="utf-8") as f:
                raw_jobs = json.load(f)
            raw_jobs = raw_jobs.get("jobs", raw_jobs) if isinstance(raw_jobs, dict) else raw_jobs
        else:
            raw_jobs = fetch_remotive(limit)
            if save_raw:
                with open(save_raw, "w", encoding="utf-8") as f:
                    json.dump({"jobs": raw_jobs}, f)

        rows = [r for r in (normalize(j) for j in raw_jobs) if r]
        if not rows:
            raise RuntimeError("source returned no usable jobs; refusing to deactivate the index")

        with engine.connect() as conn:
            existing = dict(conn.execute(text("SELECT id, content_hash FROM jobs WHERE source = 'remotive'")).all())

        changed = [r for r in rows if existing.get(r["id"]) != r["content_hash"]]
        unchanged_ids = [r["id"] for r in rows if existing.get(r["id"]) == r["content_hash"]]

        for i in range(0, len(changed), BATCH):
            batch = changed[i:i + BATCH]
            vecs = embed([embed_text(r) for r in batch])
            with engine.begin() as conn:
                for r, v in zip(batch, vecs):
                    conn.execute(UPSERT, {**r, "embedding": to_pgvector(v)})
            log.info("embedded and upserted %d/%d", min(i + BATCH, len(changed)), len(changed))

        with engine.begin() as conn:
            if unchanged_ids:
                conn.execute(
                    text("UPDATE jobs SET last_seen_at = now(), active = TRUE WHERE id = ANY(:ids)"),
                    {"ids": unchanged_ids},
                )
            if limit is None:
                deactivated = conn.execute(
                    text(
                        "UPDATE jobs SET active = FALSE "
                        "WHERE source = 'remotive' AND active "
                        "AND NOT (id = ANY(:ids))"
                    ),
                    {"ids": [r["id"] for r in rows]},
                ).rowcount
            else:
                deactivated = 0

        stats = {
            "status": "ok",
            "fetched": len(rows),
            "embedded_new_or_changed": len(changed),
            "unchanged": len(unchanged_ids),
            "deactivated_stale": deactivated,
            "seconds": round(time.time() - started, 1),
        }
        log.info("ingest done: %s", stats)
        _state["last"] = stats
        return stats
    except Exception as e:
        log.exception("ingest failed")
        _state["last"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        raise
    finally:
        _state["running"] = False
        _run_lock.release()


def start_background(**kwargs) -> bool:
    """Start ingestion in a daemon thread. Returns False if one is already running."""
    if _state["running"]:
        return False

    def _target():
        try:
            run(**kwargs)
        except Exception:
            pass  # already logged and recorded in _state

    threading.Thread(target=_target, daemon=True).start()
    return True


def status() -> dict:
    return dict(_state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max jobs to request from the API")
    ap.add_argument("--file", help="load a saved JSON file instead of calling the API")
    ap.add_argument("--save-raw", help="save the raw API response to this path")
    args = ap.parse_args()
    print(json.dumps(run(limit=args.limit, file=args.file, save_raw=args.save_raw), indent=2))
