import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ingest, llm, retrieval
from .agent import run_agent
from .config import settings
from .db import init_db
from .memory import DbMemory
from .tools import DbBackend, ToolBox

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("jobmatch")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limit(request: Request) -> None:
    fwd = request.headers.get("x-forwarded-for")
    ip = (fwd or (request.client.host if request.client else "unknown")).split(",")[0].strip()
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= settings.rate_limit_per_hour:
            raise HTTPException(429, "Rate limit reached. Please try again later.")
        q.append(now)


def _warmup() -> None:
    try:
        from .embeddings import embed_one

        embed_one("warmup")
    except Exception:
        log.exception("embedding warmup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    threading.Thread(target=_warmup, daemon=True).start()
    if settings.auto_ingest and retrieval.job_count(active_only=False) == 0:
        log.info("jobs table is empty; starting background ingest")
        ingest.start_background()
    yield


app = FastAPI(title="JobMatch Agent", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    return {"ok": True, "active_jobs": retrieval.job_count(), "ingest": ingest.status()}


@app.get("/api/search")
def api_search(
    request: Request,
    q: str = Query(min_length=2, max_length=300),
    mode: str = Query("hybrid", pattern="^(hybrid|vector|keyword)$"),
    k: int = Query(10, ge=1, le=25),
    location: str | None = None,
    job_type: str | None = None,
):
    _rate_limit(request)
    return {"mode": mode, "results": retrieval.search(q, mode=mode, k=k, location=location, job_type=job_type)}


@app.post("/api/chat")
def api_chat(body: ChatIn, request: Request):
    if not settings.llm_api_key:
        raise HTTPException(503, "LLM_API_KEY is not configured on the server.")
    _rate_limit(request)

    memory = DbMemory()
    toolbox = ToolBox(body.session_id, DbBackend())
    try:
        result = run_agent(
            body.message, body.session_id, memory, toolbox, llm.chat, max_steps=settings.max_agent_steps
        )
    except Exception:
        log.exception("agent run failed")
        raise HTTPException(502, "The language model request failed. Please try again.")

    try:
        memory.compact(body.session_id, llm.chat)
    except Exception:
        log.exception("memory compaction failed")

    return {
        "message": result.reply.message,
        "jobs": retrieval.get_jobs(result.reply.job_ids),
        "follow_up_questions": result.reply.follow_up_questions,
        "trace": result.trace,
        "meta": {
            "steps": result.steps,
            "format_retries": result.format_retries,
            "dropped_job_ids": result.dropped_job_ids,
            "fallback": result.fallback,
        },
    }


@app.get("/api/saved")
def api_saved(session_id: str = Query(min_length=8, max_length=64)):
    return {"jobs": DbBackend().list_saved(session_id)}


@app.post("/admin/ingest", status_code=202)
def admin_ingest(x_admin_token: str = Header(default="")):
    if not settings.admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(403, "Forbidden")
    return {"started": ingest.start_background()}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
