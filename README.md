# JobMatch Agent

A job-search agent over a live, changing index of remote jobs. Describe what you want in plain English; the agent
searches with hybrid (semantic + keyword) retrieval, reads job details when it needs to, remembers your preferences
across turns, and explains why each job fits.

**Live demo:** _add your Render URL_ · **Eval results:** see [Evaluation](#evaluation)

## What's in it

- **Hybrid retrieval** in Postgres: `pgvector` (HNSW, cosine) over MiniLM embeddings + full-text search
  (`tsvector`/GIN), merged with Reciprocal Rank Fusion, with location / job-type filters.
- **Tool-calling agent, written by hand (no framework):** tools `search_jobs`, `get_job`, `save_job`,
  `list_saved_jobs`; step cap; tool errors fed back to the model; final answer must be JSON that validates against a
  Pydantic schema, with retries that feed the validation error back.
- **Grounding check:** job ids in the final answer must have been returned by a tool in this session, otherwise they are
  dropped (and reported in the response `meta`). This is the guard against hallucinated matches.
- **Memory:** last 10 messages replayed each turn; older turns are folded into a rolling per-session profile summary.
- **Incremental ingestion:** content-hash diffing means unchanged postings are not re-embedded; postings that disappear
  from the source are marked inactive and excluded from search.
- **Evals:** `python -m evals.run_evals` compares keyword vs vector vs hybrid (Hit@5, P@5, MRR@10, nDCG@10).
- **Ops basics:** FastAPI, Docker Compose, per-IP rate limit and input length caps on public endpoints, LLM timeouts and
  retries, offline unit tests for the agent loop.

## Architecture

```
            ┌────────────┐   POST /api/chat    ┌───────────────────────────────┐
  Browser ─▶│ static UI  │────────────────────▶│ FastAPI                       │
            └────────────┘                     │  run_agent(): LLM ⇄ ToolBox   │
                                               │   - step cap, JSON validation │
                                               │   - grounding check           │
                                               └──────┬─────────────┬──────────┘
                                                      │             │
                              OpenAI-compatible LLM ◀─┘             ▼
                              (Groq by default)           Postgres + pgvector
                                                          jobs (vector + tsvector)
                                                          chat_messages / session_profiles
   Remotive API ──▶ app/ingest.py (hash diff → embed → upsert → deactivate stale) ─▶ jobs
```

## Run locally

```bash
cp .env.example .env            # add LLM_API_KEY (free key from console.groq.com, or any OpenAI-compatible provider)
docker compose up --build       # app on http://localhost:8000
```

On first start the app ingests jobs from Remotive automatically (`AUTO_INGEST=1`). To do it by hand:

```bash
docker compose exec app python -m app.ingest --save-raw data/remotive.json   # fetch and keep a copy
docker compose exec app python -m app.ingest --file data/remotive.json       # re-ingest without hitting the API
```

Remotive asks for infrequent API calls and credit to Remotive, so cache the raw file while developing.

Without Docker: run Postgres with the pgvector extension, set `DATABASE_URL`, then
`pip install -r requirements-dev.txt && uvicorn app.main:app --reload`.

## Evaluation

```bash
docker compose exec app python -m evals.run_evals     # writes evals/results.md and results.json
pytest                                                # offline unit tests (no DB, network or LLM key needed)
```

Paste your own results here after running them:

| Mode | Hit@5 | P@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| keyword | | | | |
| vector | | | | |
| hybrid | | | | |

**How to read these.** `evals/queries.json` has 25 queries. A job counts as relevant if one of the query's
`relevant_terms` appears in its title or tags. These are cheap "silver" labels and they flatter keyword search a little
(both look at the same fields). Queries with no relevant job in the current corpus are skipped and listed in the
report. For stronger evidence, add hand-labelled `relevant_ids` to a query; they take precedence over the terms.

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | chat UI |
| `POST /api/chat` `{session_id, message}` | agent turn → `message`, `jobs`, `follow_up_questions`, `trace`, `meta` |
| `GET /api/search?q=&mode=hybrid\|vector\|keyword&k=&location=&job_type=` | raw retrieval, for debugging and evals |
| `GET /api/saved?session_id=` | the session's shortlist |
| `GET /health` | status, active job count, last ingest result |
| `POST /admin/ingest` (header `x-admin-token`) | trigger a background ingest |

## Deploy (free tier)

1. **Database:** create a free Postgres on [Neon](https://neon.tech) (pgvector is supported; the app runs
   `CREATE EXTENSION vector` itself). Copy the connection string.
2. **App:** on [Render](https://render.com), New → Web Service → connect this repo → Runtime: **Docker**.
3. Environment variables: `DATABASE_URL` (the Neon string), `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`,
   `ADMIN_TOKEN` (any long random string), `AUTO_INGEST=1`.
4. Open the service URL. The first request after idle is slow (free instances sleep and must reload the embedding model).

## Design decisions

- **RRF over weighted score blending:** vector similarity and `ts_rank_cd` live on different scales; rank-based fusion
  needs no tuning and is hard to break.
- **`fastembed` (ONNX) instead of `sentence-transformers`:** same MiniLM model, no PyTorch, so the image is much smaller
  and fits free-tier memory.
- **Postgres for everything:** one system for vectors, full-text, sessions and the shortlist; fewer moving parts than a
  separate vector DB at this scale.
- **Final answer as validated JSON, not free text:** the UI renders job cards from ids, and the grounding check can only
  work on structured output.
- **Provider-agnostic LLM layer:** an OpenAI-compatible client with a configurable base URL, so Groq, OpenAI or Gemini's
  compatibility endpoint are an env-var change.

## Known limitations / next steps

- Single source (Remotive) and a few hundred to a few thousand jobs; HNSW with a metadata filter is a post-filter, so very
  selective filters can return fewer than `k` results.
- No re-ranker yet (cross-encoder over the fused top 50 is the obvious next step) and no personalisation beyond the
  profile summary (next: boost by saved and clicked jobs).
- Ingestion runs in-process; moving it to a worker with a Redis queue and a schedule is the next step for freshness.
- Evals use silver labels; hand-label ids for a trustworthy benchmark and add an LLM-judge check on answer quality.
- Chat history is stored per anonymous session id with no expiry; add retention before real users.
