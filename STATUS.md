# fedi-studio Status

Last updated: 2026-04-24

## Phase 0: Emergency Containment — DONE

All auto-engagement halted, 3,336 pending follow requests cancelled, account state remediated. See `/Users/tim/.claude/plans/fuzzy-weaving-map.md`.

## Phase 1: Foundation — IN PROGRESS

### Done

- [x] Repo structure at `tools/fedi-studio/`
- [x] Schema migration `migrations/001_init.sql` applied to new `fedi_studio` database on `fedi-postgresql` (rpi-12)
  - 14 tables, posts and events partitioned by month, BRIN on posted_at
  - Embedding stored as `REAL[]` (pgvector swap-in possible later)
  - `events` table replaces v1's denormalized signal columns
  - No `actions` table by design
  - 10 known-bad domains seeded in blocklist
- [x] Python 3.12 project with `uv` venv and pinned deps
- [x] `services/embedder.py`: Model2Vec `potion-base-32M` integration. 512-dim, ~100k posts/sec on CPU.
- [x] `services/scorer.py`: `Scorer` class combining SGDClassifier + cosine to user_centroid + author prior + recency decay. Hard-rules layer respects `#nobot`, `#noindex`, blocklists, language.
- [x] `models/db.py`: psycopg3 connection pool to fedi_studio
- [x] `workers/ingest_v1.py`: one-shot ingest from v1 PG into fedi_studio with embedding. Verified: **342 posts ingested with 512-dim embeddings in ~3s**.
- [x] `workers/score_all.py`: score every unscored post. Verified: **342 scored, range 0.475–0.525 (cold-start as expected)**.

### Verification (all green at last run)

```
$ kubectl exec -n fedi-discover fedi-postgresql-0 -- psql -U mastodon -d fedi_studio -c "\dt"
14 tables present.

$ python scripts/smoke_test_scorer.py
embed('hello world'): shape=(512,) norm=1.000 OK
Scorer cold-start returns ~0.5 for every input.
After 3 partial_fit calls, scores diverge as expected.

$ python -m fedi_studio.workers.ingest_v1 --limit 500
342 inserted, 0 skipped, 0 blocked. (v1 only has 342 posts in last 14 days.)

$ python -m fedi_studio.workers.score_all
342 scored. min=0.475 max=0.525 avg=0.502.
```

### Not done in this session (Phase 1 stretch goals)

- [ ] **ghcr.io / Flux migration** — eliminates rpi-1 SPOF for image registry. Python deps already use uv/pyproject for reproducible builds; need `Dockerfile` + GH Actions workflow + Flux image-automation-controller install on cluster.
- [ ] **Rust SSE listener** — Python fallback works fine for MVP. Real win is when we can have a long-lived listener that doesn't restart every 5 minutes via CronJob. Keep on roadmap.
- [ ] **NATS JetStream** — only valuable once we have multiple consumers fanning out from the firehose. Not blocking.
- [ ] **pgvector + HNSW index** — only matters when posts table exceeds ~1M rows for the digest query. Postgres image needs swap to `pgvector/pgvector:pg16`. Migration noted in `migrations/001_init.sql`.

## Phase 2: Morning Catch-Up MVP — NOT STARTED

The foundation is ready. Next session work, in order:

1. `web/app.py`: FastAPI app skeleton with HTMX templates
2. `web/routes/today.py`: GET `/today` returning top 50 scored posts grouped by topic
3. `services/digest.py`: assembly + LLM-generated topic summaries (use small local Ollama model for summaries only — never for scoring)
4. `web/routes/feedback.py`: POST endpoints for bookmark / dismiss / read that emit `events` rows AND call `scorer.partial_fit`
5. Auth middleware (reuse v1 pattern or add basic auth)
6. Cron job: daily user_centroid refresh from last 1000 bookmarked posts
7. Deploy to k8s as single Deployment, ingress under `studio.holm.community`

## How to resume

```bash
cd /Users/tim/Documents/Holm/tools/fedi-studio
# Always have this running while iterating locally:
kubectl port-forward -n fedi-discover svc/fedi-postgresql 30141:5432 &

# Smoke tests
uv run python scripts/smoke_test_scorer.py
uv run python -m fedi_studio.workers.ingest_v1 --limit 1000
uv run python -m fedi_studio.workers.score_all
```

## Critical design rules (from plan, do not violate)

1. The tool never acts on another person's content without Tim explicitly clicking a button in the same session.
2. Respect `#nobot`, `#noindex`, `discoverable=false`. Blocklist consulted at ingest.
3. Every AI output is labeled, explained, and editable. Never auto-posted.
4. Local-first. No third-party APIs called with follower data.
5. No `actions` queue. No batch workers that touch the Mastodon API for engagement. None. Ever.
