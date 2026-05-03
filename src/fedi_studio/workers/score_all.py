"""Score every post that doesn't yet have a score.

Idempotent: re-running only scores posts missing from post_scores. Run after
each ingest, or on a 5-minute timer once a stream listener is wired up.

Cold start: with no training data, all scores will hover near 0.5. That's
correct behavior; the scorer reports "I don't know" until Tim provides feedback.

If a trained model is on disk at MODEL_PATH (or its container equivalent
/app/models/scorer_v1.pkl), it is loaded automatically and scores will reflect
learned preferences instead of the uniform cold-start value.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.scorer import ScoreInput, Scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_all")

SCORER_VERSION = "v1-quality-weighted"
BATCH = 500

# Model path discovery. Order:
#   1. FEDI_STUDIO_SCORER_MODEL env var (explicit override)
#   2. /app/models/scorer_v2.pkl  (preferred container path, post-2026-04-25 retrain)
#   3. /app/models/scorer_v1.pkl  (legacy container path)
#   4. <repo>/models/scorer_v2.pkl (dev / on-host runs)
#   5. <repo>/models/scorer_v1.pkl (dev fallback for rollback)
_REPO_ROOT = Path(__file__).resolve().parents[3]
def _resolve_model_path() -> str:
    env = os.environ.get("FEDI_STUDIO_SCORER_MODEL")
    if env:
        return env
    candidates = [
        "/app/models/scorer_v2.pkl",
        "/app/models/scorer_v1.pkl",
        str(_REPO_ROOT / "models" / "scorer_v2.pkl"),
        str(_REPO_ROOT / "models" / "scorer_v1.pkl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Last-resort default (load_or_initialize handles missing files gracefully).
    return str(_REPO_ROOT / "models" / "scorer_v1.pkl")


DEFAULT_MODEL_PATH = _resolve_model_path()


def score_all(rescore: bool = False, model_path: str | None = None) -> int:
    """Score posts. If rescore=True, overwrite existing scores."""
    init_pool()
    scorer = Scorer.load_or_initialize(model_path or DEFAULT_MODEL_PATH)
    total = 0
    while True:
        with get_conn() as conn:
            cur = conn.cursor()
            if rescore:
                cur.execute(
                    """
                    SELECT p.id, p.posted_at, p.author_acct, p.content, p.embedding,
                           p.language, p.favourites_count, p.reblogs_count, p.media_count
                    FROM posts p
                    LEFT JOIN post_scores s ON s.post_id = p.id AND s.posted_at = p.posted_at
                    WHERE p.embedding IS NOT NULL
                      AND (s.scorer_version IS NULL OR s.scorer_version != %s)
                    LIMIT %s
                    """,
                    (SCORER_VERSION, BATCH),
                )
            else:
                cur.execute(
                    """
                    SELECT p.id, p.posted_at, p.author_acct, p.content, p.embedding,
                           p.language, p.favourites_count, p.reblogs_count, p.media_count
                    FROM posts p
                    LEFT JOIN post_scores s ON s.post_id = p.id AND s.posted_at = p.posted_at
                    WHERE s.post_id IS NULL
                      AND p.embedding IS NOT NULL
                    LIMIT %s
                    """,
                    (BATCH,),
                )
            rows = cur.fetchall()
            if not rows:
                break
            inserts = []
            for post_id, posted_at, author, content, emb_array, lang, favs, rebs, mc in rows:
                emb = np.array(emb_array, dtype=np.float32)
                result = scorer.score(
                    ScoreInput(
                        content=content or "",
                        author_acct=author or "",
                        posted_at=(
                            posted_at if posted_at.tzinfo
                            else posted_at.replace(tzinfo=timezone.utc)
                        ),
                        embedding=emb,
                        language=lang,
                        favourites_count=int(favs or 0),
                        reblogs_count=int(rebs or 0),
                        has_media=bool(mc and mc > 0),
                        content_length=len(content or ""),
                    )
                )
                inserts.append(
                    (post_id, posted_at, result.probability, result.reasoning, SCORER_VERSION)
                )
            cur.executemany(
                """
                INSERT INTO post_scores (post_id, posted_at, probability, reasoning, scorer_version)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (post_id, posted_at) DO UPDATE SET
                    probability = EXCLUDED.probability,
                    reasoning = EXCLUDED.reasoning,
                    scorer_version = EXCLUDED.scorer_version,
                    scored_at = NOW()
                """,
                [
                    (pid, pa, prob, __import__("json").dumps(reasoning), ver)
                    for pid, pa, prob, reasoning, ver in inserts
                ],
            )
            conn.commit()
            total += len(inserts)
            log.info("Scored %d (total %d)", len(inserts), total)
    log.info("Done: %d total", total)
    return total


if __name__ == "__main__":
    rescore = "--rescore" in sys.argv
    sys.exit(0 if score_all(rescore=rescore) >= 0 else 1)
