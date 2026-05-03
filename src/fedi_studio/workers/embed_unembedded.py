"""Backfill embeddings for posts that arrived without them.

The sync_v1_to_v2 worker (when SYNC_SKIP_EMBED=1) skips the embedding step
to keep up with mass-crawler's 200k+/hr post production. This worker picks
up those zero-embedding posts and computes real Model2Vec embeddings in
big batches so they become eligible for score_all + ranked /today display.

Architecture:
  * SELECT posts WHERE embedding[1]=0 AND embedding[2]=0 AND embedding[3]=0
    LIMIT 500 FOR UPDATE SKIP LOCKED (so multiple replicas can run in parallel
    without re-embedding the same row)
  * embed_batch the contents
  * UPDATE posts SET embedding = ... WHERE id = ...
  * commit, repeat

Read-only against Mastodon. Pure CPU work.

Run:
    python -m fedi_studio.workers.embed_unembedded
"""

from __future__ import annotations

import logging
import os
import signal
import time

import numpy as np

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed_unembedded")

BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "500"))
SLEEP_WHEN_EMPTY_S = int(os.environ.get("SLEEP_WHEN_EMPTY_S", "30"))

_running = True


def _stop(*_):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def fetch_batch(conn, n: int) -> list[tuple[int, "datetime", str]]:
    """Pull rows with zero/null embedding. FOR UPDATE SKIP LOCKED is safe for
    parallel replicas — each gets a different slice."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, posted_at, content
            FROM posts
            WHERE (embedding IS NULL
                   OR (embedding[1] = 0 AND embedding[2] = 0 AND embedding[3] = 0
                       AND embedding[4] = 0 AND embedding[5] = 0))
              AND content IS NOT NULL
              AND length(content) > 0
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (n,),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def embed_and_update(conn, batch: list) -> int:
    """Embed contents and write back. Returns number updated."""
    if not batch:
        return 0
    contents = [row[2] or " " for row in batch]
    try:
        embeddings = embed_batch(contents)
    except Exception as e:
        log.warning("embed_batch failed: %s", e)
        return 0
    assert embeddings.shape == (len(batch), EMBEDDING_DIM)

    updated = 0
    with conn.cursor() as cur:
        for (post_id, posted_at, _content), emb in zip(batch, embeddings):
            try:
                cur.execute(
                    "UPDATE posts SET embedding = %s WHERE id = %s AND posted_at = %s",
                    (list(emb.astype(float)), post_id, posted_at),
                )
                if cur.rowcount > 0:
                    updated += 1
            except Exception as e:
                log.debug("update failed for id=%s: %s", post_id, e)
    conn.commit()
    return updated


def main() -> int:
    init_pool()
    log.info("embed_unembedded starting (batch=%d, sleep_when_empty=%ds)",
             BATCH_SIZE, SLEEP_WHEN_EMPTY_S)
    cycle = 0
    total = 0
    while _running:
        cycle += 1
        t0 = time.time()
        try:
            with get_conn() as conn:
                batch = fetch_batch(conn, BATCH_SIZE)
                if not batch:
                    log.info("cycle %d: queue empty, sleeping %ds", cycle, SLEEP_WHEN_EMPTY_S)
                    for _ in range(SLEEP_WHEN_EMPTY_S):
                        if not _running:
                            break
                        time.sleep(1)
                    continue
                n = embed_and_update(conn, batch)
                total += n
        except Exception as e:
            log.warning("cycle %d error: %s", cycle, e)
            time.sleep(5)
            continue
        elapsed = time.time() - t0
        rate = n / max(elapsed, 0.001) * 60
        log.info("cycle %d: embedded=%d total=%d elapsed=%.1fs rate=%.0f/min",
                 cycle, n, total, elapsed, rate)
    log.info("clean shutdown after %d cycles, %d total embedded", cycle, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
