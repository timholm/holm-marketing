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
    """Embed contents and write back. Returns number updated.

    2026-05-02: switched from per-row UPDATE in a Python loop to a single
    bulk UPDATE ... FROM (VALUES ...) statement. With BATCH_SIZE=500 this
    drops ~500 round-trips per cycle to 1, taking cycle time from ~30s
    to a few seconds.

    Tries psycopg2.extras.execute_values first (fast path if image has
    psycopg2-binary installed); falls back to psycopg3 by composing a
    multi-VALUES UPDATE with a flat parameter list.
    """
    if not batch:
        return 0
    contents = [row[2] or " " for row in batch]
    try:
        embeddings = embed_batch(contents)
    except Exception as e:
        log.warning("embed_batch failed: %s", e)
        return 0
    assert embeddings.shape == (len(batch), EMBEDDING_DIM)

    update_rows = [
        (int(post_id), posted_at, list(emb.astype(float)))
        for (post_id, posted_at, _content), emb in zip(batch, embeddings)
    ]

    bulk_sql_template = (
        "UPDATE posts AS p SET embedding = u.emb "
        "FROM (VALUES {values}) AS u(id, posted_at, emb) "
        "WHERE p.id = u.id AND p.posted_at = u.posted_at"
    )
    row_template = "(%s, %s::timestamptz, %s::real[])"

    updated = 0
    try:
        # Fast path: psycopg2 has execute_values, which lets the driver
        # build the VALUES clause for us.
        import psycopg2.extras as _pg2_extras  # type: ignore[import-not-found]
        sql_with_pct_s = bulk_sql_template.format(values="%s")
        with conn.cursor() as cur:
            _pg2_extras.execute_values(
                cur, sql_with_pct_s, update_rows, template=row_template,
            )
            updated = cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(update_rows)
        conn.commit()
        return updated
    except ImportError:
        pass

    # psycopg3 path: build VALUES clause manually with N copies of the row
    # template and a single flat parameter list. Still ONE round-trip.
    values_clause = ",".join([row_template] * len(update_rows))
    sql = bulk_sql_template.format(values=values_clause)
    flat_params: list = []
    for (post_id, posted_at, emb_list) in update_rows:
        flat_params.extend([post_id, posted_at, emb_list])
    try:
        with conn.cursor() as cur:
            cur.execute(sql, flat_params)
            updated = cur.rowcount if cur.rowcount and cur.rowcount > 0 else len(update_rows)
        conn.commit()
    except Exception as e:
        log.warning("bulk UPDATE failed: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
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
