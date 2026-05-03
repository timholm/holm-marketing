"""Backfill v1 post_scores into v2 post_scores for high-confidence matches.

This is a one-shot worker that transfers scoring data from v1 fedi_discover to v2
fedi_studio for posts that already exist in v2. We match via:
  1. URI match (v2.posts.uri = v1.posts.url)
  2. Content hash match (fallback)

For each match with a v1 score, we insert into v2 post_scores with:
  - probability = score / 100.0 (clamped to [0, 1])
  - reasoning = {source: 'v1_backfill', v1_score: score}
  - scorer_version = 'v1-backfill'

Uses ON CONFLICT DO NOTHING to avoid overwriting existing v2 scores.
Idempotent and restartable.

Read-only against v1. No outbound Mastodon API calls.

Usage:
    python -m fedi_studio.workers.backfill_v1_scores [--max-rows N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import psycopg

from fedi_studio.models.db import get_conn, get_dsn, init_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_v1_scores")

V1_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon",
)

BATCH_SIZE = 500


def fetch_v2_batch(
    conn: psycopg.Connection,
    last_id: int,
    batch_size: int,
) -> list[dict]:
    """Fetch up to batch_size v2 posts with id > last_id, ordered by id ASC.

    Returns post id, uri, posted_at, and content_hash.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, uri, posted_at, content_hash
            FROM posts
            WHERE id > %s
            ORDER BY id ASC
            LIMIT %s
            """,
            (last_id, batch_size),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_v1_matches(
    v1_conn: psycopg.Connection,
    v2_posts: list[dict],
) -> dict[int, dict]:
    """Find matching v1 rows for a batch of v2 posts.

    Strategy:
      1. First try URI match (v1.posts.url = v2.posts.uri)
      2. Fallback to content_hash match (v1.posts.content_hash = v2.posts.content_hash)

    Returns: dict[v2_post_id] = {score, ai_score, ...}
    """
    if not v2_posts:
        return {}

    v2_by_uri = {p["uri"]: p for p in v2_posts if p["uri"]}
    v2_by_hash = {p["content_hash"]: p for p in v2_posts}

    matches: dict[int, dict] = {}

    # Strategy 1: URI match
    if v2_by_uri:
        uris = list(v2_by_uri.keys())
        with v1_conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.url, s.score, s.ai_score
                FROM posts p
                LEFT JOIN post_scores s ON s.post_id = p.id
                WHERE p.url = ANY(%s)
                """,
                (uris,),
            )
            for url, score, ai_score in cur.fetchall():
                if url in v2_by_uri:
                    v2_post_id = v2_by_uri[url]["id"]
                    # Take the higher of score and ai_score
                    best_score = None
                    if score is not None:
                        best_score = score
                    if ai_score is not None:
                        best_score = max(best_score or 0, ai_score)
                    if best_score is not None:
                        matches[v2_post_id] = {"score": best_score}

    # Strategy 2: Content hash match (for posts not matched by URI)
    remaining_hashes = [h for h in v2_by_hash.keys() if h not in matches]
    if remaining_hashes:
        with v1_conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.content_hash, s.score, s.ai_score
                FROM posts p
                LEFT JOIN post_scores s ON s.post_id = p.id
                WHERE p.content_hash = ANY(%s)
                """,
                (remaining_hashes,),
            )
            for content_hash, score, ai_score in cur.fetchall():
                if content_hash in v2_by_hash:
                    v2_post_id = v2_by_hash[content_hash]["id"]
                    if v2_post_id not in matches:  # Don't override URI match
                        best_score = None
                        if score is not None:
                            best_score = score
                        if ai_score is not None:
                            best_score = max(best_score or 0, ai_score)
                        if best_score is not None:
                            matches[v2_post_id] = {"score": best_score}

    return matches


def insert_v2_scores(
    v2_conn: psycopg.Connection,
    v2_posts: list[dict],
    v1_matches: dict[int, dict],
) -> int:
    """Insert backfilled scores into v2 post_scores.

    Returns count of rows inserted.
    """
    inserted = 0
    with v2_conn.cursor() as cur:
        for v2_post in v2_posts:
            post_id = v2_post["id"]
            if post_id not in v1_matches:
                continue

            match = v1_matches[post_id]
            v1_score = match["score"]

            # Clamp probability to [0, 1]
            probability = max(0.0, min(1.0, v1_score / 100.0))
            reasoning = {"source": "v1_backfill", "v1_score": int(v1_score)}

            cur.execute(
                """
                INSERT INTO post_scores (post_id, posted_at, probability, reasoning, scorer_version, scored_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (post_id, posted_at) DO NOTHING
                """,
                (
                    post_id,
                    v2_post["posted_at"],
                    probability,
                    json.dumps(reasoning),
                    "v1-backfill",
                    datetime.now(timezone.utc),
                ),
            )
            inserted += cur.rowcount

    v2_conn.commit()
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after scanning this many v2 rows (debug)",
    )
    args = parser.parse_args()

    log.info("backfill_v1_scores: starting")
    log.info("V1 DSN: %s", V1_DSN)
    log.info("V2 DSN: %s", get_dsn())

    init_pool()

    last_id = 0
    total_scanned = 0
    total_backfilled = 0
    total_skipped = 0
    started = time.time()
    last_log = started

    with psycopg.connect(V1_DSN, connect_timeout=10) as v1_conn:
        v1_conn.read_only = True
        while True:
            with get_conn() as v2_conn:
                batch = fetch_v2_batch(v2_conn, last_id, BATCH_SIZE)

            if not batch:
                log.info("No more rows. Done.")
                break

            last_id = batch[-1]["id"]
            total_scanned += len(batch)

            # Find matching v1 rows
            v1_matches = fetch_v1_matches(v1_conn, batch)

            # Insert matched scores into v2
            with get_conn() as v2_conn:
                backfilled = insert_v2_scores(v2_conn, batch, v1_matches)

            total_backfilled += backfilled
            total_skipped += len(batch) - backfilled

            # Log progress every ~1000 rows
            now = time.time()
            if total_scanned % 1000 < BATCH_SIZE or (now - last_log) >= 30:
                elapsed = now - started
                rate = total_scanned / elapsed if elapsed > 0 else 0.0
                log.info(
                    "scanned=%d backfilled=%d skipped=%d last_id=%d rate=%.1f/s elapsed=%.0fs",
                    total_scanned,
                    total_backfilled,
                    total_skipped,
                    last_id,
                    rate,
                    elapsed,
                )
                last_log = now

            if args.max_rows and total_scanned >= args.max_rows:
                log.info("Hit --max-rows limit (%d), stopping", args.max_rows)
                break

    elapsed = time.time() - started
    log.info(
        "DONE: scanned=%d backfilled=%d skipped=%d in %.1fs (%.1f rows/s)",
        total_scanned,
        total_backfilled,
        total_skipped,
        elapsed,
        total_scanned / elapsed if elapsed > 0 else 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
