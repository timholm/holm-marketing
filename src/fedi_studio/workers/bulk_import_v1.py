"""Bulk import the high-signal subset of v1 fedi_discover into v2 fedi_studio.

This is a one-shot, restartable seed of the v2 corpus. v1 crawled ~14M posts;
post_scores has ~250k rows scoring >= 40. The subset that lives in the v2
partition window (2025-11..2026-06) and still has a matching row in posts is
~11k. We pull every such row, embed with Model2Vec, and upsert into
fedi_studio.posts.

Differences from `ingest_v1.py` (the older 14-day-window worker):
    - joins post_scores so we only carry posts with v1 signal (score >= 40)
    - 180-day lookback instead of 14
    - skips posts outside the v2 partition window (2025-11..2026-06)
    - upserts engagement counts on conflict instead of DO NOTHING
    - batch size 500, savepoint-per-row so a bad row doesn't kill the batch
    - strips HTML (v1 content is mixed plain/HTML)

Read-only against v1. No outbound Mastodon API calls. v2 inserts only.

Usage:
    python -m fedi_studio.workers.bulk_import_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import psycopg

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch
from fedi_studio.workers.pull_home import slim_media, strip_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bulk_import_v1")

V1_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon",
)

BATCH_SIZE = 500
LOOKBACK_DAYS = 180
MIN_SCORE = 40

# v2 has partitions for 2025-11 through 2026-06 (inclusive).
# We refuse to insert outside that range.
PARTITION_START = datetime(2025, 11, 1, tzinfo=timezone.utc)
PARTITION_END = datetime(2026, 7, 1, tzinfo=timezone.utc)


def load_blocklist(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT pattern FROM blocklist")
        return {r[0].lower() for r in cur}


def is_blocked(acct: str, blocklist: set[str]) -> bool:
    if not acct:
        return False
    a = acct.lower()
    if a in blocklist:
        return True
    if "@" in a:
        domain = a.split("@", 1)[1]
        if domain in blocklist:
            return True
    return False


def parse_posted_at(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        # 2026-01-28T23:27:56.298Z, 2026-04-25 02:43:45, etc
        s = s.replace("Z", "+00:00")
        # Some rows lack tz; assume UTC.
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def parse_tags(raw) -> list[str]:
    """v1 stored tags as a JSON-encoded text column."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    s = str(raw).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, list):
        out: list[str] = []
        for t in parsed:
            if isinstance(t, str):
                out.append(t)
            elif isinstance(t, dict) and "name" in t:
                out.append(str(t["name"]))
        return out
    return []


def fetch_v1_batch(
    conn: psycopg.Connection,
    since: datetime,
    last_id: int,
    batch_size: int,
    min_score: int,
) -> list[dict]:
    """Return up to `batch_size` rows with id > last_id, ordered by id ASC.

    Keyset pagination on posts.id keeps the query cheap and resumable across
    restarts of this script, even if the server crashes mid-import.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.id,
                p.url,
                p.mastodon_id,
                p.author_acct,
                p.author_display_name,
                p.content,
                p.tags,
                p.favourites_count,
                p.reblogs_count,
                p.media_count,
                p.posted_at,
                s.score
            FROM posts p
            JOIN post_scores s ON s.post_id = p.id
            WHERE p.id > %s
              AND p.posted_at::timestamptz >= %s
              AND p.content IS NOT NULL
              AND length(p.content) > 30
              AND s.score >= %s
            ORDER BY p.id ASC
            LIMIT %s
            """,
            (last_id, since.isoformat(), min_score, batch_size),
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def insert_post(cur: psycopg.Cursor, post: dict, embedding) -> bool:
    """Insert one v1 row into v2.posts. Returns True if inserted/updated."""
    raw_content = post.get("content") or ""
    content = strip_html(raw_content)
    if len(content) < 10:
        return False

    posted_at = parse_posted_at(post.get("posted_at"))
    if posted_at is None:
        return False
    if posted_at < PARTITION_START or posted_at >= PARTITION_END:
        return False

    content_hash = hashlib.md5(content.encode("utf-8")).digest()
    tags = parse_tags(post.get("tags"))
    url = post.get("url")
    if not url:
        return False

    cur.execute(
        """
        INSERT INTO posts (
            uri, url, author_acct, content, content_hash,
            tags, language, in_reply_to_id, sensitive,
            media_count, favourites_count, reblogs_count,
            posted_at, embedding,
            local_id, media_attachments, account_avatar, account_display_name
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (uri, posted_at) DO UPDATE SET
            content = EXCLUDED.content,
            content_hash = EXCLUDED.content_hash,
            tags = EXCLUDED.tags,
            media_count = EXCLUDED.media_count,
            favourites_count = EXCLUDED.favourites_count,
            reblogs_count = EXCLUDED.reblogs_count,
            local_id = COALESCE(EXCLUDED.local_id, posts.local_id),
            account_display_name = COALESCE(EXCLUDED.account_display_name, posts.account_display_name),
            embedding = EXCLUDED.embedding
        """,
        (
            url,                                            # uri (v1 has no uri, fall back to url)
            url,                                            # url
            post.get("author_acct") or "",
            content,
            content_hash,
            tags,
            None,                                           # language: v1 doesn't track it
            None,                                           # in_reply_to_id: v1 doesn't track
            False,                                          # sensitive: v1 doesn't track
            int(post.get("media_count") or 0),
            int(post.get("favourites_count") or 0),
            int(post.get("reblogs_count") or 0),
            posted_at,
            embedding.tolist(),
            post.get("mastodon_id"),                        # local_id
            json.dumps(slim_media([])),                     # media_attachments: v1 doesn't keep urls
            None,                                           # account_avatar: v1 doesn't keep
            post.get("author_display_name") or "",
        ),
    )
    return cur.rowcount > 0


def process_batch(
    v2_conn: psycopg.Connection,
    rows: list[dict],
    blocklist: set[str],
) -> tuple[int, int, int]:
    """Insert one batch. Returns (inserted, skipped, blocked)."""
    # Filter blocklist before embedding to avoid wasting work
    kept: list[dict] = []
    blocked = 0
    for r in rows:
        if is_blocked(r.get("author_acct") or "", blocklist):
            blocked += 1
            continue
        kept.append(r)

    if not kept:
        return 0, 0, blocked

    # Build the texts we'll embed: HTML-stripped, just like we'd insert.
    texts = [strip_html(r.get("content") or "") for r in kept]
    embeddings = embed_batch(texts)
    assert embeddings.shape == (len(kept), EMBEDDING_DIM), (
        f"embeddings shape {embeddings.shape} vs {len(kept)} rows"
    )

    inserted = 0
    skipped = 0
    with v2_conn.cursor() as cur:
        for row, emb in zip(kept, embeddings):
            cur.execute("SAVEPOINT row_sp")
            try:
                if insert_post(cur, row, emb):
                    inserted += 1
                    cur.execute("RELEASE SAVEPOINT row_sp")
                else:
                    skipped += 1
                    cur.execute("RELEASE SAVEPOINT row_sp")
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT row_sp")
                log.warning("Row %s aborted: %s", row.get("id"), e)
                skipped += 1
    v2_conn.commit()
    return inserted, skipped, blocked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lookback-days", type=int, default=LOOKBACK_DAYS,
        help=f"How many days back to import (default {LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--min-score", type=int, default=MIN_SCORE,
        help=f"Minimum v1 post_scores.score (default {MIN_SCORE})",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after this many rows scanned (debug)",
    )
    args = parser.parse_args()

    min_score = args.min_score
    since = datetime.now(timezone.utc) - timedelta(days=args.lookback_days)

    log.info(
        "bulk_import_v1: lookback=%dd min_score=%d window=[%s, %s)",
        args.lookback_days, args.min_score,
        PARTITION_START.isoformat(), PARTITION_END.isoformat(),
    )

    init_pool()
    with get_conn() as conn:
        blocklist = load_blocklist(conn)
    log.info("Loaded %d blocklist patterns", len(blocklist))

    # Warm the model so the first batch's first call doesn't pay for the load
    log.info("Warming embedder...")
    _ = embed_batch(["warm up"])
    log.info("Embedder ready (dim=%d)", EMBEDDING_DIM)

    last_id = 0
    total_seen = 0
    total_inserted = 0
    total_skipped = 0
    total_blocked = 0
    started = time.time()
    last_log = started

    with psycopg.connect(V1_DSN, connect_timeout=10) as v1_conn:
        v1_conn.read_only = True
        while True:
            rows = fetch_v1_batch(v1_conn, since, last_id, BATCH_SIZE, min_score)
            if not rows:
                log.info("No more rows. Done.")
                break

            last_id = rows[-1]["id"]
            total_seen += len(rows)

            with get_conn() as v2_conn:
                inserted, skipped, blocked = process_batch(v2_conn, rows, blocklist)
            total_inserted += inserted
            total_skipped += skipped
            total_blocked += blocked

            # Progress log: every ~1000 rows scanned
            now = time.time()
            if total_seen % 1000 < BATCH_SIZE or (now - last_log) >= 30:
                elapsed = now - started
                rate = total_seen / elapsed if elapsed > 0 else 0.0
                log.info(
                    "scanned=%d inserted=%d skipped=%d blocked=%d "
                    "last_id=%d rate=%.1f/s elapsed=%.0fs",
                    total_seen, total_inserted, total_skipped, total_blocked,
                    last_id, rate, elapsed,
                )
                last_log = now

            if args.max_rows and total_seen >= args.max_rows:
                log.info("Hit --max-rows limit (%d), stopping", args.max_rows)
                break

    elapsed = time.time() - started
    log.info(
        "DONE: scanned=%d inserted=%d skipped=%d blocked=%d in %.1fs (%.1f rows/s)",
        total_seen, total_inserted, total_skipped, total_blocked, elapsed,
        total_seen / elapsed if elapsed > 0 else 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
