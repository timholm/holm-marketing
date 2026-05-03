"""One-shot ingest: copy posts from v1 fedi_discover into v2 fedi_studio.

Usage:
    python -m fedi_studio.workers.ingest_v1 [--limit N] [--since YYYY-MM-DD]

This is a transitional worker. Once the Rust SSE listener is running, posts will
flow live into v2 directly. For Phase 2 (Morning Catch-Up MVP) we just need a
seed of recent posts to score and serve.

Strategy:
    1. Pull posts from v1 posts table (last 14 days, has content, English-y)
    2. Embed each post with Model2Vec
    3. Insert into v2 posts table with embedding
    4. Skip duplicates (content_hash collision)
    5. Skip posts from blocklisted domains/authors

Rate: ~30k posts/min on a single Pi core (Model2Vec is the bottleneck, not PG).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterator

import psycopg

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_v1")

V1_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon",
)

BATCH_SIZE = 200


def fetch_v1_posts(since: datetime, limit: int) -> Iterator[list[dict]]:
    """Yield batches of posts from v1 DB, ordered by posted_at DESC."""
    with psycopg.connect(V1_DSN, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.url,
                    p.author_acct,
                    p.author_display_name,
                    p.content,
                    p.tags,
                    p.favourites_count,
                    p.reblogs_count,
                    p.media_count,
                    p.posted_at
                FROM posts p
                WHERE p.posted_at::timestamptz >= %s
                  AND p.content IS NOT NULL
                  AND length(p.content) > 30
                ORDER BY p.posted_at::timestamptz DESC
                LIMIT %s
                """,
                (since.isoformat(), limit),
            )
            cols = [d.name for d in cur.description]
            buf: list[dict] = []
            for row in cur:
                buf.append(dict(zip(cols, row)))
                if len(buf) >= BATCH_SIZE:
                    yield buf
                    buf = []
            if buf:
                yield buf


def load_blocklist(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT pattern FROM blocklist")
        return {r[0].lower() for r in cur}


def is_blocked(acct: str, blocklist: set[str]) -> bool:
    """Return True if the account or its domain is blocked."""
    acct_lower = acct.lower()
    if acct_lower in blocklist:
        return True
    if "@" in acct_lower:
        domain = acct_lower.split("@", 1)[1]
        if domain in blocklist:
            return True
    return False


def parse_posted_at(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        v = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(v)
    except (ValueError, TypeError):
        return None


def insert_batch(conn: psycopg.Connection, posts: list[dict], embeddings) -> tuple[int, int]:
    """Insert a batch of posts. Returns (inserted, skipped)."""
    inserted = 0
    skipped = 0
    with conn.cursor() as cur:
        for post, emb in zip(posts, embeddings):
            try:
                content = post["content"] or ""
                content_hash = hashlib.md5(content.encode()).digest()
                posted_at = parse_posted_at(post.get("posted_at"))
                if posted_at is None:
                    skipped += 1
                    continue
                # The schema has partitions only for 2026-04 and 2026-05 right now.
                # Skip posts outside that range — we'll add partitions on demand.
                if posted_at.year != 2026 or posted_at.month not in (4, 5):
                    skipped += 1
                    continue

                tags = post.get("tags") or []
                if isinstance(tags, str):
                    # v1 stored tags as JSON-encoded text
                    import json
                    try:
                        tags = json.loads(tags)
                    except Exception:
                        tags = []

                cur.execute(
                    """
                    INSERT INTO posts (
                        uri, url, author_acct, content, content_hash,
                        tags, media_count, favourites_count, reblogs_count,
                        posted_at, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        post["url"],
                        post["url"],
                        post["author_acct"],
                        content,
                        content_hash,
                        tags,
                        int(post.get("media_count") or 0),
                        int(post.get("favourites_count") or 0),
                        int(post.get("reblogs_count") or 0),
                        posted_at,
                        emb.tolist(),
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
            except psycopg.errors.UniqueViolation:
                conn.rollback()
                skipped += 1
            except Exception as e:
                log.warning("Insert error: %s", e)
                conn.rollback()
                skipped += 1
        conn.commit()
    return inserted, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--since", default=None, help="ISO date, default 14 days ago")
    args = parser.parse_args()

    since = (
        datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        if args.since
        else datetime.now(timezone.utc) - timedelta(days=14)
    )

    init_pool()
    with get_conn() as conn:
        blocklist = load_blocklist(conn)
    log.info("Loaded %d blocklist patterns", len(blocklist))

    log.info("Streaming v1 posts since %s (limit %d)...", since, args.limit)
    total_inserted = 0
    total_skipped = 0
    total_blocked = 0
    total_seen = 0

    for batch in fetch_v1_posts(since, args.limit):
        # Filter blocklist
        kept = []
        for p in batch:
            total_seen += 1
            if is_blocked(p["author_acct"] or "", blocklist):
                total_blocked += 1
            else:
                kept.append(p)
        if not kept:
            continue

        # Embed all kept posts at once
        contents = [p["content"] for p in kept]
        embeddings = embed_batch(contents)
        assert embeddings.shape == (len(kept), EMBEDDING_DIM)

        with get_conn() as conn:
            inserted, skipped = insert_batch(conn, kept, embeddings)
        total_inserted += inserted
        total_skipped += skipped

        log.info(
            "Progress: seen=%d inserted=%d skipped=%d blocked=%d",
            total_seen,
            total_inserted,
            total_skipped,
            total_blocked,
        )

    log.info(
        "DONE: seen=%d inserted=%d skipped=%d blocked=%d",
        total_seen,
        total_inserted,
        total_skipped,
        total_blocked,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
