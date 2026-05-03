"""Continuous v1 -> v2 sync worker.

Watches v1.fedi_discover.posts and incrementally copies new rows into
v2.fedi_studio.posts (with embeddings). Long-lived process: every 60 seconds
pulls a batch of fresh v1 posts past the cursor, embeds them, inserts.

Cursor lives in v2 in table `v2_import_cursor` (created on first run).

Hard rules:
  - Read-only on v1. We only SELECT from v1.posts.
  - No Mastodon API calls. No follow/like/boost.
  - Filter blocklisted authors/domains.
  - Skip content < 30 chars (too short to embed meaningfully).

Run:
    python -m fedi_studio.workers.sync_v1_to_v2
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import psycopg

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sync_v1_to_v2")

V1_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon",
)

CURSOR_KEY = "v1_posts"
BATCH_LIMIT = 500
SLEEP_SECONDS = 60
RECENT_WINDOW = "24 hours"  # only sync posts seen in the last 24h to avoid backfill load

# v2 has partitions for 2025-11 through 2026-06 (inclusive). Refuse to insert
# anything outside that window — v1 has bogus posted_at values reaching back to
# year 0535, which would otherwise create useless empty partitions in v2.
PARTITION_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PARTITION_END = datetime(2027, 1, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------
# Cursor management
# ------------------------------------------------------------------

def ensure_cursor_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS v2_import_cursor (
                key         TEXT PRIMARY KEY,
                last_id     BIGINT NOT NULL DEFAULT 0,
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            "INSERT INTO v2_import_cursor (key, last_id) VALUES (%s, 0) ON CONFLICT (key) DO NOTHING",
            (CURSOR_KEY,),
        )
    conn.commit()


def read_cursor(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT last_id FROM v2_import_cursor WHERE key = %s", (CURSOR_KEY,))
        row = cur.fetchone()
        return int(row[0]) if row else 0


def write_cursor(conn: psycopg.Connection, last_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE v2_import_cursor
               SET last_id = %s, updated_at = NOW()
             WHERE key = %s AND %s > last_id
            """,
            (last_id, CURSOR_KEY, last_id),
        )
    conn.commit()


# ------------------------------------------------------------------
# Partitioning safety net
# ------------------------------------------------------------------

_partitions_known: set[str] = set()


def ensure_partition_for(conn: psycopg.Connection, posted_at: datetime) -> bool:
    """Lazily create monthly partitions when a post date falls outside the existing
    set. Returns True if the partition exists (or was created), False if it could
    not be created."""
    name = f"posts_{posted_at.year:04d}_{posted_at.month:02d}"
    if name in _partitions_known:
        return True
    try:
        # Compute first day of month and first day of next month
        year, month = posted_at.year, posted_at.month
        if month == 12:
            ny, nm = year + 1, 1
        else:
            ny, nm = year, month + 1
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF posts "
                f"FOR VALUES FROM ('{year:04d}-{month:02d}-01') TO ('{ny:04d}-{nm:02d}-01')"
            )
        conn.commit()
        _partitions_known.add(name)
        return True
    except Exception as e:
        conn.rollback()
        log.warning("could not ensure partition %s: %s", name, e)
        return False


# ------------------------------------------------------------------
# Fetch / filter / insert
# ------------------------------------------------------------------

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
    try:
        v = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(v)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def parse_tags(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, str)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [t for t in parsed if isinstance(t, str)]
        except Exception:
            return []
    return []


def fetch_v1_batch(after_id: int, limit: int) -> list[dict]:
    """Pull fresh posts from v1 with id > after_id.

    Date filter: posted_at MUST land inside the v2 partition window
    [PARTITION_START, PARTITION_END). v1 has bogus posted_at values from
    year 0535 etc; without this filter we end up creating dozens of empty
    pre-2025 partitions in v2.
    """
    sql = f"""
        SELECT
            id, url, author_acct, content, tags,
            favourites_count, reblogs_count, media_count,
            posted_at
        FROM posts
        WHERE id > %s
          AND posted_at::timestamptz >= %s
          AND posted_at::timestamptz <  %s
          AND content IS NOT NULL
          AND length(content) > 30
        ORDER BY id ASC
        LIMIT %s
        -- (note: 24h first_seen_at filter intentionally removed so historical v1 posts get backfilled
        -- after partition window extension. Cursor handles incremental progress.)
    """
    with psycopg.connect(V1_DSN, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (after_id, PARTITION_START.isoformat(), PARTITION_END.isoformat(), limit),
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur]


def insert_into_v2(conn: psycopg.Connection, posts: list[dict], embeddings) -> tuple[int, int]:
    """Insert filtered posts into v2 with savepoints. Returns (inserted, skipped)."""
    inserted = 0
    skipped = 0
    with conn.cursor() as cur:
        for post, emb in zip(posts, embeddings):
            content = post["content"] or ""
            posted_at = parse_posted_at(post.get("posted_at"))
            if posted_at is None:
                skipped += 1
                continue
            # Defense in depth: SQL filter already excludes out-of-window rows,
            # but a misparsed timestamp could still slip through. Refuse to
            # create new partitions outside the v2 window.
            if posted_at < PARTITION_START or posted_at >= PARTITION_END:
                skipped += 1
                continue
            if not ensure_partition_for(conn, posted_at):
                skipped += 1
                continue

            content_hash = hashlib.md5(content.encode()).digest()
            tags = parse_tags(post.get("tags"))

            cur.execute("SAVEPOINT s")
            try:
                cur.execute(
                    """
                    INSERT INTO posts (
                        uri, url, author_acct, content, content_hash,
                        tags, media_count, favourites_count, reblogs_count,
                        posted_at, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (uri, posted_at) DO UPDATE SET
                        favourites_count = EXCLUDED.favourites_count,
                        reblogs_count    = EXCLUDED.reblogs_count,
                        media_count      = EXCLUDED.media_count
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
                # Capture rowcount BEFORE RELEASE SAVEPOINT (which clobbers it).
                row_changed = cur.rowcount
                cur.execute("RELEASE SAVEPOINT s")
                if row_changed > 0:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT s")
                log.warning("row insert error: %s", e)
                skipped += 1
        conn.commit()
    return inserted, skipped


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

_running = True


def _stop(signum, frame):
    global _running
    _running = False
    log.info("received signal %s, stopping after current cycle", signum)


def run_cycle(blocklist: set[str]) -> tuple[int, int, int, int]:
    """One pull/embed/insert cycle. Returns (fetched, kept, inserted, last_id)."""
    with get_conn() as conn:
        cursor_value = read_cursor(conn)

    fetched = fetch_v1_batch(cursor_value, BATCH_LIMIT)
    if not fetched:
        return 0, 0, 0, cursor_value

    last_id = max(int(p["id"]) for p in fetched)

    # Filter blocklist
    kept: list[dict] = []
    blocked = 0
    for p in fetched:
        acct = p.get("author_acct") or ""
        if is_blocked(acct, blocklist):
            blocked += 1
            continue
        kept.append(p)

    if not kept:
        with get_conn() as conn:
            write_cursor(conn, last_id)
        return len(fetched), 0, 0, last_id

    # Embed — but skip if SYNC_SKIP_EMBED=1, which lets the score_all worker
    # do embedding+scoring later. That ~10x's sync throughput when v1 is
    # producing way faster than we can embed.
    if os.environ.get("SYNC_SKIP_EMBED", "0") == "1":
        # NULL embedding placeholder (numpy zeros). insert_into_v2 expects an
        # ndarray; score_all later overwrites with a real embedding when it
        # scores the post.
        import numpy as _np
        embeddings = _np.zeros((len(kept), EMBEDDING_DIM), dtype=_np.float32)
    else:
        contents = [p["content"] for p in kept]
        embeddings = embed_batch(contents)
        assert embeddings.shape == (len(kept), EMBEDDING_DIM), (
            f"embedding shape {embeddings.shape} != expected ({len(kept)}, {EMBEDDING_DIM})"
        )

    # Insert
    with get_conn() as conn:
        inserted, _ = insert_into_v2(conn, kept, embeddings)
        write_cursor(conn, last_id)

    return len(fetched), len(kept), inserted, last_id


def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    init_pool()

    with get_conn() as conn:
        ensure_cursor_table(conn)
        # Cache the existing partition list so we don't try to create them
        with conn.cursor() as cur:
            cur.execute(
                "SELECT inhrelid::regclass::text FROM pg_inherits "
                "WHERE inhparent = 'posts'::regclass"
            )
            for (name,) in cur:
                _partitions_known.add(name)
        starting = read_cursor(conn)
        blocklist = load_blocklist(conn)

    log.info(
        "sync_v1_to_v2 starting: cursor=%s blocklist=%d partitions=%d",
        starting,
        len(blocklist),
        len(_partitions_known),
    )

    cycle = 0
    while _running:
        cycle += 1
        t0 = time.time()
        try:
            fetched, kept, inserted, last_id = run_cycle(blocklist)
            elapsed = time.time() - t0
            log.info(
                "cycle %d: fetched=%d kept=%d inserted=%d last_id=%s elapsed=%.1fs",
                cycle, fetched, kept, inserted, last_id, elapsed,
            )
        except Exception as e:
            log.exception("cycle %d failed: %s", cycle, e)

        # Reload blocklist every 30 cycles in case it changed
        if cycle % 30 == 0:
            try:
                with get_conn() as conn:
                    blocklist = load_blocklist(conn)
                log.info("reloaded blocklist (%d patterns)", len(blocklist))
            except Exception as e:
                log.warning("blocklist reload failed: %s", e)

        # Sleep, but break early if signaled
        slept = 0
        while _running and slept < SLEEP_SECONDS:
            time.sleep(1)
            slept += 1

    log.info("sync_v1_to_v2 stopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
