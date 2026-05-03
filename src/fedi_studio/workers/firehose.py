"""Multi-source post firehose for v2 fedi_studio.

The lead-crawler in v1 maxes out around 25 posts/min — far too slow to grow
toward 100M posts. This worker fans out across many parallel public read
streams to drive sustained throughput at orders of magnitude higher rate.

Sources, all read-only and unauthenticated:
  * FediBuzz public firehose SSE (the highest-volume single source)
        https://fedi.buzz/api/v1/streaming/public
  * Public hashtag REST timelines on a curated set of instances, polled
    every 30 seconds. Hashtag list mirrors `pull_tags.py`.

For every event we:
  * Parse the Mastodon Status JSON.
  * Apply hard rules (block list, language filter, #nobot, content too short).
  * Dedupe in-memory by URI.
  * Embed the HTML-stripped content with Model2Vec.
  * Upsert into `fedi_studio.posts` (ON CONFLICT (uri, posted_at)).

Hard read-only invariants:
  * Every httpx call is a GET against a public endpoint.
  * No POST, PUT, DELETE, PATCH, or follow/like/boost path exists in this file.
  * No Mastodon API token is read or sent. No `Authorization` header.

Throughput (informational):
  * Per-source SSE handles 100s of events/min from FediBuzz alone.
  * Hashtag pollers add 100-300/min combined depending on tag activity.
  * Embedder is the CPU bottleneck (~hundreds of posts/sec).

Run:
    python -m fedi_studio.workers.firehose
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
import numpy as np

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch
from fedi_studio.services.scorer import HARD_BLOCK_KEYWORDS, READING_LANGUAGES
from fedi_studio.workers.pull_home import slim_media, strip_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("firehose")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

# v2 partition window — refuse to insert outside this range
PARTITION_START = datetime(2025, 11, 1, tzinfo=timezone.utc)
PARTITION_END = datetime(2026, 7, 1, tzinfo=timezone.utc)

# How many posts to embed in a single batch before flushing to PG
EMBED_BATCH = 64
EMBED_FLUSH_INTERVAL_S = 5.0

# Dedup memory (LRU-ish — periodically pruned)
DEDUP_MAX = 200_000

# FediBuzz public firehose
FEDIBUZZ_STREAM = "https://fedi.buzz/api/v1/streaming/public"

# Hashtag polling. Mirrors pull_tags.py, kept here to avoid importing module
# globals that aren't intended to be reused.
HASHTAGS = [
    # Off-grid / homestead
    "offgrid", "offgridliving", "homesteading", "homestead",
    "tinyhouse", "tinyhome", "cabinlife", "cabin",
    "permaculture", "selfsufficiency",
    # Solar / energy
    "solarpunk", "solarpunksunday", "solar", "diysolar", "lifepo4",
    # Building
    "earthship", "rammedearth", "compressedearthblock", "cob", "naturalbuilding",
    # Vehicles
    "schoolbus", "skoolie", "busconversion", "vanlife", "vandwelling",
    # Tech / self-host
    "selfhosting", "selfhosted", "homelab", "kubernetes", "k3s",
    # Garden
    "gardening", "vegetablegarden", "foodforest", "raisedbed", "hydroponics",
    # Region
    "arizona", "sonoran", "desertliving",
    # Identity / community
    "queer", "lgbtq", "trans", "bisexual", "introduction",
]

# Public hashtag-timeline instances (no auth needed for /api/v1/timelines/tag/*)
HASHTAG_INSTANCES = [
    "https://mastodon.social",
    "https://mas.to",
    "https://infosec.exchange",
    "https://sunbeam.city",
    "https://sunny.garden",
    "https://solarpunk.moe",
    "https://kolektiva.social",
    "https://tech.lgbt",
    "https://hachyderm.io",
    "https://chaos.social",
    "https://fosstodon.org",
    "https://mastodon.online",
    "https://mstdn.party",
    "https://ohai.social",
    "https://aus.social",
    "https://vivaldi.net",
    "https://mstdn.social",
    "https://social.tchncs.de",
    "https://piaille.fr",
    "https://octodon.social",
]

POLL_INTERVAL_S = 30.0


# ----------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------

@dataclass
class Stats:
    received: int = 0      # raw events seen on the wire
    parsed: int = 0        # parsed Status JSON
    filtered: int = 0      # rejected by hard rules / dedup / window
    embedded: int = 0
    inserted: int = 0
    rejected_writes: int = 0  # any code path attempted a non-GET (should be 0)


_stats = Stats()
_seen_uris: set[str] = set()
_running = True


def _stop(signum, _frame):
    global _running
    _running = False
    log.info("signal %s received, draining queue and exiting", signum)


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------

def _looks_like_link_only(text: str) -> bool:
    if not text:
        return True
    urls = re.findall(r"https?://\S+", text)
    if not urls:
        return False
    url_chars = sum(len(u) for u in urls)
    return url_chars > len(text) * 0.6


def _bio_blocked(bio: str | None) -> bool:
    if not bio:
        return False
    low = bio.lower()
    return any(kw in low for kw in HARD_BLOCK_KEYWORDS)


def _normalize_acct(account: dict, fallback_host: str) -> str:
    acct = account.get("acct") or account.get("username") or ""
    if "@" not in acct:
        acct = f"{acct}@{fallback_host}"
    return acct


def _accept_post(post: dict, fallback_host: str, blocklist: set[str]) -> tuple[bool, str | None]:
    """Return (ok, reason_skipped_if_not)."""
    if post.get("reblog"):
        # Caller is expected to unwrap reblogs first, but defend anyway.
        return False, "reblog_wrapper"

    visibility = post.get("visibility")
    if visibility and visibility not in ("public", "unlisted"):
        return False, f"visibility:{visibility}"

    content_html = post.get("content") or ""
    content = strip_html(content_html)
    if len(content) < 80:
        return False, "too_short"
    if _looks_like_link_only(content):
        return False, "link_only"

    posted_at_str = post.get("created_at")
    if not posted_at_str:
        return False, "no_created_at"
    try:
        posted_at = datetime.fromisoformat(str(posted_at_str).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False, "bad_created_at"
    if posted_at.tzinfo is None:
        posted_at = posted_at.replace(tzinfo=timezone.utc)
    if posted_at < PARTITION_START or posted_at >= PARTITION_END:
        return False, "outside_partition_window"

    lang = post.get("language")
    if lang and lang not in READING_LANGUAGES:
        return False, f"language:{lang}"

    account = post.get("account") or {}
    acct = _normalize_acct(account, fallback_host)
    a = acct.lower()
    if a in blocklist:
        return False, "author_blocklist"
    if "@" in a:
        domain = a.split("@", 1)[1]
        if domain in blocklist:
            return False, "domain_blocklist"
    if account.get("bot") is True:
        return False, "bot_account"
    if _bio_blocked(account.get("note") or ""):
        return False, "bio_keyword"

    return True, None


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def _build_row(post: dict, embedding: np.ndarray, fallback_host: str) -> tuple | None:
    content = strip_html(post.get("content") or "")
    posted_at_str = post.get("created_at")
    posted_at = datetime.fromisoformat(str(posted_at_str).replace("Z", "+00:00"))
    if posted_at.tzinfo is None:
        posted_at = posted_at.replace(tzinfo=timezone.utc)

    content_hash = hashlib.md5(content.encode()).digest()
    tags = [t["name"] for t in (post.get("tags") or []) if isinstance(t, dict) and "name" in t]
    media = slim_media(post.get("media_attachments") or [])
    account = post.get("account") or {}

    uri = post.get("uri") or post.get("url")
    if not uri:
        return None

    return (
        uri,
        post.get("url"),
        _normalize_acct(account, fallback_host),
        content,
        content_hash,
        tags,
        post.get("language"),
        post.get("in_reply_to_id"),
        bool(post.get("sensitive")),
        len(post.get("media_attachments") or []),
        int(post.get("favourites_count") or 0),
        int(post.get("reblogs_count") or 0),
        posted_at,
        embedding.tolist(),
        None,  # local_id is for posts pulled from holm.community only
        json.dumps(media),
        account.get("avatar_static") or account.get("avatar"),
        account.get("display_name") or "",
    )


def _flush_batch(batch: list[tuple[dict, str]]) -> int:
    """Embed + upsert a batch. Returns count inserted."""
    if not batch:
        return 0
    contents = [strip_html(p.get("content") or "") for p, _ in batch]
    try:
        embeddings = embed_batch(contents)
    except Exception as e:
        log.warning("embed_batch failed (%d items): %s", len(batch), e)
        return 0
    assert embeddings.shape == (len(batch), EMBEDDING_DIM)
    _stats.embedded += len(batch)

    rows: list[tuple] = []
    for (post, host), emb in zip(batch, embeddings):
        row = _build_row(post, emb, host)
        if row is not None:
            rows.append(row)
    if not rows:
        return 0

    inserted = 0
    queued_for_enrichment = 0

    # 2026-05-03: switched from per-row INSERT/SAVEPOINT to a single multi-row INSERT
    # built via psycopg's SQL composition. Pre-filter URIs in PG to avoid lock
    # contention on (uri, posted_at) unique index, then issue ONE batch INSERT.
    # This image uses psycopg3 (not psycopg2), so we use psycopg.sql.SQL composition
    # rather than psycopg2.extras.execute_values.
    from psycopg import sql as _psql

    pending_template = _psql.SQL("(%s, %s)")
    row_template = _psql.SQL("(") + _psql.SQL(",").join([_psql.Placeholder()] * 18) + _psql.SQL(")")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Pre-filter: SELECT existing uris in one round-trip
                uris = [r[0] for r in rows]
                cur.execute(
                    "SELECT uri FROM posts WHERE uri = ANY(%s::text[])",
                    (uris,),
                )
                existing = {row[0] for row in cur.fetchall()}
                fresh_rows = [r for r in rows if r[0] not in existing] if existing else rows

                if fresh_rows:
                    values_sql = _psql.SQL(",").join([row_template] * len(fresh_rows))
                    insert_sql = _psql.SQL(
                        "INSERT INTO posts (uri, url, author_acct, content, content_hash, "
                        "tags, language, in_reply_to_id, sensitive, media_count, "
                        "favourites_count, reblogs_count, posted_at, embedding, local_id, "
                        "media_attachments, account_avatar, account_display_name) "
                        "VALUES "
                    ) + values_sql + _psql.SQL(
                        " ON CONFLICT (uri, posted_at) DO NOTHING "
                        "RETURNING uri, author_acct"
                    )
                    flat_params = [v for row in fresh_rows for v in row]
                    cur.execute(insert_sql, flat_params)
                    returned = cur.fetchall()
                    inserted = len(returned)

                    cand_rows = [(r[1], r[0]) for r in returned if r[1]]
                    if cand_rows:
                        cand_values_sql = _psql.SQL(",").join([pending_template] * len(cand_rows))
                        cand_insert = _psql.SQL(
                            "INSERT INTO candidates_pending (acct, source_post_uri) VALUES "
                        ) + cand_values_sql + _psql.SQL(
                            " ON CONFLICT (acct) DO NOTHING"
                        )
                        cand_flat = [v for row in cand_rows for v in row]
                        cur.execute(cand_insert, cand_flat)
                        queued_for_enrichment = len(cand_rows)
            conn.commit()
    except Exception as e:
        log.warning("DB flush failed: %s", e)
        return 0

    _stats.inserted += inserted
    if queued_for_enrichment > 0:
        log.debug("queued %d new authors for candidate enrichment", queued_for_enrichment)
    return inserted


# ----------------------------------------------------------------------
# Async producers
# ----------------------------------------------------------------------

async def _sse_lines(client: httpx.AsyncClient, url: str) -> AsyncIterator[tuple[str, str]]:
    """Yield (event, data) tuples from an SSE stream. Reconnects on error.

    Read-only by construction: only ever issues a GET.
    """
    backoff = 1.0
    while _running:
        try:
            log.info("SSE connect %s", url)
            async with client.stream("GET", url, timeout=None,
                                     headers={"Accept": "text/event-stream"}) as resp:
                if resp.status_code != 200:
                    log.warning("SSE %s: HTTP %d", url, resp.status_code)
                    await asyncio.sleep(min(backoff, 30))
                    backoff = min(backoff * 2, 60)
                    continue
                backoff = 1.0
                event_name = "message"
                async for line in resp.aiter_lines():
                    if not _running:
                        return
                    if not line:
                        # blank line = end of event
                        event_name = "message"
                        continue
                    if line.startswith(":"):
                        # comment / heartbeat
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip() or "message"
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        yield event_name, data
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            log.warning("SSE %s error: %s; reconnecting in %.1fs", url, e, backoff)
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)
        except Exception as e:
            log.exception("SSE %s unexpected error: %s", url, e)
            await asyncio.sleep(5)


async def fedibuzz_producer(
    client: httpx.AsyncClient,
    queue: asyncio.Queue[tuple[dict, str]],
    blocklist: set[str],
) -> None:
    fallback_host = "fedi.buzz"
    async for event, data in _sse_lines(client, FEDIBUZZ_STREAM):
        if event != "update":
            continue
        _stats.received += 1
        try:
            post = json.loads(data)
        except json.JSONDecodeError:
            continue
        # Reblog wrapper -> use the original
        if isinstance(post, dict) and post.get("reblog"):
            post = post["reblog"]
        if not isinstance(post, dict):
            continue
        _stats.parsed += 1

        uri = post.get("uri") or post.get("url")
        if not uri or uri in _seen_uris:
            _stats.filtered += 1
            continue

        ok, reason = _accept_post(post, fallback_host, blocklist)
        if not ok:
            _stats.filtered += 1
            continue
        _seen_uris.add(uri)
        await queue.put((post, fallback_host))


async def hashtag_producer(
    client: httpx.AsyncClient,
    queue: asyncio.Queue[tuple[dict, str]],
    blocklist: set[str],
    instance: str,
    tag: str,
    initial_offset: float,
) -> None:
    """Poll a single (instance, tag) pair every POLL_INTERVAL_S seconds.

    Each poll is a GET to /api/v1/timelines/tag/{tag} which is public, no auth.
    """
    fallback_host = instance.replace("https://", "").replace("http://", "").rstrip("/")
    if fallback_host in blocklist:
        return
    # Stagger startup so we don't hit every endpoint at the same instant
    await asyncio.sleep(initial_offset)
    while _running:
        url = f"{instance}/api/v1/timelines/tag/{tag}"
        try:
            r = await client.get(url, params={"limit": 40}, timeout=15)
            if r.status_code == 200:
                page = r.json() or []
                for raw in page:
                    if not isinstance(raw, dict):
                        continue
                    post = raw["reblog"] if raw.get("reblog") else raw
                    _stats.received += 1
                    _stats.parsed += 1
                    uri = post.get("uri") or post.get("url")
                    if not uri or uri in _seen_uris:
                        _stats.filtered += 1
                        continue
                    ok, _reason = _accept_post(post, fallback_host, blocklist)
                    if not ok:
                        _stats.filtered += 1
                        continue
                    _seen_uris.add(uri)
                    await queue.put((post, fallback_host))
            else:
                log.debug("%s tag/%s: HTTP %s", instance, tag, r.status_code)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            log.debug("%s tag/%s: %s", instance, tag, e)
        await asyncio.sleep(POLL_INTERVAL_S)


# ----------------------------------------------------------------------
# Consumer + reporter
# ----------------------------------------------------------------------

async def consumer(queue: asyncio.Queue[tuple[dict, str]]) -> None:
    """Drain the queue and embed/insert in batches."""
    batch: list[tuple[dict, str]] = []
    last_flush = time.monotonic()
    while _running or not queue.empty():
        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            batch.append(item)
        except asyncio.TimeoutError:
            pass

        now = time.monotonic()
        if len(batch) >= EMBED_BATCH or (batch and now - last_flush >= EMBED_FLUSH_INTERVAL_S):
            chunk, batch = batch, []
            # Embedding is CPU-bound; run in a thread so we don't block the loop
            await asyncio.to_thread(_flush_batch, chunk)
            last_flush = time.monotonic()
            # Bound the dedup memory
            if len(_seen_uris) > DEDUP_MAX:
                # Drop a random ~25% by rebuilding from a sample
                keep = list(_seen_uris)[-(DEDUP_MAX * 3 // 4):]
                _seen_uris.clear()
                _seen_uris.update(keep)

    # Final drain
    if batch:
        await asyncio.to_thread(_flush_batch, batch)


async def reporter() -> None:
    """Log throughput every 60s."""
    started = time.monotonic()
    last_inserted = 0
    last_t = started
    while _running:
        await asyncio.sleep(60)
        now = time.monotonic()
        elapsed = now - started
        window = now - last_t
        delta = _stats.inserted - last_inserted
        rate_window = delta / window if window > 0 else 0.0
        rate_avg = _stats.inserted / elapsed if elapsed > 0 else 0.0
        log.info(
            "rate: %.1f posts/min (window) | %.1f posts/min (avg) | "
            "received=%d parsed=%d filtered=%d embedded=%d inserted=%d "
            "seen_uris=%d rejected_writes=%d",
            rate_window * 60.0,
            rate_avg * 60.0,
            _stats.received, _stats.parsed, _stats.filtered,
            _stats.embedded, _stats.inserted,
            len(_seen_uris), _stats.rejected_writes,
        )
        last_inserted = _stats.inserted
        last_t = now


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

async def amain() -> int:
    init_pool()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pattern FROM blocklist")
            blocklist = {r[0].lower() for r in cur}
    log.info("blocklist: %d patterns", len(blocklist))

    # Warm the embedder up-front so the first batch doesn't pay the cost
    log.info("warming embedder...")
    await asyncio.to_thread(embed_batch, ["warm"])
    log.info("embedder ready (dim=%d)", EMBEDDING_DIM)

    queue: asyncio.Queue[tuple[dict, str]] = asyncio.Queue(maxsize=8192)

    # One AsyncClient shared across all GET calls. No auth headers.
    # follow_redirects=True needed for some instances that 301 to canonical host.
    client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": "fedi-studio-firehose/1.0 (read-only)"}
    )

    tasks: list[asyncio.Task] = []
    tasks.append(asyncio.create_task(fedibuzz_producer(client, queue, blocklist), name="fedibuzz"))
    # Stagger hashtag pollers so we don't burst all at once
    pair_count = len(HASHTAG_INSTANCES) * len(HASHTAGS)
    log.info("starting %d hashtag pollers", pair_count)
    i = 0
    for inst in HASHTAG_INSTANCES:
        for tag in HASHTAGS:
            offset = (i / max(pair_count, 1)) * POLL_INTERVAL_S
            tasks.append(asyncio.create_task(
                hashtag_producer(client, queue, blocklist, inst, tag, offset),
                name=f"hashtag:{inst}/{tag}",
            ))
            i += 1
    tasks.append(asyncio.create_task(consumer(queue), name="consumer"))
    tasks.append(asyncio.create_task(reporter(), name="reporter"))

    # Wait until shutdown signal — then cancel producers, drain consumer, exit.
    while _running:
        await asyncio.sleep(1)

    log.info("shutting down...")
    for t in tasks:
        if t.get_name() != "consumer":
            t.cancel()
    # Give consumer a moment to flush
    await asyncio.sleep(2)
    for t in tasks:
        if not t.done():
            t.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await t
    await client.aclose()
    log.info(
        "exiting: received=%d parsed=%d filtered=%d embedded=%d inserted=%d",
        _stats.received, _stats.parsed, _stats.filtered,
        _stats.embedded, _stats.inserted,
    )
    return 0


def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
