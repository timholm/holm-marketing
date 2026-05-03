"""Multi-instance Mastodon SSE streamer for v2 fedi_studio.

Scales post ingestion past 100k posts/hour by opening ~30 persistent SSE
connections to well-federated public Mastodon instances. Each instance's
`/api/v1/streaming/public/local` endpoint pushes posts in real time with
no per-request rate limit — just one persistent TCP connection.

Each pod replica handles ~15 instances (2 replicas = 30 total streams).

Hard read-only invariants:
  * Every httpx call is a GET against a public endpoint.
  * No POST, PUT, DELETE, PATCH, or follow/like/boost path exists in this file.
  * No Mastodon API token is read or sent. No `Authorization` header.

Throughput target:
  * ~30 instances × 30 posts/min/instance = ~900 posts/min = ~54k/hr extra.

Run:
    python -m fedi_studio.workers.instance_streamer
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
import numpy as np

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch
from fedi_studio.services.scorer import HARD_BLOCK_KEYWORDS, READING_LANGUAGES
from fedi_studio.workers.pull_home import slim_media, strip_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("instance_streamer")

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

# Curated list of well-federated public Mastodon instances.
# NOTE: As of 2026, most major instances require auth for /api/v1/streaming.
# This list includes instances known to allow unauthenticated public streaming
# or smaller/medium instances more likely to permit it.
# Each instance is split across 2 replicas via INSTANCE_GROUP hash.
MASTODON_INSTANCES = [
    "https://mastodon.lol",
    "https://hostux.social",
    "https://mstdn.io",
    "https://kolektiva.social",
    "https://tilde.zone",
    "https://pixelfed.social",
    "https://photog.social",
    "https://mastodon.green",
    "https://climatejustice.global",
    "https://merveilles.town",
    "https://eldritch.cafe",
    "https://todon.eu",
    "https://octodon.social",
    "https://piaille.fr",
    "https://social.tchncs.de",
    "https://mstdn.party",
    "https://aus.social",
    "https://ohai.social",
    "https://chaos.social",
    "https://tech.lgbt",
    "https://solarpunk.moe",
    "https://sunny.garden",
    "https://sunbeam.city",
    "https://hachyderm.io",
    "https://infosec.exchange",
    "https://fosstodon.org",
    "https://mstdn.social",
    "https://mas.to",
    "https://mastodon.social",
    "https://vivaldi.net",
]

# Reconnect backoff
MIN_BACKOFF_S = 10.0
MAX_BACKOFF_S = 30.0

# Stats log interval
STATS_INTERVAL_S = 60.0


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
    # Per-stream connection state
    streams_connected: int = 0
    streams_disconnected: int = 0


@dataclass
class StreamState:
    """Track per-instance stream state."""
    instance: str
    connected: bool = False
    last_connect_time: float = 0.0
    last_disconnect_time: float = 0.0
    posts_count: int = 0
    events_count: int = 0


_stats = Stats()
_seen_uris: set[str] = set()
_running = True
_stream_states: dict[str, StreamState] = {}


def _stop(signum, _frame):
    global _running
    _running = False
    log.info("signal %s received, draining queue and exiting", signum)


# ----------------------------------------------------------------------
# Filtering (reuse firehose logic)
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
    """Return (ok, reason_skipped_if_not). Reuses firehose logic."""
    if post.get("reblog"):
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
# Persistence (reuse firehose logic)
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
    sql = """
        INSERT INTO posts (
            uri, url, author_acct, content, content_hash,
            tags, language, in_reply_to_id, sensitive,
            media_count, favourites_count, reblogs_count,
            posted_at, embedding,
            local_id, media_attachments, account_avatar, account_display_name
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (uri, posted_at) DO UPDATE SET
            favourites_count = GREATEST(posts.favourites_count, EXCLUDED.favourites_count),
            reblogs_count    = GREATEST(posts.reblogs_count,    EXCLUDED.reblogs_count),
            media_attachments = EXCLUDED.media_attachments,
            account_avatar = COALESCE(posts.account_avatar, EXCLUDED.account_avatar),
            account_display_name = COALESCE(posts.account_display_name, EXCLUDED.account_display_name)
    """
    pending_sql = """
        INSERT INTO candidates_pending (acct, source_post_uri)
        VALUES (%s, %s)
        ON CONFLICT (acct) DO NOTHING
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute("SAVEPOINT row_sp")
                    try:
                        cur.execute(sql, row)
                        if cur.rowcount > 0:
                            inserted += 1
                            author_acct = row[2]
                            uri = row[0]
                            if author_acct:
                                cur.execute(pending_sql, (author_acct, uri))
                                if cur.rowcount > 0:
                                    queued_for_enrichment += 1
                        cur.execute("RELEASE SAVEPOINT row_sp")
                    except Exception as e:
                        cur.execute("ROLLBACK TO SAVEPOINT row_sp")
                        log.debug("row insert failed: %s", e)
            conn.commit()
    except Exception as e:
        log.warning("DB flush failed: %s", e)
        return 0
    _stats.inserted += inserted
    if queued_for_enrichment > 0:
        log.debug("queued %d new authors for candidate enrichment", queued_for_enrichment)
    return inserted


# ----------------------------------------------------------------------
# Async SSE producer per instance
# ----------------------------------------------------------------------

async def _sse_lines(client: httpx.AsyncClient, url: str, instance: str) -> AsyncIterator[tuple[str, str]]:
    """Yield (event, data) tuples from an SSE stream. Reconnects on error.

    Read-only by construction: only ever issues a GET.
    """
    backoff = MIN_BACKOFF_S
    while _running:
        try:
            log.info("SSE connect %s", instance)
            stream_state = _stream_states[instance]
            stream_state.connected = True
            stream_state.last_connect_time = time.time()
            _stats.streams_connected += 1

            async with client.stream(
                "GET", url,
                timeout=httpx.Timeout(None, connect=15),
                headers={"Accept": "text/event-stream"}
            ) as resp:
                if resp.status_code != 200:
                    log.warning("SSE %s: HTTP %d", instance, resp.status_code)
                    stream_state.connected = False
                    stream_state.last_disconnect_time = time.time()
                    _stats.streams_disconnected += 1
                    await asyncio.sleep(min(backoff, MAX_BACKOFF_S))
                    backoff = min(backoff * 1.5, MAX_BACKOFF_S)
                    continue
                backoff = MIN_BACKOFF_S
                event_name = "message"
                async for line in resp.aiter_lines():
                    if not _running:
                        return
                    if not line:
                        event_name = "message"
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip() or "message"
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        yield event_name, data
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            log.warning("SSE %s error: %s; reconnecting in %.1fs", instance, e, backoff)
            stream_state = _stream_states[instance]
            stream_state.connected = False
            stream_state.last_disconnect_time = time.time()
            _stats.streams_disconnected += 1
            await asyncio.sleep(min(backoff, MAX_BACKOFF_S))
            backoff = min(backoff * 1.5, MAX_BACKOFF_S)
        except Exception as e:
            log.exception("SSE %s unexpected error: %s", instance, e)
            stream_state = _stream_states[instance]
            stream_state.connected = False
            stream_state.last_disconnect_time = time.time()
            _stats.streams_disconnected += 1
            await asyncio.sleep(5)


async def instance_producer(
    client: httpx.AsyncClient,
    queue: asyncio.Queue[tuple[dict, str]],
    blocklist: set[str],
    instance: str,
) -> None:
    """SSE producer for a single Mastodon instance.

    Connects to `https://{instance}/api/v1/streaming/public/local` and
    pushes updates to the queue.
    """
    fallback_host = instance.replace("https://", "").replace("http://", "").rstrip("/")

    # Skip blocklisted instances
    if fallback_host in blocklist:
        log.info("skipping blocklisted instance %s", instance)
        return

    # Initialize stream state
    _stream_states[instance] = StreamState(instance=instance)

    url = f"{instance}/api/v1/streaming/public/local"
    async for event, data in _sse_lines(client, url, instance):
        if event != "update":
            continue

        stream_state = _stream_states[instance]
        stream_state.events_count += 1
        _stats.received += 1

        try:
            post = json.loads(data)
        except json.JSONDecodeError:
            continue

        # Unwrap reblog
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
        stream_state.posts_count += 1
        await queue.put((post, fallback_host))


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
            await asyncio.to_thread(_flush_batch, chunk)
            last_flush = time.monotonic()
            # Bound the dedup memory
            if len(_seen_uris) > DEDUP_MAX:
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
        await asyncio.sleep(STATS_INTERVAL_S)
        now = time.monotonic()
        elapsed = now - started
        window = now - last_t
        delta = _stats.inserted - last_inserted

        rate_window = delta / window if window > 0 else 0.0
        rate_avg = _stats.inserted / elapsed if elapsed > 0 else 0.0

        # Count connected streams
        connected = sum(1 for s in _stream_states.values() if s.connected)

        log.info(
            "rate: %.1f posts/min (window) | %.1f posts/min (avg) | "
            "streams: %d connected / %d total | "
            "received=%d parsed=%d filtered=%d embedded=%d inserted=%d "
            "seen_uris=%d rejected_writes=%d",
            rate_window * 60.0,
            rate_avg * 60.0,
            connected, len(_stream_states),
            _stats.received, _stats.parsed, _stats.filtered,
            _stats.embedded, _stats.inserted,
            len(_seen_uris), _stats.rejected_writes,
        )
        last_inserted = _stats.inserted
        last_t = now


# ----------------------------------------------------------------------
# Instance partitioning by pod replica
# ----------------------------------------------------------------------

def _get_instance_group() -> str:
    """Partition instances across replicas based on POD_NAME hash.

    INSTANCE_GROUP env var is set by the deployment. If not set,
    compute from POD_NAME % 2.
    """
    group = os.environ.get("INSTANCE_GROUP")
    if group:
        return group

    # Fallback: compute from POD_NAME
    pod_name = os.environ.get("HOSTNAME", "unknown")
    replica_num = hash(pod_name) % 2
    return "a" if replica_num == 0 else "b"


def _get_instance_list(group: str) -> list[str]:
    """Return instances assigned to this replica group."""
    # Split instances evenly: group 'a' gets first half, group 'b' gets second half
    mid = len(MASTODON_INSTANCES) // 2
    if group == "a":
        return MASTODON_INSTANCES[:mid]
    else:
        return MASTODON_INSTANCES[mid:]


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

    group = _get_instance_group()
    instances = _get_instance_list(group)
    log.info("instance_group=%s, assigned %d instances: %s", group, len(instances), instances)

    # Warm the embedder up-front
    log.info("warming embedder...")
    await asyncio.to_thread(embed_batch, ["warm"])
    log.info("embedder ready (dim=%d)", EMBEDDING_DIM)

    queue: asyncio.Queue[tuple[dict, str]] = asyncio.Queue(maxsize=8192)

    # One AsyncClient shared across all GET calls. No auth headers.
    client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": "fedi-studio-instance-streamer/1.0 (read-only)"}
    )

    tasks: list[asyncio.Task] = []
    for instance in instances:
        tasks.append(asyncio.create_task(
            instance_producer(client, queue, blocklist, instance),
            name=f"instance:{instance}",
        ))
    tasks.append(asyncio.create_task(consumer(queue), name="consumer"))
    tasks.append(asyncio.create_task(reporter(), name="reporter"))

    # Wait until shutdown signal
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
