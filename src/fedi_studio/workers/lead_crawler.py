"""Crawl v1 profiles table (60K+ pre-filtered accounts) to feed v2 posts.

The v1 profiles table contains accounts discovered and pre-filtered to match Tim's
interests (6-month crawl of quality hashtags + public timelines). This worker pulls
recent statuses from each profile into v2 posts, massively increasing candidate
discovery throughput.

Architecture:
  * Source: v1 `fedi_discover_full.profiles` table (~60K rows, pre-filtered)
  * Cursor: ordered by `last_crawled_at` (added via migration), batch of 200
  * Per-profile: extract mastodon_id from raw_data JSON, GET /accounts/{id}/statuses
  * Embed + upsert into v2 posts (same pipeline as firehose)
  * Concurrency: 30 simultaneous accounts, 1 req/sec per instance
  * Idempotent: restartable via cursor, skips already-crawled profiles

Hard read-only invariants:
  * Every httpx call is a GET against public endpoints
  * No POST, PUT, DELETE, PATCH anywhere
  * No Mastodon API token needed; all endpoints are public
  * No auth headers sent

Run:
    python -m fedi_studio.workers.lead_crawler
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import httpx
import numpy as np
import psycopg

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch
from fedi_studio.services.scorer import HARD_BLOCK_KEYWORDS, READING_LANGUAGES
from fedi_studio.workers.pull_home import slim_media, strip_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lead_crawler")

# Database connections
V1_DSN = os.environ.get(
    "V1_DSN", "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon"
)
V2_DSN = os.environ.get(
    "FEDI_STUDIO_DSN", "host=localhost port=30141 dbname=fedi_studio user=mastodon password=mastodon"
)

# Tuning
PARTITION_START = datetime(2025, 11, 1, tzinfo=timezone.utc)
PARTITION_END = datetime(2026, 7, 1, tzinfo=timezone.utc)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "200"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "30"))
EMBED_BATCH_SIZE = 64
EMBED_FLUSH_INTERVAL_S = 5.0
STATUSES_PER_ACCOUNT = int(os.environ.get("STATUSES_PER_ACCOUNT", "40"))

# Deep-pagination per profile. When > 1 we paginate via max_id to fetch up to
# MAX_PAGES_PER_ACCOUNT pages of STATUSES_PER_ACCOUNT statuses each
# (e.g. 8 pages × 40 = 320 posts/profile). Set to 300 for a 12k-post backfill.
# Each extra page is one more request against the same instance — respect rate limits.
MAX_PAGES_PER_ACCOUNT = int(os.environ.get("MAX_PAGES_PER_ACCOUNT", "1"))

# Per-instance rate limit: min seconds between requests to same instance
MIN_INTERVAL_PER_INSTANCE = 1.0

# Per-instance DB-backed cooldown (seconds): prevents parallel replicas from
# hammering the same instance
INSTANCE_COOLDOWN_S = float(os.environ.get("INSTANCE_COOLDOWN_S", "2.0"))

# High-403 threshold: skip instances that got >= this many 403s in v1
INSTANCE_403_THRESHOLD = 5

# Skip leads crawled in the last 7 days
MIN_CRAWL_INTERVAL_DAYS = 7

# Dedup
DEDUP_MAX = 200_000

# Shared state
@dataclass
class Stats:
    leads_processed: int = 0
    accounts_fetched: int = 0
    statuses_received: int = 0
    filtered: int = 0
    embedded: int = 0
    inserted: int = 0
    errors: int = 0
    rate_limit_429: int = 0
    throttle_skipped: int = 0

_stats = Stats()
_seen_uris: set[str] = set()
_running = True
_instance_last_hit: dict[str, float] = {}


def _stop(signum, _frame):
    global _running
    _running = False
    log.info("signal %s received, draining and exiting", signum)


def _looks_like_link_only(text: str) -> bool:
    if not text:
        return True
    import re
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
        None,
        json.dumps(media),
        account.get("avatar_static") or account.get("avatar"),
        account.get("display_name") or "",
    )


def _flush_batch(batch: list[tuple[dict, str]]) -> int:
    """Embed + upsert a batch. Returns count inserted.

    Performance fix (2026-05-02): replaced per-row SAVEPOINT/INSERT with
    pre-SELECT URI filter + single multi-VALUES batch INSERT. Cuts DB
    round-trips from O(N) per flush to ~3, and eliminates row-lock contention
    on the (uri, posted_at) unique index when 2 replicas flush concurrently.

    Mirrors the pattern used in tools/fedi/src/fedi/services/mass_crawler.py
    `_flush_posts`, adapted for psycopg3 (multi-VALUES sql.Composed instead
    of psycopg2.extras.execute_values).
    """
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

    # Build candidates_pending rows in lockstep with v2 rows
    cand_rows: list[tuple[str, str]] = []
    for row in rows:
        uri = row[0]
        author_acct = row[2]
        if author_acct and uri:
            cand_rows.append((author_acct.lower(), uri))

    inserted = 0
    queued_for_enrichment = 0

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Pre-filter URIs already present to avoid row-lock contention on
                # the (uri, posted_at) unique index. Belt-and-suspenders ON
                # CONFLICT DO NOTHING below handles any race.
                try:
                    uris = [r[0] for r in rows]
                    cur.execute(
                        "SELECT uri FROM posts WHERE uri = ANY(%s::text[])",
                        (uris,),
                    )
                    existing = {r[0] for r in cur.fetchall()}
                    if existing:
                        rows = [r for r in rows if r[0] not in existing]
                except Exception as _e:
                    log.warning("prefilter failed (continuing with all rows): %s", _e)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

                if rows:
                    # Single multi-VALUES INSERT. psycopg3 has no execute_values,
                    # so we build the VALUES list with sql.SQL placeholders.
                    placeholder_row = "(" + ", ".join(["%s"] * 18) + ")"
                    values_sql = ", ".join([placeholder_row] * len(rows))
                    flat_params: list = []
                    for r in rows:
                        flat_params.extend(r)
                    insert_sql = (
                        "INSERT INTO posts ("
                        "uri, url, author_acct, content, content_hash, "
                        "tags, language, in_reply_to_id, sensitive, "
                        "media_count, favourites_count, reblogs_count, "
                        "posted_at, embedding, "
                        "local_id, media_attachments, account_avatar, account_display_name"
                        ") VALUES " + values_sql + " "
                        "ON CONFLICT (uri, posted_at) DO NOTHING "
                        "RETURNING uri"
                    )
                    cur.execute(insert_sql, flat_params)
                    returned = cur.fetchall()
                    inserted = len(returned)

                    # Batch INSERT candidates_pending for all candidates we built;
                    # ON CONFLICT (acct) DO NOTHING dedups against existing rows.
                    if cand_rows:
                        cand_placeholder = "(" + ", ".join(["%s"] * 2) + ")"
                        cand_values_sql = ", ".join([cand_placeholder] * len(cand_rows))
                        cand_flat: list = []
                        for cr in cand_rows:
                            cand_flat.extend(cr)
                        cand_sql = (
                            "INSERT INTO candidates_pending (acct, source_post_uri) "
                            "VALUES " + cand_values_sql + " "
                            "ON CONFLICT (acct) DO NOTHING"
                        )
                        cur.execute(cand_sql, cand_flat)
                        queued_for_enrichment = cur.rowcount or 0
            conn.commit()
    except Exception as e:
        log.warning("DB flush failed: %s", e)
        return 0
    _stats.inserted += inserted
    if inserted > 0 or queued_for_enrichment > 0:
        log.info(
            "[batch-write] stored=%d candidates_queued=%d batch=%d",
            inserted, queued_for_enrichment, len(batch),
        )
    return inserted


async def _wait_for_instance_ratelimit(instance: str) -> None:
    """Enforce 1 req/sec per instance (local replica throttle)."""
    now = time.monotonic()
    last_hit = _instance_last_hit.get(instance, 0.0)
    elapsed = now - last_hit
    if elapsed < MIN_INTERVAL_PER_INSTANCE:
        await asyncio.sleep(MIN_INTERVAL_PER_INSTANCE - elapsed)
    _instance_last_hit[instance] = time.monotonic()


def try_acquire_instance_slot(v2_conn: psycopg.Connection, instance: str) -> bool:
    """Acquire a per-instance slot from Postgres-backed throttle.

    Returns True if we can proceed (either created row or cooldown elapsed).
    Returns False if another replica beat us to it recently (within INSTANCE_COOLDOWN_S).

    This is called before each HTTP request to ensure only one replica is hitting
    a given instance at a time, preventing 429 storms on popular instances.
    """
    try:
        with v2_conn.cursor() as cur:
            # Check if we already have a row and if cooldown has elapsed
            cur.execute(
                "SELECT last_hit_at FROM instance_throttle WHERE instance = %s",
                (instance,)
            )
            existing = cur.fetchone()

            if existing is None:
                # No row yet, insert it and proceed
                cur.execute(
                    "INSERT INTO instance_throttle (instance, last_hit_at) VALUES (%s, NOW())",
                    (instance,)
                )
                v2_conn.commit()
                return True

            # Row exists; check if cooldown has elapsed
            last_hit_at = existing[0]
            now = datetime.now(timezone.utc)
            if (now - last_hit_at).total_seconds() >= INSTANCE_COOLDOWN_S:
                # Cooldown elapsed, update and proceed
                cur.execute(
                    "UPDATE instance_throttle SET last_hit_at = NOW() WHERE instance = %s",
                    (instance,)
                )
                v2_conn.commit()
                return True

            # Cooldown not yet elapsed, skip this lead
            v2_conn.commit()
            return False
    except Exception as e:
        log.debug("instance_throttle check %s failed: %s", instance, e)
        # Fail open on DB errors — let the request go through
        return True


async def _fetch_account_id(
    client: httpx.AsyncClient, instance: str, acct_local: str, v2_conn: psycopg.Connection
) -> str | None:
    """Resolve acct (local user part) to mastodon_id via /accounts/lookup."""
    # Check Postgres-backed throttle to prevent parallel replicas from hammering
    if not try_acquire_instance_slot(v2_conn, instance):
        _stats.throttle_skipped += 1
        log.debug("%s lookup: skipped (throttled by another replica)", instance)
        return None

    try:
        await _wait_for_instance_ratelimit(instance)
        url = f"https://{instance}/api/v1/accounts/lookup"
        r = await client.get(url, params={"acct": acct_local}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return str(data.get("id"))
        elif r.status_code == 429:
            _stats.rate_limit_429 += 1
            log.debug("%s lookup: 429", instance)
        else:
            log.debug("%s lookup %s: HTTP %d", instance, acct_local, r.status_code)
    except Exception as e:
        log.debug("lookup %s@%s: %s", acct_local, instance, e)
    return None


async def _fetch_statuses(
    client: httpx.AsyncClient,
    instance: str,
    mastodon_id: str,
    blocklist: set[str],
    v2_conn: psycopg.Connection,
) -> list[tuple[dict, str]]:
    """Fetch last 40 statuses from an account."""
    # Check Postgres-backed throttle to prevent parallel replicas from hammering
    if not try_acquire_instance_slot(v2_conn, instance):
        _stats.throttle_skipped += 1
        log.debug("%s statuses: skipped (throttled by another replica)", instance)
        return []

    results: list[tuple[dict, str]] = []
    max_id: str | None = None
    pages_fetched = 0
    try:
        url = f"https://{instance}/api/v1/accounts/{mastodon_id}/statuses"
        for page in range(MAX_PAGES_PER_ACCOUNT):
            await _wait_for_instance_ratelimit(instance)
            params: dict[str, str | int] = {
                "limit": STATUSES_PER_ACCOUNT,
                "exclude_replies": "true",
                "exclude_reblogs": "true",
            }
            if max_id:
                params["max_id"] = max_id
            r = await client.get(url, params=params, timeout=15)
            if r.status_code == 429:
                _stats.rate_limit_429 += 1
                log.debug("%s statuses: 429 at page %d", instance, page)
                break
            if r.status_code != 200:
                log.debug("%s statuses %s: HTTP %d", instance, mastodon_id, r.status_code)
                break
            statuses = r.json() or []
            if not isinstance(statuses, list) or not statuses:
                break
            pages_fetched += 1
            _stats.statuses_received += len(statuses)
            new_max_id: str | None = None
            for status in statuses:
                if not isinstance(status, dict):
                    continue
                sid = status.get("id")
                if sid:
                    # Mastodon IDs are sortable strings; track the smallest seen
                    if new_max_id is None or str(sid) < new_max_id:
                        new_max_id = str(sid)
                ok, _reason = _accept_post(status, instance, blocklist)
                if not ok:
                    _stats.filtered += 1
                    continue
                uri = status.get("uri") or status.get("url")
                if uri and uri not in _seen_uris:
                    _seen_uris.add(uri)
                    results.append((status, instance))
            # If we got a partial page, this is the end of the timeline
            if len(statuses) < STATUSES_PER_ACCOUNT:
                break
            # Advance cursor; bail if we couldn't extract a max_id
            if new_max_id is None or new_max_id == max_id:
                break
            max_id = new_max_id
    except Exception as e:
        log.debug("statuses %s/%s: %s", instance, mastodon_id, e)
    if pages_fetched > 1:
        log.debug("%s/%s: pulled %d statuses across %d pages",
                  instance, mastodon_id, len(results), pages_fetched)
    return results


async def _process_lead(
    client: httpx.AsyncClient,
    profile: dict,
    v1_conn: psycopg.Connection,
    v2_conn: psycopg.Connection,
    blocklist: set[str],
    high_403_instances: set[str],
) -> None:
    """Process a single profile: fetch statuses, queue for embedding."""
    instance = profile["instance"]
    profile_id = profile["id"]
    acct = profile["acct"]

    if not instance:
        return

    # Skip blocked instances
    if instance in blocklist or instance in high_403_instances:
        return

    # Prefer the cached mastodon_id (from v1's profiles.raw_data->>'id'). It hits
    # the origin instance directly with no auth and no holm.community quota usage —
    # critical for parallel scaling without rate-limit collapse.
    mastodon_id = profile.get("mastodon_id")
    if not mastodon_id:
        # Fallback: resolve via /accounts/lookup on the ORIGIN instance (still
        # avoids holm.community). This path is only used for profiles that v1
        # never cached an id for.
        if "@" in acct:
            local_username = acct.split("@")[0]
        else:
            local_username = acct
        mastodon_id = await _fetch_account_id(client, instance, local_username, v2_conn)
        if not mastodon_id:
            log.debug("profile %d (%s@%s): failed to resolve mastodon_id", profile_id, acct, instance)
            return

    _stats.accounts_fetched += 1

    statuses = await _fetch_statuses(client, instance, mastodon_id, blocklist, v2_conn)
    if statuses:
        # Queue for embedding/insertion
        async with _embed_queue_lock:
            _embed_queue.extend(statuses)

    # Mark profile as crawled (only on success to allow retries on throttle)
    try:
        with v1_conn.cursor() as cur:
            cur.execute(
                "UPDATE profiles SET last_crawled_at = %s WHERE id = %s",
                (datetime.now(timezone.utc), profile_id),
            )
        v1_conn.commit()
    except Exception as e:
        log.debug("update profile %d: %s", profile_id, e)


# Global queue for posts to embed
_embed_queue: list[tuple[dict, str]] = []
_embed_queue_lock = asyncio.Lock()


async def consumer() -> None:
    """Drain the embed queue and flush in batches."""
    batch: list[tuple[dict, str]] = []
    last_flush = time.monotonic()
    while _running or _embed_queue:
        async with _embed_queue_lock:
            if _embed_queue:
                batch.extend(_embed_queue)
                _embed_queue.clear()

        now = time.monotonic()
        if len(batch) >= EMBED_BATCH_SIZE or (batch and now - last_flush >= EMBED_FLUSH_INTERVAL_S):
            chunk, batch = batch, []
            await asyncio.to_thread(_flush_batch, chunk)
            last_flush = time.monotonic()

            # Prune dedup memory
            if len(_seen_uris) > DEDUP_MAX:
                keep = list(_seen_uris)[-(DEDUP_MAX * 3 // 4) :]
                _seen_uris.clear()
                _seen_uris.update(keep)

        await asyncio.sleep(0.1)

    # Final drain
    if batch:
        await asyncio.to_thread(_flush_batch, batch)


async def crawler() -> None:
    """Main crawler loop: read profiles in batches, fetch statuses concurrently."""
    # Load blocklist
    blocklist: set[str] = set()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pattern FROM blocklist")
                blocklist = {r[0].lower() for r in cur}
    except Exception as e:
        log.warning("failed to load blocklist: %s", e)

    # Note: v1 instance_activity doesn't track 403s; we skip this filter
    high_403_instances: set[str] = set()

    log.info("blocklist: %d patterns, high-403: %d instances", len(blocklist), len(high_403_instances))

    client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": "fedi-studio-lead-crawler/1.0 (read-only)"},
    )

    try:
        while _running:
            # Read next batch of profiles from v1, ordered by last_crawled_at ASC
            # Prioritize profiles with cached mastodon_id (2x faster, no lookup needed)
            profiles_batch = []
            v1_conn = None
            try:
                v1_conn = psycopg.connect(V1_DSN, connect_timeout=10)
                with v1_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, acct, instance, is_active,
                               (raw_data::jsonb->>'id') as mastodon_id,
                               COALESCE(last_crawled_at, '1970-01-01') as last_crawled
                        FROM profiles
                        WHERE is_active = 1
                          AND (last_crawled_at IS NULL
                               OR last_crawled_at < NOW() - INTERVAL '%d days')
                        ORDER BY (raw_data::jsonb ? 'id') DESC, COALESCE(last_crawled_at, '1970-01-01') ASC
                        LIMIT %s
                        """
                        % (MIN_CRAWL_INTERVAL_DAYS, BATCH_SIZE)
                    )
                    profiles_batch = [
                        {
                            "id": r[0],
                            "acct": r[1],
                            "instance": r[2],
                            "is_active": r[3],
                            "mastodon_id": r[4],
                            "last_crawled": r[5],
                        }
                        for r in cur
                    ]
            except Exception as e:
                log.warning("failed to fetch profiles batch: %s", e)
                if v1_conn:
                    v1_conn.close()
                await asyncio.sleep(5)
                continue

            if not profiles_batch:
                log.info("no more profiles to crawl; sleeping 60s")
                if v1_conn:
                    v1_conn.close()
                await asyncio.sleep(60)
                continue

            _stats.leads_processed += len(profiles_batch)
            log.info("processing batch of %d profiles", len(profiles_batch))

            # Get a single v2 connection for the entire batch (throttle checks)
            v2_conn = None
            try:
                v2_conn = psycopg.connect(V2_DSN, connect_timeout=10)
            except Exception as e:
                log.warning("failed to connect to v2 db: %s", e)
                if v1_conn:
                    v1_conn.close()
                await asyncio.sleep(5)
                continue

            # Process profiles concurrently
            tasks = []
            for profile in profiles_batch:
                if not _running:
                    break
                task = asyncio.create_task(
                    _process_lead(client, profile, v1_conn, v2_conn, blocklist, high_403_instances)
                )
                tasks.append(task)

                # Limit concurrency
                if len(tasks) >= MAX_CONCURRENCY:
                    await asyncio.gather(*tasks)
                    tasks = []

            if tasks:
                await asyncio.gather(*tasks)

            if v2_conn:
                v2_conn.close()
            if v1_conn:
                v1_conn.close()

    finally:
        await client.aclose()


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
            "leads=%d accounts=%d statuses=%d filtered=%d embedded=%d inserted=%d "
            "errors=%d 429s=%d throttle_skipped=%d seen_uris=%d",
            rate_window * 60.0,
            rate_avg * 60.0,
            _stats.leads_processed,
            _stats.accounts_fetched,
            _stats.statuses_received,
            _stats.filtered,
            _stats.embedded,
            _stats.inserted,
            _stats.errors,
            _stats.rate_limit_429,
            _stats.throttle_skipped,
            len(_seen_uris),
        )
        last_inserted = _stats.inserted
        last_t = now


async def amain() -> int:
    init_pool()

    log.info("warming embedder...")
    await asyncio.to_thread(embed_batch, ["warm"])
    log.info("embedder ready (dim=%d)", EMBEDDING_DIM)

    tasks = [
        asyncio.create_task(crawler(), name="crawler"),
        asyncio.create_task(consumer(), name="consumer"),
        asyncio.create_task(reporter(), name="reporter"),
    ]

    while _running:
        await asyncio.sleep(1)

    log.info("shutting down...")
    for t in tasks:
        if t.get_name() != "consumer":
            t.cancel()
    await asyncio.sleep(2)
    for t in tasks:
        if not t.done():
            t.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await t

    log.info(
        "exiting: leads=%d statuses=%d filtered=%d embedded=%d inserted=%d errors=%d 429s=%d throttle_skipped=%d",
        _stats.leads_processed,
        _stats.statuses_received,
        _stats.filtered,
        _stats.embedded,
        _stats.inserted,
        _stats.errors,
        _stats.rate_limit_429,
        _stats.throttle_skipped,
    )
    return 0


def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
