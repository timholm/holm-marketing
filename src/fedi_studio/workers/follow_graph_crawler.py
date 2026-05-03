"""Follow-graph crawler: expand candidates via high-scoring seed accounts.

READ-ONLY worker that seeds from candidates with score >= threshold and
crawls their following/followers, inserting new candidates_pending rows
to break the 50% lookup_failed waste rate from v1 crawling dead authors.

Seed selection: one seed at a time, ordered by score DESC, skipping those
already graph_crawled_at IS NOT NULL.

Per seed:
1. Fetch holm_account_id if not cached: GET /api/v1/accounts/lookup?acct={acct}
   - Store to candidates.holm_account_id (optimize future runs)
2. Concurrently fetch both:
   - GET /api/v1/accounts/{id}/following?limit=80 (paginate up to 5 pages = 400)
   - GET /api/v1/accounts/{id}/followers?limit=80 (paginate up to 5 pages = 400)
3. For each unique acct in results (skip tim@holm.community):
   - INSERT INTO candidates_pending (acct, source_post_uri, holm_account_id)
   - ON CONFLICT (acct) DO NOTHING (deduplicate across runs)
4. UPDATE candidates.graph_crawled_at = NOW()
5. Sleep 2s between seeds (polite rate to holm.community)

Rate limit: ~11 API calls per seed, target 100 seeds/5min ≈ 1100 calls total.
Auth: Use MASTODON_TOKEN for 1500/5min limit vs 300 unauth.

Error handling:
- 429 Retry-After backoff
- 404/410 mark as crawled (don't retry dead profiles)
- Network errors: log and continue

Run:
    python -m fedi_studio.workers.follow_graph_crawler
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import psycopg

from fedi_studio.models.db import get_dsn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("follow_graph_crawler")

FEDI_STUDIO_DSN = get_dsn()
MASTODON_URL = os.environ.get("MASTODON_URL", "https://holm.community").rstrip("/")
MASTODON_TOKEN = os.environ.get("MASTODON_TOKEN", "")

HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "12.0"))
USER_AGENT = "fedi-studio-follow-graph-crawler/1.0 (read-only; tim@holm.community)"
SEED_BATCH_SIZE = int(os.environ.get("SEED_BATCH_SIZE", "1"))
SLEEP_BETWEEN_SEEDS_S = float(os.environ.get("SLEEP_BETWEEN_SEEDS_S", "2.0"))
MAX_PAGES_PER_RELATION = int(os.environ.get("MAX_PAGES_PER_RELATION", "5"))

# Tim's instance account for filtering
TIM_ACCT = "tim@holm.community"


async def _get_json(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> tuple[Any, dict]:
    """GET with backoff for 429/5xx. Returns (json, headers) or (None, {}) on error."""
    backoff = 1.0
    for attempt in range(6):
        try:
            r = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", backoff))
                log.warning(
                    "429 Retry-After %.1fs on %s", wait, url.split("?")[0]
                )
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, 60)
                continue
            if r.status_code in (404, 410):
                log.warning("404/410 on %s — skipping", url.split("?")[0])
                return None, {}
            if r.status_code >= 500:
                log.warning(
                    "5xx %d on %s — backoff %.1fs", r.status_code, url.split("?")[0], backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            return r.json(), r.headers
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            log.warning("request error on %s (%s) — backoff %.1fs", url.split("?")[0], type(e).__name__, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
    log.error("exhausted retries on %s", url.split("?")[0])
    return None, {}


async def _paginate(
    client: httpx.AsyncClient, url: str, max_pages: int = 5
) -> set[str]:
    """Paginate endpoint, collecting all unique accts. Returns set of acct strings."""
    accts = set()
    next_url = url
    for page_num in range(max_pages):
        if not next_url:
            break
        data, headers = await _get_json(client, next_url)
        if data is None:
            break
        if not isinstance(data, list):
            log.warning("unexpected response format (not list) from %s", url.split("?")[0])
            break
        for account in data:
            if isinstance(account, dict) and "acct" in account:
                accts.add(account["acct"])
        # Look for Link header for next page
        next_url = None
        link_header = headers.get("Link", "")
        if link_header:
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    # Extract URL from <url>; rel="next"
                    try:
                        next_url = part.split("<")[1].split(">")[0]
                    except IndexError:
                        pass
                    break
    return accts


async def _fetch_following_and_followers(
    client: httpx.AsyncClient, account_id: str
) -> tuple[set[str], set[str]]:
    """Concurrently fetch following and followers for an account."""
    following_url = f"{MASTODON_URL}/api/v1/accounts/{account_id}/following"
    followers_url = f"{MASTODON_URL}/api/v1/accounts/{account_id}/followers"

    following_task = _paginate(client, following_url, MAX_PAGES_PER_RELATION)
    followers_task = _paginate(client, followers_url, MAX_PAGES_PER_RELATION)

    following, followers = await asyncio.gather(following_task, followers_task)
    return following, followers


async def _get_account_id(client: httpx.AsyncClient, acct: str) -> str | None:
    """Lookup account by acct, return mastodon id or None."""
    url = f"{MASTODON_URL}/api/v1/accounts/lookup"
    data, _ = await _get_json(client, url, params={"acct": acct})
    if data and isinstance(data, dict):
        return data.get("id")
    return None


def _get_next_seed(conn: psycopg.Connection) -> tuple[int, str, str | None] | None:
    """Get next unprocessed high-scoring seed. Returns (id, acct, holm_account_id) or None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, acct, holm_account_id
            FROM candidates
            WHERE reviewed = FALSE AND graph_crawled_at IS NULL
            ORDER BY score DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if row:
            return row
    return None


def _update_holm_account_id(conn: psycopg.Connection, candidate_id: int, holm_account_id: str) -> None:
    """Cache holm_account_id in candidates table."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE candidates SET holm_account_id = %s WHERE id = %s",
            (holm_account_id, candidate_id),
        )
        conn.commit()


def _insert_pending_candidates(
    conn: psycopg.Connection, accts: set[str], seed_acct: str, holm_account_id: str | None
) -> int:
    """Insert new accts into candidates_pending. Returns count of new rows."""
    if not accts:
        return 0
    source_uri = f"graph:from:{seed_acct}"
    new_count = 0
    with conn.cursor() as cur:
        for acct in accts:
            if acct == TIM_ACCT:
                continue
            try:
                cur.execute(
                    """
                    INSERT INTO candidates_pending (acct, source_post_uri, holm_account_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (acct) DO NOTHING
                    """,
                    (acct, source_uri, holm_account_id),
                )
                if cur.rowcount > 0:
                    new_count += 1
            except Exception as e:
                log.warning("failed to insert %s: %s", acct, e)
        conn.commit()
    return new_count


def _mark_crawled(conn: psycopg.Connection, candidate_id: int) -> None:
    """Mark candidate as crawled."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE candidates SET graph_crawled_at = NOW() WHERE id = %s",
            (candidate_id,),
        )
        conn.commit()


async def crawl_seed(
    client: httpx.AsyncClient, conn: psycopg.Connection, seed: tuple[int, str, str | None]
) -> dict:
    """Crawl one seed: lookup if needed, fetch following/followers, insert pending."""
    seed_id, seed_acct, holm_account_id = seed

    # Step 1: get holm_account_id if not cached
    if not holm_account_id:
        holm_account_id = await _get_account_id(client, seed_acct)
        if not holm_account_id:
            log.warning("seed=%s lookup failed", seed_acct)
            _mark_crawled(conn, seed_id)
            return {"seed": seed_acct, "lookup_failed": True}
        _update_holm_account_id(conn, seed_id, holm_account_id)

    # Step 2: fetch both lists concurrently
    following, followers = await _fetch_following_and_followers(client, holm_account_id)

    # Step 3: deduplicate and insert
    all_accts = following | followers
    new_count = _insert_pending_candidates(conn, all_accts, seed_acct, holm_account_id)

    # Step 4: mark as crawled
    _mark_crawled(conn, seed_id)

    return {
        "seed": seed_acct,
        "followings": len(following),
        "followers": len(followers),
        "new_pending": new_count,
    }


async def main() -> None:
    """Main loop: crawl seeds one at a time."""
    if not MASTODON_TOKEN:
        log.error("MASTODON_TOKEN not set")
        return

    headers = {"Authorization": f"Bearer {MASTODON_TOKEN}", "User-Agent": USER_AGENT}

    conn = psycopg.connect(FEDI_STUDIO_DSN)
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        seed_count = 0
        while True:
            seed = _get_next_seed(conn)
            if not seed:
                log.info("no more seeds, sleeping 60s")
                await asyncio.sleep(60)
                continue

            result = await crawl_seed(client, conn, seed)
            if "lookup_failed" in result:
                log.info("seed=%s lookup_failed=true", result["seed"])
            else:
                log.info(
                    "seed=%s followings=%d followers=%d new_pending=%d",
                    result["seed"],
                    result["followings"],
                    result["followers"],
                    result["new_pending"],
                )
            seed_count += 1

            # Polite rate: sleep between seeds
            await asyncio.sleep(SLEEP_BETWEEN_SEEDS_S)


if __name__ == "__main__":
    asyncio.run(main())
