"""Pull Tim's own posts from holm.community into v2 every 10 minutes.

Without this, /weekly's #SolarPunkSunday section can't surface Tim's media because
the firehose only pulls hashtag timelines and federated streams (which usually
don't include his own outbox unless he's used those tags).

Read-only: GET /api/v1/accounts/{tim_id}/statuses. No engagement actions.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import psycopg

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import embed_batch
from fedi_studio.workers.pull_home import slim_media, strip_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pull_tim_outbox")

MASTODON_URL = os.environ.get("MASTODON_URL", "https://holm.community").rstrip("/")
MASTODON_TOKEN = os.environ.get("MASTODON_TOKEN", "")
POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "600"))  # 10 min
USER_AGENT = "fedi-studio-pull-tim-outbox/1.0 (read-only)"

_running = True


def _stop(*_):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def fetch_tim_statuses(client: httpx.AsyncClient = None) -> list[dict]:
    """Synchronously fetch Tim's recent statuses (last 80 = ~2 weeks worth)."""
    if not MASTODON_TOKEN:
        log.error("MASTODON_TOKEN not set")
        return []
    headers = {
        "Authorization": f"Bearer {MASTODON_TOKEN}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    with httpx.Client(headers=headers, timeout=20.0) as cl:
        # Resolve tim's id
        r = cl.get(f"{MASTODON_URL}/api/v1/accounts/verify_credentials")
        if r.status_code != 200:
            log.error("verify_credentials %s", r.status_code)
            return []
        tim_id = r.json()["id"]
        # Pull last 80 with replies/reblogs included
        all_statuses: list[dict] = []
        max_id = None
        for _ in range(2):  # 2 pages of 40 = 80
            params = {"limit": 40, "exclude_replies": "false", "exclude_reblogs": "true"}
            if max_id:
                params["max_id"] = max_id
            r = cl.get(f"{MASTODON_URL}/api/v1/accounts/{tim_id}/statuses", params=params)
            if r.status_code != 200:
                break
            page = r.json()
            if not page:
                break
            all_statuses.extend(page)
            max_id = page[-1].get("id")
        return all_statuses


def upsert_status(cur: psycopg.Cursor, post: dict, embedding) -> bool:
    """Upsert a single Tim status into v2 posts. Returns True if inserted."""
    raw = post.get("content") or ""
    content = strip_html(raw)
    posted_at_str = post.get("created_at")
    if not posted_at_str:
        return False
    posted_at = datetime.fromisoformat(posted_at_str.replace("Z", "+00:00"))
    if posted_at.tzinfo is None:
        posted_at = posted_at.replace(tzinfo=timezone.utc)
    uri = post.get("uri")
    url = post.get("url")
    if not uri:
        return False

    account = post.get("account") or {}
    author_acct = account.get("acct") or ""
    if author_acct and "@" not in author_acct:
        author_acct = f"{author_acct}@{MASTODON_URL.replace('https://','').replace('http://','').rstrip('/')}"

    import hashlib
    content_hash = hashlib.md5(content.encode("utf-8")).digest() if content else b""

    tags_raw = post.get("tags") or []
    tags = [t.get("name") for t in tags_raw if t.get("name")]
    media = slim_media(post.get("media_attachments") or [])
    import json as _json
    cur.execute(
        """
        INSERT INTO posts (
            uri, url, author_acct, content, content_hash,
            tags, language, in_reply_to_id, sensitive,
            media_count, favourites_count, reblogs_count,
            posted_at, embedding,
            local_id, media_attachments, account_avatar, account_display_name
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (uri, posted_at) DO UPDATE SET
            content = EXCLUDED.content,
            favourites_count = GREATEST(posts.favourites_count, EXCLUDED.favourites_count),
            reblogs_count    = GREATEST(posts.reblogs_count,    EXCLUDED.reblogs_count),
            media_attachments = EXCLUDED.media_attachments
        """,
        (
            uri, url, author_acct, content, content_hash,
            tags, post.get("language"), post.get("in_reply_to_id"),
            bool(post.get("sensitive")),
            len(media), int(post.get("favourites_count") or 0),
            int(post.get("reblogs_count") or 0),
            posted_at, list(embedding.astype(float)),
            post.get("id"), _json.dumps(media),
            account.get("avatar"), account.get("display_name"),
        ),
    )
    return cur.rowcount > 0


def run_once() -> int:
    init_pool()
    statuses = fetch_tim_statuses()
    if not statuses:
        log.info("no statuses returned")
        return 0
    contents = [strip_html(s.get("content") or "") for s in statuses]
    if not any(contents):
        return 0
    embeddings = embed_batch([c or " " for c in contents])
    inserted = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for s, emb in zip(statuses, embeddings):
                try:
                    if upsert_status(cur, s, emb):
                        inserted += 1
                except Exception as e:
                    log.debug("upsert failed for %s: %s", s.get("id"), e)
        conn.commit()
    return inserted


def main() -> None:
    log.info("pull_tim_outbox starting (poll every %ds)", POLL_INTERVAL_S)
    while _running:
        try:
            n = run_once()
            log.info("pulled %d new tim posts", n)
        except Exception as e:
            log.warning("cycle failed: %s", e)
        # interruptible sleep
        for _ in range(POLL_INTERVAL_S):
            if not _running:
                break
            time.sleep(1)
    log.info("clean shutdown")


if __name__ == "__main__":
    main()
