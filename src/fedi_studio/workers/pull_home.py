"""Pull recent posts from Tim's Mastodon home timeline.

This is a transitional worker for Phase 2 MVP. Once the Rust SSE listener is
running, the home timeline updates flow live. For now we pull pages of recent
posts so the digest has something to show.

Strategy:
    GET /api/v1/timelines/home?limit=40&max_id=... walking back in time
    until we have N posts or we hit something already in v2 PG.

Stores in fedi_studio.posts with embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone

import httpx

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pull_home")

MASTODON_URL = os.environ.get("MASTODON_URL", "https://holm.community")
MASTODON_TOKEN = os.environ.get("MASTODON_TOKEN", "")


def strip_html(html: str) -> str:
    """Quick-and-dirty HTML to text. Good enough for embedding."""
    import re
    # Replace common breaks with spaces/newlines
    s = re.sub(r"<br\s*/?>", "\n", html)
    s = re.sub(r"</p>", "\n", s)
    # Strip remaining tags
    s = re.sub(r"<[^>]+>", "", s)
    # Decode common entities
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    s = s.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return s.strip()


def fetch_home_pages(client: httpx.Client, total: int) -> list[dict]:
    """Walk the home timeline backwards, returning up to `total` posts."""
    posts: list[dict] = []
    max_id: str | None = None
    while len(posts) < total:
        params: dict[str, str | int] = {"limit": 40}
        if max_id:
            params["max_id"] = max_id
        r = client.get(f"{MASTODON_URL}/api/v1/timelines/home", params=params, timeout=30)
        if r.status_code != 200:
            log.warning("Home timeline %s: %s", r.status_code, r.text[:200])
            break
        page = r.json()
        if not page:
            log.info("Empty page, stopping")
            break
        posts.extend(page)
        max_id = page[-1]["id"]
        log.info("Fetched %d posts (total %d, max_id=%s)", len(page), len(posts), max_id)
    return posts[:total]


def normalize_acct(account: dict) -> str:
    """Format as username@instance even for local accounts."""
    acct = account.get("acct") or ""
    if "@" not in acct:
        # Local user, derive instance from server URL
        host = MASTODON_URL.replace("https://", "").replace("http://", "").rstrip("/")
        acct = f"{acct}@{host}"
    return acct


def slim_media(attachments: list[dict]) -> list[dict]:
    """Keep only what we need to render media: type, url, preview, alt, dimensions."""
    out = []
    for m in attachments or []:
        out.append(
            {
                "type": m.get("type"),
                "url": m.get("url"),
                "preview_url": m.get("preview_url"),
                "remote_url": m.get("remote_url"),
                "description": m.get("description"),
                "blurhash": m.get("blurhash"),
                "meta": m.get("meta") or {},
            }
        )
    return out


def insert_post(cur, post: dict, embedding) -> bool:
    """Insert a single post. Return True if inserted."""
    import json as _json

    content_html = post.get("content") or ""
    content = strip_html(content_html)
    if len(content) < 10:
        return False

    posted_at = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
    content_hash = hashlib.md5(content.encode()).digest()
    tags = [t["name"] for t in (post.get("tags") or [])]
    media = slim_media(post.get("media_attachments") or [])
    account = post.get("account") or {}

    try:
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
                local_id = EXCLUDED.local_id,
                media_attachments = EXCLUDED.media_attachments,
                account_avatar = EXCLUDED.account_avatar,
                account_display_name = EXCLUDED.account_display_name,
                media_count = EXCLUDED.media_count,
                favourites_count = EXCLUDED.favourites_count,
                reblogs_count = EXCLUDED.reblogs_count
            """,
            (
                post.get("uri") or post.get("url"),
                post.get("url"),
                normalize_acct(account),
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
                post.get("id"),
                _json.dumps(media),
                account.get("avatar_static") or account.get("avatar"),
                account.get("display_name") or "",
            ),
        )
        return cur.rowcount > 0
    except Exception as e:
        log.warning("Insert failed: %s", e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=400)
    args = parser.parse_args()

    if not MASTODON_TOKEN:
        log.error("Set MASTODON_TOKEN env var")
        return 1

    client = httpx.Client(
        headers={"Authorization": f"Bearer {MASTODON_TOKEN}"},
        follow_redirects=True,
    )

    log.info("Fetching up to %d posts from %s home timeline", args.total, MASTODON_URL)
    raw = fetch_home_pages(client, args.total)
    log.info("Got %d raw posts", len(raw))
    if not raw:
        return 0

    # Skip reblogs (they appear as posts with reblog field non-null)
    originals = [p["reblog"] if p.get("reblog") else p for p in raw]
    contents = [strip_html(p.get("content") or "") for p in originals]

    log.info("Embedding...")
    embeddings = embed_batch(contents)
    assert embeddings.shape[1] == EMBEDDING_DIM

    init_pool()
    inserted = 0
    skipped = 0
    with get_conn() as conn:
        for post, emb in zip(originals, embeddings):
            with conn.cursor() as cur:
                cur.execute("SAVEPOINT row_sp")
                try:
                    if insert_post(cur, post, emb):
                        inserted += 1
                    else:
                        skipped += 1
                    cur.execute("RELEASE SAVEPOINT row_sp")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT row_sp")
                    log.warning("Row aborted: %s", e)
                    skipped += 1
        conn.commit()

    log.info("DONE: %d inserted, %d skipped", inserted, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
