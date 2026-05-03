"""Pull posts from hashtag timelines across multiple instances.

This is what actually fills the digest. Home timeline is whatever Tim follows;
this hits topic hashtags directly. The same hashtag pulled from 3 different
instances surfaces a diverse population.

Strategy:
    For each (instance, hashtag) pair:
      GET https://{instance}/api/v1/timelines/tag/{hashtag}?limit=40
    Embed all, ingest with media, dedup by content_hash.

Quality filters applied at insert time:
    - content >= 80 chars after HTML strip
    - not predominantly a URL (link-only posts skipped)
    - not from blocklisted domain/account
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone

import httpx

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch
from fedi_studio.workers.pull_home import slim_media, strip_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pull_tags")

# Tim's interest hashtags. Lower-case, as Mastodon hashtag API expects.
TAGS = [
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
]

# Instances to query for these hashtags. Mix of large and niche.
INSTANCES = [
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
]


def looks_like_link_only(content: str) -> bool:
    """Posts that are 70%+ URL by length are link-only - skip."""
    if not content:
        return True
    urls = re.findall(r"https?://\S+", content)
    if not urls:
        return False
    url_chars = sum(len(u) for u in urls)
    return url_chars > len(content) * 0.6


def normalize_acct(account: dict, fallback_instance: str) -> str:
    """Format username@instance regardless of source."""
    acct = account.get("acct") or account.get("username") or ""
    if "@" not in acct:
        host = fallback_instance.replace("https://", "").replace("http://", "").rstrip("/")
        acct = f"{acct}@{host}"
    return acct


def fetch_tag_page(client: httpx.Client, instance: str, tag: str) -> list[dict]:
    """Single page of public tag timeline. No auth required."""
    url = f"{instance}/api/v1/timelines/tag/{tag}"
    try:
        r = client.get(url, params={"limit": 40}, timeout=15)
        if r.status_code != 200:
            log.warning("%s tag/%s: HTTP %d", instance, tag, r.status_code)
            return []
        return r.json() or []
    except Exception as e:
        log.warning("%s tag/%s: %s", instance, tag, e)
        return []


def insert_post(cur, post: dict, embedding, fallback_instance: str) -> bool:
    content_html = post.get("content") or ""
    content = strip_html(content_html)
    if len(content) < 80:
        return False
    if looks_like_link_only(content):
        return False

    posted_at_str = post.get("created_at")
    if not posted_at_str:
        return False
    posted_at = datetime.fromisoformat(posted_at_str.replace("Z", "+00:00"))
    # Only ingest into existing partitions
    if posted_at.year != 2026 or posted_at.month not in (3, 4, 5, 6):
        return False

    content_hash = hashlib.md5(content.encode()).digest()
    tags = [t["name"] for t in (post.get("tags") or [])]
    media = slim_media(post.get("media_attachments") or [])
    account = post.get("account") or {}
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
            favourites_count = GREATEST(posts.favourites_count, EXCLUDED.favourites_count),
            reblogs_count = GREATEST(posts.reblogs_count, EXCLUDED.reblogs_count),
            media_attachments = EXCLUDED.media_attachments,
            local_id = COALESCE(posts.local_id, EXCLUDED.local_id),
            account_avatar = COALESCE(posts.account_avatar, EXCLUDED.account_avatar),
            account_display_name = COALESCE(posts.account_display_name, EXCLUDED.account_display_name)
        """,
        (
            post.get("uri") or post.get("url"),
            post.get("url"),
            normalize_acct(account, fallback_instance),
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
            None,  # local_id only valid for posts pulled from holm.community
            json.dumps(media),
            account.get("avatar_static") or account.get("avatar"),
            account.get("display_name") or "",
        ),
    )
    return cur.rowcount > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-tag", type=int, default=40)
    args = parser.parse_args()

    init_pool()

    # Load blocklist once
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pattern FROM blocklist")
        blocklist = {r[0].lower() for r in cur}
    log.info("Blocklist: %d patterns", len(blocklist))

    client = httpx.Client(follow_redirects=True, timeout=15)

    # Phase 1: collect raw posts across all (instance, tag) pairs
    raw: list[tuple[dict, str]] = []
    seen_uris: set[str] = set()
    for instance in INSTANCES:
        for tag in TAGS:
            page = fetch_tag_page(client, instance, tag)
            for post in page:
                # Reblog? Use the original
                if post.get("reblog"):
                    post = post["reblog"]
                uri = post.get("uri") or post.get("url")
                if not uri or uri in seen_uris:
                    continue
                acct = (post.get("account") or {}).get("acct") or ""
                # Skip blocklisted domains
                if "@" in acct:
                    domain = acct.split("@", 1)[1].lower()
                    if domain in blocklist:
                        continue
                if instance.replace("https://", "") in blocklist:
                    continue
                seen_uris.add(uri)
                raw.append((post, instance))
            time.sleep(0.05)  # gentle on remote servers
        log.info("After %s: %d unique candidate posts", instance, len(raw))

    log.info("Total unique candidates: %d", len(raw))
    if not raw:
        return 0

    # Phase 2: embed
    log.info("Embedding %d posts...", len(raw))
    contents = [strip_html(p.get("content") or "") for p, _ in raw]
    embeddings = embed_batch(contents)
    assert embeddings.shape[1] == EMBEDDING_DIM

    # Phase 3: insert with savepoints (one bad row doesn't poison the batch)
    inserted = 0
    skipped = 0
    with get_conn() as conn:
        for (post, source), emb in zip(raw, embeddings):
            with conn.cursor() as cur:
                cur.execute("SAVEPOINT row_sp")
                try:
                    if insert_post(cur, post, emb, source):
                        inserted += 1
                    else:
                        skipped += 1
                    cur.execute("RELEASE SAVEPOINT row_sp")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT row_sp")
                    log.warning("Row aborted: %s", e)
                    skipped += 1
        conn.commit()

    log.info("DONE: %d inserted, %d skipped (filtered)", inserted, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
