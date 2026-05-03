"""Pull every interaction Tim is a party to from Mastodon.

Sources:
    GET /api/v1/notifications?types[]=mention&types[]=follow
    GET /api/v1/conversations
    GET /api/v1/timelines/home  -- to fetch parent context for replies

Stores in mentions table. Idempotent (notification_id is unique).

This is the foundation for relationship-tracking. Every mention/reply/DM
becomes a row Tim can review. The Ollama drafter then proposes replies for
in-interest topics only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import httpx

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.workers.pull_home import strip_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pull_mentions")

MASTODON_URL = os.environ.get("MASTODON_URL", "https://holm.community")
MASTODON_TOKEN = os.environ.get("MASTODON_TOKEN", "")


def fetch_notifications(client: httpx.Client, limit: int = 200) -> list[dict]:
    """Pull recent mention/follow notifications, paginating."""
    out: list[dict] = []
    max_id: str | None = None
    while len(out) < limit:
        params: dict[str, str | int | list[str]] = {
            "limit": 40,
            "types[]": ["mention", "follow"],
        }
        if max_id:
            params["max_id"] = max_id
        r = client.get(f"{MASTODON_URL}/api/v1/notifications", params=params, timeout=30)
        if r.status_code != 200:
            log.warning("notifications: HTTP %d", r.status_code)
            break
        page = r.json()
        if not page:
            break
        out.extend(page)
        max_id = page[-1]["id"]
        log.info("Fetched %d notifications (total %d)", len(page), len(out))
    return out[:limit]


def fetch_conversations(client: httpx.Client, limit: int = 100) -> list[dict]:
    """DMs."""
    out: list[dict] = []
    max_id: str | None = None
    while len(out) < limit:
        params: dict[str, str | int] = {"limit": 40}
        if max_id:
            params["max_id"] = max_id
        r = client.get(f"{MASTODON_URL}/api/v1/conversations", params=params, timeout=30)
        if r.status_code != 200:
            log.warning("conversations: HTTP %d", r.status_code)
            break
        page = r.json()
        if not page:
            break
        out.extend(page)
        max_id = page[-1]["id"]
    return out[:limit]


def fetch_status(client: httpx.Client, status_id: str) -> dict | None:
    """Fetch a single status (used to load Tim's parent post for context)."""
    r = client.get(f"{MASTODON_URL}/api/v1/statuses/{status_id}", timeout=15)
    if r.status_code != 200:
        return None
    return r.json()


def normalize_acct(account: dict) -> str:
    acct = account.get("acct") or ""
    if "@" not in acct:
        host = MASTODON_URL.replace("https://", "").replace("http://", "").rstrip("/")
        acct = f"{acct}@{host}"
    return acct


def insert_mention(cur, kind: str, status: dict, parent: dict | None, notification_id: str | None = None) -> bool:
    account = status.get("account") or {}
    content_html = status.get("content") or ""
    content = strip_html(content_html)
    created = datetime.fromisoformat(status["created_at"].replace("Z", "+00:00"))
    parent_snippet = ""
    in_reply_to_uri = None
    if parent:
        in_reply_to_uri = parent.get("uri") or parent.get("url")
        parent_snippet = strip_html(parent.get("content") or "")[:300]
    cur.execute(
        """
        INSERT INTO mentions (
            notification_id, kind, author_acct, author_display_name, author_avatar,
            post_uri, post_url, post_local_id, post_content, post_html,
            in_reply_to_uri, parent_content, visibility, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (notification_id, created_at) WHERE notification_id IS NOT NULL DO NOTHING
        """,
        (
            notification_id,
            kind,
            normalize_acct(account),
            account.get("display_name") or "",
            account.get("avatar_static") or account.get("avatar"),
            status.get("uri") or status.get("url"),
            status.get("url"),
            status.get("id"),
            content,
            content_html,
            in_reply_to_uri,
            parent_snippet,
            status.get("visibility"),
            created,
        ),
    )
    return cur.rowcount > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    if not MASTODON_TOKEN:
        log.error("Set MASTODON_TOKEN env var")
        return 1

    client = httpx.Client(
        headers={"Authorization": f"Bearer {MASTODON_TOKEN}"},
        follow_redirects=True,
    )

    log.info("Pulling notifications + DMs from %s", MASTODON_URL)
    notifs = fetch_notifications(client, args.limit)
    convos = fetch_conversations(client, 50)
    log.info("Got %d notifications, %d conversations", len(notifs), len(convos))

    init_pool()
    inserted = 0
    skipped = 0

    parent_cache: dict[str, dict | None] = {}

    with get_conn() as conn:
        for n in notifs:
            with conn.cursor() as cur:
                cur.execute("SAVEPOINT row_sp")
                try:
                    kind = n.get("type")  # mention, follow
                    status = n.get("status")
                    if kind == "mention" and status:
                        parent_id = status.get("in_reply_to_id")
                        parent = None
                        if parent_id:
                            if parent_id not in parent_cache:
                                parent_cache[parent_id] = fetch_status(client, parent_id)
                            parent = parent_cache[parent_id]
                        if insert_mention(cur, "mention", status, parent, notification_id=str(n["id"])):
                            inserted += 1
                        else:
                            skipped += 1
                    elif kind == "follow":
                        # Manufacture a synthetic "post" record with no content
                        synthetic = {
                            "uri": f"follow:{n['id']}",
                            "url": None,
                            "id": None,
                            "content": "",
                            "created_at": n["created_at"],
                            "account": n.get("account") or {},
                            "visibility": "public",
                        }
                        if insert_mention(cur, "follow", synthetic, None, notification_id=str(n["id"])):
                            inserted += 1
                        else:
                            skipped += 1
                    cur.execute("RELEASE SAVEPOINT row_sp")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT row_sp")
                    log.warning("notification %s aborted: %s", n.get("id"), e)
                    skipped += 1

        for convo in convos:
            last = convo.get("last_status")
            if not last:
                continue
            with conn.cursor() as cur:
                cur.execute("SAVEPOINT row_sp")
                try:
                    if insert_mention(cur, "dm", last, None, notification_id=f"convo:{convo['id']}"):
                        inserted += 1
                    else:
                        skipped += 1
                    cur.execute("RELEASE SAVEPOINT row_sp")
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT row_sp")
                    log.warning("convo %s aborted: %s", convo.get("id"), e)
                    skipped += 1

        conn.commit()

    log.info("DONE: %d inserted, %d skipped/duplicate", inserted, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
