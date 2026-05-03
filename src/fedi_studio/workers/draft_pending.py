"""For every mention without a draft yet, generate one (or record refusal).

Idempotent: skips mentions that already have a mention_drafts row.
"""

from __future__ import annotations

import logging
import sys

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.drafter import draft_reply

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("draft_pending")


def main() -> int:
    init_pool()
    drafted = 0
    refused = 0
    with get_conn() as conn:
        cur = conn.cursor()
        # Tim's own acct gets skipped — DMs we sent show up as Tim-as-author.
        cur.execute(
            """
            SELECT m.id, m.created_at, m.kind, m.author_acct, m.post_content, m.parent_content
            FROM mentions m
            LEFT JOIN mention_drafts d
                ON d.mention_id = m.id AND d.mention_created_at = m.created_at
            WHERE m.kind IN ('mention', 'reply', 'dm')
              AND d.id IS NULL
              AND m.author_acct != 'tim@holm.community'
              AND length(coalesce(m.post_content, '')) > 20
            ORDER BY m.created_at DESC
            LIMIT 200
            """
        )
        rows = cur.fetchall()

    log.info("Drafting for %d mentions", len(rows))
    for mention_id, created_at, kind, author, content, parent in rows:
        d = draft_reply(content or "", parent or "", author or "")
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO mention_drafts (
                    mention_id, mention_created_at, draft_text, rationale,
                    on_topic, topic_match
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (mention_id, created_at, d.text, (d.rationale or d.refused_reason or ""), d.on_topic, d.topic_match),
            )
            conn.commit()
        if d.text:
            drafted += 1
            log.info("DRAFTED for %s: %s", author, d.text[:80])
        else:
            refused += 1
            log.info("REFUSED %s: %s", author, d.refused_reason)

    log.info("DONE: drafted=%d refused=%d", drafted, refused)
    return 0


if __name__ == "__main__":
    sys.exit(main())
