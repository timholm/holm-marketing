"""Smoke test: embed and score real posts from v1 fedi_discover DB.

Usage (from project root, after installing deps):
    python scripts/smoke_test_scorer.py

What it does:
1. Loads Model2Vec potion-base-32M (downloads ~30MB on first run).
2. Pulls 20 random posts from v1's posts table.
3. Embeds them all in a single batch.
4. Scores each post with a cold-start (untrained) Scorer.
5. Prints score, components, and post snippet so we can sanity-check.

Verification:
- Embeddings are non-zero 256-dim vectors.
- Untrained scorer returns ~0.5 probability for everything (correct cold-start).
- After partial_fit on a few examples, scores diverge from 0.5.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import psycopg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedi_studio.services.embedder import EMBEDDING_DIM, embed, embed_batch
from fedi_studio.services.scorer import ScoreInput, Scorer

DB_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon",
)


def main() -> int:
    print("=== Smoke test: embedder ===")
    v = embed("hello world")
    assert v.shape == (EMBEDDING_DIM,), f"expected ({EMBEDDING_DIM},), got {v.shape}"
    assert v.dtype == np.float32
    assert not np.allclose(v, 0)
    print(f"  embed('hello world'): shape={v.shape} norm={np.linalg.norm(v):.3f} OK")

    print("\n=== Pulling 20 posts from v1 ===")
    try:
        with psycopg.connect(DB_DSN, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content, author_acct, posted_at FROM posts "
                    "WHERE content IS NOT NULL AND length(content) > 50 "
                    "ORDER BY random() LIMIT 20"
                )
                posts = cur.fetchall()
    except Exception as e:
        print(f"  Could not reach v1 PG: {e}")
        print("  Falling back to synthetic test posts.")
        posts = [
            (
                "Just finished pouring the slab for my off-grid cabin in Arizona. Solar panels arriving next week!",
                "homesteader@example.com",
                "2026-04-25 12:00:00+00",
            ),
            (
                "Watching the Lakers game tonight, what a comeback!",
                "sportsfan@example.com",
                "2026-04-25 11:00:00+00",
            ),
            (
                "Set up my Kubernetes cluster on Raspberry Pis this weekend. K3s + Longhorn working great.",
                "homelab@example.com",
                "2026-04-25 10:00:00+00",
            ),
        ]

    print(f"  Got {len(posts)} posts")
    contents = [p[0] for p in posts]

    print("\n=== Batch embedding ===")
    embeddings = embed_batch(contents)
    print(f"  Shape: {embeddings.shape} (expected ({len(posts)}, {EMBEDDING_DIM}))")
    assert embeddings.shape == (len(posts), EMBEDDING_DIM)
    assert embeddings.dtype == np.float32

    print("\n=== Scoring (cold start, untrained) ===")
    scorer = Scorer()
    for i, (content, author, posted_at) in enumerate(posts[:5]):
        if isinstance(posted_at, str):
            posted_at = datetime.fromisoformat(posted_at).replace(tzinfo=timezone.utc)
        elif posted_at and posted_at.tzinfo is None:
            posted_at = posted_at.replace(tzinfo=timezone.utc)
        elif posted_at is None:
            posted_at = datetime.now(timezone.utc)

        result = scorer.score(
            ScoreInput(
                content=content,
                author_acct=author or "unknown",
                posted_at=posted_at,
                embedding=embeddings[i],
            )
        )
        snippet = content.replace("\n", " ")[:80]
        print(f"  [{result.probability:.3f}] {author}: {snippet}")

    print("\n=== Online learning test ===")
    print("  Teaching scorer that posts 0 and 2 are positive, 1 is negative...")
    scorer.partial_fit(embeddings[0], 1, author_acct=posts[0][1] or "")
    scorer.partial_fit(embeddings[1], 0, author_acct=posts[1][1] or "")
    scorer.partial_fit(embeddings[2], 1, author_acct=posts[2][1] or "")

    print("\n=== Scoring after training (3 examples) ===")
    for i, (content, author, posted_at) in enumerate(posts[:5]):
        if isinstance(posted_at, str):
            posted_at = datetime.fromisoformat(posted_at).replace(tzinfo=timezone.utc)
        elif posted_at and posted_at.tzinfo is None:
            posted_at = posted_at.replace(tzinfo=timezone.utc)
        elif posted_at is None:
            posted_at = datetime.now(timezone.utc)

        result = scorer.score(
            ScoreInput(
                content=content,
                author_acct=author or "unknown",
                posted_at=posted_at,
                embedding=embeddings[i],
            )
        )
        snippet = content.replace("\n", " ")[:80]
        print(f"  [{result.probability:.3f}] {author}: {snippet}")

    print("\n=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
