"""Rescore existing candidates using the new weights validated 2026-04-25.

Reads `candidates.reasoning` JSON (which already contains mean_prob,
centroid_sim, topic_bonus from the original scoring run), recomputes the
score with the new weights, and updates `candidates.score` and
`candidates.reasoning`. Idempotent — safe to re-run.

This avoids re-querying Mastodon API or running embeddings.
"""

from __future__ import annotations

import json
import logging
import os

import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rescore_candidates")

V2_DSN = os.environ.get(
    "V2_DSN",
    "host=localhost port=31141 dbname=fedi_studio user=mastodon password=mastodon",
)

WEIGHT_MEAN_PROB = 0.30
WEIGHT_CENTROID_SIM = 0.00
WEIGHT_TOPIC_BONUS = 0.70
WEIGHTS_VERSION = "2026-04-25"


def main() -> None:
    log.info("connecting to %s", V2_DSN.split("dbname=")[1].split()[0])
    conn = psycopg.connect(V2_DSN)
    n_rescored = 0
    n_skipped = 0
    n_already_v2 = 0

    with conn.cursor() as cur:
        cur.execute("SELECT id, score, reasoning FROM candidates")
        rows = cur.fetchall()
    log.info("loaded %d candidates", len(rows))

    BATCH = 500
    updates: list[tuple] = []
    for cand_id, old_score, reasoning in rows:
        if not reasoning:
            n_skipped += 1
            continue
        # Skip if already at the new weights version
        if reasoning.get("weights_version") == WEIGHTS_VERSION:
            n_already_v2 += 1
            continue

        mean_prob = reasoning.get("mean_prob")
        centroid_sim = reasoning.get("centroid_sim")
        topic_bonus = reasoning.get("topic_bonus")
        if mean_prob is None or centroid_sim is None or topic_bonus is None:
            n_skipped += 1
            continue

        new_score = (
            WEIGHT_MEAN_PROB * float(mean_prob)
            + WEIGHT_CENTROID_SIM * float(centroid_sim)
            + WEIGHT_TOPIC_BONUS * float(topic_bonus)
        )
        new_score = max(0.0, min(1.0, new_score))

        new_reasoning = dict(reasoning)
        new_reasoning["weights"] = {
            "mean_prob": WEIGHT_MEAN_PROB,
            "centroid": WEIGHT_CENTROID_SIM,
            "topic_bonus": WEIGHT_TOPIC_BONUS,
        }
        new_reasoning["weights_version"] = WEIGHTS_VERSION
        new_reasoning["previous_score"] = float(old_score) if old_score is not None else None

        updates.append((new_score, json.dumps(new_reasoning), cand_id))
        n_rescored += 1

    log.info("computed updates: %d (skipped=%d, already_v2=%d)", n_rescored, n_skipped, n_already_v2)
    if not updates:
        log.info("nothing to update")
        conn.close()
        return

    with conn.cursor() as cur:
        for i in range(0, len(updates), BATCH):
            batch = updates[i : i + BATCH]
            cur.executemany(
                "UPDATE candidates SET score=%s, reasoning=%s::jsonb WHERE id=%s",
                batch,
            )
            conn.commit()
            log.info("committed batch %d/%d (%d rows)",
                     i // BATCH + 1, (len(updates) + BATCH - 1) // BATCH, len(batch))

    log.info("rescored %d candidates", n_rescored)

    # Print before/after stats
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*), MAX(score), MIN(score), AVG(score),
                   COUNT(*) FILTER (WHERE score >= 0.5),
                   COUNT(*) FILTER (WHERE score >= 0.3),
                   COUNT(*) FILTER (WHERE score >= 0.2),
                   COUNT(*) FILTER (WHERE score >= 0.1)
            FROM candidates
            """
        )
        row = cur.fetchone()
        log.info(
            "after rescore: total=%d max=%.3f min=%.3f avg=%.3f >=0.5:%d >=0.3:%d >=0.2:%d >=0.1:%d",
            *row,
        )
    conn.close()


if __name__ == "__main__":
    main()
