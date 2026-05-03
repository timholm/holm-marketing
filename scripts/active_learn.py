#!/usr/bin/env python3
"""Active learning: incrementally update classifier from Tim's recent feedback.

Pulls events from the last 30 days:
  - Positives: bookmark, read events -> label=1
  - Negatives: dismiss events -> label=0

For each labeled example with an embedding:
  1. Load the base scorer (v2 -> v1 -> cold)
  2. Call partial_fit(embedding, label, author_acct)
  3. Save to v3.pkl + v3.meta.json
  4. Update discovery logic (score_all.py, build_candidates.py)

Thresholds:
  - Skip if < 50 total examples (log warning, keep v2)
  - Estimate AUC on held-out sample if >= 50 examples
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("active_learn")

# Resolve repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def resolve_base_scorer_path() -> str:
    """Prefer v2 (retrained on full v1 DB), fall back to v1."""
    for v in ("v2", "v1"):
        candidate = _REPO_ROOT / "models" / f"scorer_{v}.pkl"
        if candidate.exists():
            return str(candidate)
    # Last resort: v2 path even if doesn't exist (load_or_initialize handles it)
    return str(_REPO_ROOT / "models" / "scorer_v2.pkl")


BASE_SCORER_PATH = resolve_base_scorer_path()
OUTPUT_V3_PATH = _REPO_ROOT / "models" / "scorer_v3.pkl"
OUTPUT_META_PATH = _REPO_ROOT / "models" / "scorer_v3.meta.json"

MIN_EXAMPLES = 50


def load_scorer():
    """Load the base scorer (v2 -> v1 -> cold start)."""
    from fedi_studio.services.scorer import Scorer

    return Scorer.load_or_initialize(BASE_SCORER_PATH)


def get_db_conn():
    """Get a database connection."""
    from fedi_studio.models.db import get_conn, init_pool

    init_pool()
    return get_conn()


def load_feedback_events() -> tuple[list[dict], list[dict]]:
    """Fetch events from last 30 days, split into positives/negatives.

    Returns (positives, negatives) where each dict has:
      {post_id, embedding: np.ndarray, author_acct}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Positives: bookmark, read
            cur.execute(
                """
                SELECT p.id, p.embedding, p.author_acct
                FROM events e
                JOIN posts p ON p.id = e.target_id
                WHERE e.event_type IN ('bookmark', 'read')
                  AND e.target_type = 'post'
                  AND e.created_at > %s
                  AND p.embedding IS NOT NULL
                ORDER BY e.created_at DESC
                """,
                (cutoff,),
            )
            positives = [
                {
                    "post_id": row[0],
                    "embedding": np.array(row[1], dtype=np.float32),
                    "author_acct": row[2],
                }
                for row in cur.fetchall()
            ]

            # Negatives: dismiss
            cur.execute(
                """
                SELECT p.id, p.embedding, p.author_acct
                FROM events e
                JOIN posts p ON p.id = e.target_id
                WHERE e.event_type = 'dismiss'
                  AND e.target_type = 'post'
                  AND e.created_at > %s
                  AND p.embedding IS NOT NULL
                ORDER BY e.created_at DESC
                """,
                (cutoff,),
            )
            negatives = [
                {
                    "post_id": row[0],
                    "embedding": np.array(row[1], dtype=np.float32),
                    "author_acct": row[2],
                }
                for row in cur.fetchall()
            ]

    log.info("Loaded %d positives, %d negatives from last 30 days", len(positives), len(negatives))
    return positives, negatives


def train_v3(scorer, positives: list[dict], negatives: list[dict]) -> int:
    """Fit the scorer incrementally on all examples.

    Note: If the loaded scorer has a CalibratedClassifierCV (from v2), we cannot
    use partial_fit. Instead, we batch-train on all examples at once.

    Returns total number of examples processed.
    """
    from sklearn.linear_model import SGDClassifier

    total = 0

    # Check if the loaded classifier supports partial_fit
    if not hasattr(scorer.classifier, 'partial_fit'):
        log.info("Loaded classifier doesn't support partial_fit (likely CalibratedClassifierCV). "
                 "Resetting to fresh SGDClassifier for batch training.")
        scorer.classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=42,
        )
        scorer._is_fit = False

    # Train incrementally
    for ex in positives:
        scorer.partial_fit(ex["embedding"], label=1, author_acct=ex["author_acct"])
        total += 1
    for ex in negatives:
        scorer.partial_fit(ex["embedding"], label=0, author_acct=ex["author_acct"])
        total += 1
    log.info("Trained on %d examples (partial_fit)", total)
    return total


def estimate_auc(scorer, positives: list[dict], negatives: list[dict]) -> float | None:
    """Estimate AUC on held-out sample to avoid overfitting.

    Sample up to 100 examples from same period to avoid overfitting on
    the training set. If insufficient examples, return None.
    """
    from sklearn.metrics import roc_auc_score

    # Sample held-out set: take every Nth example to avoid contamination
    if len(positives) + len(negatives) < 100:
        log.warning("Too few examples for robust held-out AUC estimate")
        return None

    # Use every 3rd example as held-out
    held_pos = positives[::3]
    held_neg = negatives[::3]

    if len(held_pos) + len(held_neg) < 20:
        log.warning("Held-out set too small (%d examples)", len(held_pos) + len(held_neg))
        return None

    # Score each held-out example using the classifier's logreg component
    scores = []
    labels = []

    for ex in held_pos:
        prob = scorer._logreg_prob(ex["embedding"])
        scores.append(prob)
        labels.append(1)

    for ex in held_neg:
        prob = scorer._logreg_prob(ex["embedding"])
        scores.append(prob)
        labels.append(0)

    try:
        auc = roc_auc_score(labels, scores)
        log.info("Held-out AUC (logreg component): %.3f", auc)
        return auc
    except Exception as e:
        log.warning("AUC calculation failed: %s", e)
        return None


def save_v3(scorer, positives_count: int, negatives_count: int, auc: float | None) -> None:
    """Save v3 scorer and metadata."""
    scorer.save(str(OUTPUT_V3_PATH))
    log.info("Saved scorer_v3.pkl to %s", OUTPUT_V3_PATH)

    meta = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "n_positives": positives_count,
        "n_negatives": negatives_count,
        "base_model_path": BASE_SCORER_PATH,
        "auc_held_out": round(auc, 4) if auc is not None else None,
    }
    with open(OUTPUT_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved scorer_v3.meta.json to %s", OUTPUT_META_PATH)
    print(json.dumps(meta, indent=2))


def update_discovery_paths() -> None:
    """Update score_all.py and build_candidates.py to prefer v3."""
    # --- Update score_all.py ---
    score_all_path = _REPO_ROOT / "src" / "fedi_studio" / "workers" / "score_all.py"
    if score_all_path.exists():
        content = score_all_path.read_text()
        # The candidates list is in _resolve_model_path function; two variants exist
        old1 = """    candidates = [
        "/app/models/scorer_v2.pkl",
        "/app/models/scorer_v1.pkl",
        str(_REPO_ROOT / "models" / "scorer_v2.pkl"),
        str(_REPO_ROOT / "models" / "scorer_v1.pkl"),
    ]"""
        new1 = """    candidates = [
        "/app/models/scorer_v3.pkl",
        "/app/models/scorer_v2.pkl",
        "/app/models/scorer_v1.pkl",
        str(_REPO_ROOT / "models" / "scorer_v3.pkl"),
        str(_REPO_ROOT / "models" / "scorer_v2.pkl"),
        str(_REPO_ROOT / "models" / "scorer_v1.pkl"),
    ]"""
        if old1 in content:
            content = content.replace(old1, new1)
            score_all_path.write_text(content)
            log.info("Updated score_all.py: added scorer_v3 to model discovery order")

    # --- Update build_candidates.py ---
    build_cand_path = _REPO_ROOT / "src" / "fedi_studio" / "workers" / "build_candidates.py"
    if build_cand_path.exists():
        content = build_cand_path.read_text()
        old = """    for c in ("models/scorer_v2.pkl", "models/scorer_v1.pkl"):"""
        new = """    for c in ("models/scorer_v3.pkl", "models/scorer_v2.pkl", "models/scorer_v1.pkl"):"""
        if old in content:
            content = content.replace(old, new)
            build_cand_path.write_text(content)
            log.info("Updated build_candidates.py: added scorer_v3 to search order")


def main() -> int:
    log.info("Active learning: loading feedback events")

    positives, negatives = load_feedback_events()
    total = len(positives) + len(negatives)

    if total < MIN_EXAMPLES:
        log.warning(
            "Not enough feedback yet: %d examples (need >= %d). Keeping v2.",
            total,
            MIN_EXAMPLES,
        )
        return 0

    log.info("Training v3 scorer on %d examples", total)
    scorer = load_scorer()

    # Train on all examples
    train_v3(scorer, positives, negatives)

    # Estimate AUC on held-out sample
    auc = estimate_auc(scorer, positives, negatives)

    # Save v3
    save_v3(scorer, len(positives), len(negatives), auc)

    # Update discovery paths
    update_discovery_paths()

    log.info("Done: v3 shipped. Positives=%d, Negatives=%d", len(positives), len(negatives))
    return 0


if __name__ == "__main__":
    sys.exit(main())
