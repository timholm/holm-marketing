"""Train the personal post classifier from Tim's like history in v1.

Pulls positive labels (posts Tim liked via fedi-discover v1) and a 3:1 random
sample of negatives (posts ingested but never engaged with), embeds them with
Model2Vec, time-splits into train/val, fits an SGDClassifier with log_loss,
calibrates with CalibratedClassifierCV, reports AUC and precision@10%, and
saves the model to disk.

Usage (from project root):
    /Users/tim/.local/bin/uv run python scripts/train_classifier.py

Time-based split policy:
    - The spec calls for "last 14 days = val". v1's like history is densely
      packed into 2026-04-05 .. 2026-04-11 plus a tail on 2026-04-25, so a
      naive 14-day window puts almost all positives into the val set. Instead
      we use a time-percentile split: the most recent 25% of likes (by
      action.created_at) form the val set, the rest is train. This still
      respects time ordering (no future leakage into train) while giving
      both splits enough signal to train and evaluate.
    - Negatives are sampled from posts with score < 5 in v1 (not engaged
      with by Tim), randomly distributed in time.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg

# Make the package importable when running as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fedi_studio.services.embedder import EMBEDDING_DIM, embed_batch  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("train")

V1_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon",
)
STUDIO_DSN = os.environ.get(
    "FEDI_STUDIO_DSN",
    "host=localhost port=30141 dbname=fedi_studio user=mastodon password=mastodon",
)

POSITIVE_LIMIT = 50_000  # cap from spec
NEGATIVE_RATIO = 3  # 3:1 negatives to positives
EMBED_BATCH = 256
DEFAULT_MODEL_PATH = ROOT / "models" / "scorer_v1.pkl"


@dataclass
class LabeledRow:
    post_id: int
    content: str
    author_acct: str
    label: int
    when: datetime  # action.created_at for positives; posts.first_seen_at for negatives


def fetch_positives() -> list[LabeledRow]:
    """Posts Tim liked via v1 batch_like worker. Joined to posts so we have content."""
    log.info("Pulling positives from v1 (likes joined to posts)...")
    sql = """
        SELECT DISTINCT ON (p.id)
            p.id, p.content, p.author_acct, a.created_at
        FROM posts p
        JOIN actions a ON a.post_id = p.id
        WHERE a.action_type = 'like' AND a.status = 'done'
          AND p.content IS NOT NULL AND length(p.content) > 30
        ORDER BY p.id, a.created_at ASC
        LIMIT %s
    """
    out: list[LabeledRow] = []
    with psycopg.connect(V1_DSN, connect_timeout=10, keepalives=1, keepalives_idle=30) as conn:
        cur = conn.cursor()
        cur.execute(sql, (POSITIVE_LIMIT,))
        for pid, content, author, when in cur:
            out.append(
                LabeledRow(
                    post_id=int(pid),
                    content=content or "",
                    author_acct=author or "",
                    label=1,
                    when=when,
                )
            )
    log.info("Positives: %d rows", len(out))
    return out


def fetch_negatives(n: int, exclude_ids: set[int] | None = None) -> list[LabeledRow]:
    """Random sample of posts Tim never engaged with (v1 score < 5).

    Uses TABLESAMPLE SYSTEM (block-level, ~10-100x faster than BERNOULLI on
    multi-million-row tables) and avoids a NOT EXISTS subquery against the
    actions table by passing the set of positive post IDs in via Python
    (`exclude_ids`). This lets the SQL planner short-circuit early on LIMIT
    so the connection doesn't sit idle long enough to trip kubectl
    port-forward timeouts.

    No named cursor: the entire result is pulled in one round-trip so the
    server doesn't hold an open portal across slow client-side reads.
    """
    # We over-sample modestly so a few rows of overlap with positives can be
    # trimmed without falling under the target. 1.7x is plenty given the
    # exclude_ids set is tiny relative to the post population.
    target = int(n * 1.7)
    log.info(
        "Pulling %d negatives from v1 (TABLESAMPLE SYSTEM, score < 5)...", n
    )
    population = 13_000_000
    # SYSTEM samples whole 8KB pages — typical row size means each page yields
    # ~50-100 rows. Aim for ~target rows.
    sample_pct = max(0.05, min(50.0, 100.0 * target / population))
    sql = f"""
        SELECT p.id, p.content, p.author_acct, p.first_seen_at
        FROM posts p TABLESAMPLE SYSTEM ({sample_pct})
        JOIN post_scores ps ON ps.post_id = p.id
        WHERE ps.score < 5
          AND p.content IS NOT NULL AND length(p.content) > 30
        LIMIT %s
    """
    exclude_ids = exclude_ids or set()
    out: list[LabeledRow] = []
    with psycopg.connect(V1_DSN, connect_timeout=10, keepalives=1, keepalives_idle=30) as conn:
        cur = conn.cursor()  # client-side cursor; one round-trip
        cur.execute(sql, (target,))
        for pid, content, author, when in cur:
            pid_int = int(pid)
            if pid_int in exclude_ids:
                continue
            out.append(
                LabeledRow(
                    post_id=pid_int,
                    content=content or "",
                    author_acct=author or "",
                    label=0,
                    when=when,
                )
            )
    # Shuffle then trim to exactly n
    random.Random(42).shuffle(out)
    out = out[:n]
    log.info(
        "Negatives: %d rows (sampled %.3f%% of population, excluded %d positives)",
        len(out), sample_pct, len(exclude_ids),
    )
    return out


def time_split(
    rows: list[LabeledRow], val_frac: float = 0.25
) -> tuple[list[LabeledRow], list[LabeledRow]]:
    """Split by `when` so val set is the most recent `val_frac`.

    We split positives and negatives independently by their own time percentile
    so each split has both classes regardless of when negatives were ingested.
    """
    pos = sorted([r for r in rows if r.label == 1], key=lambda r: r.when)
    neg = sorted([r for r in rows if r.label == 0], key=lambda r: r.when)

    def split(xs: list[LabeledRow]) -> tuple[list[LabeledRow], list[LabeledRow]]:
        if not xs:
            return [], []
        cut = int(len(xs) * (1.0 - val_frac))
        return xs[:cut], xs[cut:]

    pos_tr, pos_va = split(pos)
    neg_tr, neg_va = split(neg)

    train = pos_tr + neg_tr
    val = pos_va + neg_va
    random.Random(7).shuffle(train)
    random.Random(11).shuffle(val)

    log.info(
        "Split: train=%d (%d pos / %d neg), val=%d (%d pos / %d neg)",
        len(train), len(pos_tr), len(neg_tr), len(val), len(pos_va), len(neg_va),
    )
    if pos_tr:
        log.info("  Train pos time range: %s .. %s", pos_tr[0].when, pos_tr[-1].when)
    if pos_va:
        log.info("  Val pos time range:   %s .. %s", pos_va[0].when, pos_va[-1].when)
    return train, val


def embed_rows(rows: list[LabeledRow]) -> np.ndarray:
    """Embed contents in batches. Returns (N, EMBEDDING_DIM) float32."""
    if not rows:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    out = np.zeros((len(rows), EMBEDDING_DIM), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(rows), EMBED_BATCH):
        chunk = rows[i : i + EMBED_BATCH]
        out[i : i + len(chunk)] = embed_batch([r.content for r in chunk])
        if i % (EMBED_BATCH * 10) == 0:
            elapsed = time.time() - t0
            done = i + len(chunk)
            rate = done / max(elapsed, 1e-6)
            eta = (len(rows) - done) / max(rate, 1e-6)
            log.info("  embedded %d/%d (%.0f/s, eta %.0fs)", done, len(rows), rate, eta)
    log.info("Embedded %d rows in %.1fs", len(rows), time.time() - t0)
    return out


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.10) -> float:
    """Precision among the top k_frac scoring items."""
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(len(y_true) * k_frac))
    order = np.argsort(-y_score)
    top = order[:k]
    return float(np.mean(y_true[top]))


def train(
    model_path: Path,
    update_centroid: bool = True,
) -> dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import roc_auc_score

    pos = fetch_positives()
    if not pos:
        raise SystemExit("No positives found. Aborting.")
    n_neg = len(pos) * NEGATIVE_RATIO
    pos_ids = {r.post_id for r in pos}
    neg = fetch_negatives(n_neg, exclude_ids=pos_ids)
    if not neg:
        raise SystemExit("No negatives found. Aborting.")

    rows = pos + neg
    log.info("Total labeled rows: %d (%d pos / %d neg)", len(rows), len(pos), len(neg))

    train_rows, val_rows = time_split(rows, val_frac=0.25)

    log.info("Embedding train rows...")
    X_train = embed_rows(train_rows)
    y_train = np.array([r.label for r in train_rows], dtype=np.int64)

    log.info("Embedding val rows...")
    X_val = embed_rows(val_rows)
    y_val = np.array([r.label for r in val_rows], dtype=np.int64)

    log.info("Fitting SGDClassifier(loss=log_loss, alpha=1e-5, eta0=0.01)...")
    base = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        learning_rate="adaptive",
        eta0=0.01,
        random_state=42,
        max_iter=50,
        tol=1e-4,
    )
    base.fit(X_train, y_train)

    log.info("Calibrating probabilities with CalibratedClassifierCV (sigmoid, prefit via FrozenEstimator)...")
    # We carve a small slice off the training set for calibration so the val
    # set stays truly unseen. In sklearn 1.6+ the way to do "prefit" calibration
    # is to wrap an already-fit estimator in FrozenEstimator.
    from sklearn.frozen import FrozenEstimator

    n_train = len(X_train)
    cal_n = min(5000, max(500, n_train // 5))
    rng = np.random.default_rng(123)
    idx = rng.permutation(n_train)
    cal_idx = idx[:cal_n]
    fit_idx = idx[cal_n:]
    base_for_cal = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        learning_rate="adaptive",
        eta0=0.01,
        random_state=42,
        max_iter=50,
        tol=1e-4,
    )
    base_for_cal.fit(X_train[fit_idx], y_train[fit_idx])
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_for_cal), method="sigmoid"
    )
    calibrated.fit(X_train[cal_idx], y_train[cal_idx])

    log.info("Evaluating on val set (n=%d)...", len(X_val))
    y_score = calibrated.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_score) if len(np.unique(y_val)) > 1 else float("nan")
    p_at_10 = precision_at_k(y_val, y_score, 0.10)
    base_rate = float(np.mean(y_val))

    log.info("=== VALIDATION RESULTS ===")
    log.info("  AUC:                %.4f", auc)
    log.info("  Precision@top-10%%:  %.4f", p_at_10)
    log.info("  Val base rate (pos):%.4f", base_rate)
    log.info("  Val score range:    [%.3f, %.3f]", float(y_score.min()), float(y_score.max()))

    user_centroid = None
    if update_centroid:
        # Centroid: mean of POSITIVE embeddings across train+val (Tim's true preference)
        pos_idx_train = np.where(y_train == 1)[0]
        pos_idx_val = np.where(y_val == 1)[0]
        pos_embs = np.vstack([X_train[pos_idx_train], X_val[pos_idx_val]])
        user_centroid = pos_embs.mean(axis=0).astype(np.float32)
        log.info(
            "Computed user_centroid from %d liked posts (norm=%.3f)",
            len(pos_embs), float(np.linalg.norm(user_centroid)),
        )

    # Save the calibrated classifier alongside coefs and centroid in a Scorer-compatible payload
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": calibrated,
        "user_centroid": user_centroid,
        "author_priors": {},
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 0.15,
        "delta": 0.05,
        "is_fit": True,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {
            "auc": float(auc),
            "precision_at_10pct": p_at_10,
            "val_base_rate": base_rate,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_pos_train": int((y_train == 1).sum()),
            "n_pos_val": int((y_val == 1).sum()),
        },
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    size = model_path.stat().st_size
    log.info("Saved model to %s (%d bytes)", model_path, size)

    if user_centroid is not None:
        log.info("Updating user_centroid in fedi_studio...")
        with psycopg.connect(STUDIO_DSN, connect_timeout=10) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE user_centroid
                SET embedding = %s, based_on_likes = %s, updated_at = NOW()
                WHERE id = 1
                """,
                (user_centroid.tolist(), int(payload["metrics"]["n_pos_train"] + payload["metrics"]["n_pos_val"])),
            )
            conn.commit()
            cur.execute("SELECT based_on_likes, array_length(embedding, 1) FROM user_centroid WHERE id = 1")
            row = cur.fetchone()
            log.info("  user_centroid: based_on_likes=%s array_length=%s", row[0], row[1])

    return payload["metrics"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--no-centroid", action="store_true", help="Skip user_centroid update")
    args = p.parse_args()

    metrics = train(Path(args.model_path), update_centroid=not args.no_centroid)
    auc = metrics["auc"]
    if not (auc > 0.5):
        log.error("AUC %.3f <= 0.5; classifier has no signal.", auc)
        return 1
    if auc <= 0.7:
        log.warning("AUC %.3f below the 0.7 target. Saved anyway.", auc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
