"""Validate the candidate score weights using Tim's existing follow set as ground truth.

Pipeline:
1. POSITIVES: pull Tim's followings from Mastodon API (paginated). For each:
   - profile (bio) from API
   - last 10 posts (from v2 posts with probabilities, fall back to v1 posts)
2. NEGATIVES: random 3000 sample from v1.follow_tracking where followed_back=0,
   joined with v1.profiles for bio and v1.posts for last 10 posts.
3. For each labeled account, compute the 3 raw components used in compute_score:
   mean_prob, centroid_sim, topic_bonus.
4. Evaluate current weights (0.55, 0.30, 0.15) — AUC, precision/recall at 0.5.
5. Fit logistic regression to find optimal weights, report new AUC.
6. Report.

READ-ONLY against Mastodon API. Only GET endpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any  # noqa: F401

import httpx
import numpy as np
import psycopg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add repo to path so we can import fedi_studio modules
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fedi_studio.services.embedder import cosine_similarity, embed  # noqa: E402
from fedi_studio.workers.build_candidates import (  # noqa: E402
    _normalize,
    _topic_bonus,
    load_user_centroid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validate_weights")

V1_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=31141 dbname=fedi_discover_full user=mastodon password=mastodon",
)
V2_DSN = os.environ.get(
    "V2_DSN",
    "host=localhost port=31141 dbname=fedi_studio user=mastodon password=mastodon",
)
MASTODON_URL = "https://holm.community"
MASTODON_TOKEN = os.environ["MASTODON_TOKEN"]
HTTP_TIMEOUT = 12.0
USER_AGENT = "fedi-studio-validate-weights/1.0 (read-only)"


# --- Mastodon API helpers ---------------------------------------------------


async def _get_json(client: httpx.AsyncClient, url: str, params=None) -> tuple[Any, dict]:
    backoff = 1.0
    for _ in range(5):
        try:
            r = await client.get(url, params=params)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", backoff))
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, 30.0)
                continue
            if r.status_code >= 500:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            if r.status_code in (401, 403, 404, 410, 422):
                return None, dict(r.headers)
            r.raise_for_status()
            return r.json(), dict(r.headers)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError):
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    return None, {}


async def fetch_followings_with_profiles(
    client: httpx.AsyncClient, tim_id: str
) -> list[dict]:
    """Walk Mastodon Link-header pagination of accounts/{id}/following.

    Returns a list of full profile dicts. Each /following response actually
    returns full account objects already, so we don't need a second lookup.
    """
    out: list[dict] = []
    url = f"{MASTODON_URL}/api/v1/accounts/{tim_id}/following"
    params = {"limit": 80}
    page = 0
    while url:
        data, headers = await _get_json(client, url, params=params)
        params = None
        if not isinstance(data, list):
            break
        out.extend(data)
        page += 1
        if page % 5 == 0:
            log.info("[following] page %d, total=%d", page, len(out))
        link = headers.get("link") or headers.get("Link") or ""
        nxt = None
        for part in link.split(","):
            part = part.strip()
            if 'rel="next"' in part and "<" in part and ">" in part:
                nxt = part[part.index("<") + 1 : part.index(">")]
                break
        url = nxt
    log.info("[following] done, total=%d", len(out))
    return out


# --- Local DB ---------------------------------------------------------------


def _reconnect_on_failure(get_dsn: str, label: str):
    """Decorator-style helper: returns a (cur_factory, conn_holder) such that
    on OperationalError we open a fresh connection. Used in the chunked loops
    below to survive port-forward flaps without losing collected results.
    """
    state = {"conn": None}

    def get_conn():
        if state["conn"] is None or state["conn"].closed:
            for attempt in range(8):
                try:
                    state["conn"] = psycopg.connect(get_dsn, connect_timeout=15, autocommit=True)
                    return state["conn"]
                except psycopg.OperationalError as e:
                    log.warning("  reconnect %s attempt %d failed: %s", label, attempt + 1, e)
                    time.sleep(min(2 ** attempt, 20))
            raise RuntimeError(f"reconnect to {label} exhausted")
        return state["conn"]

    return get_conn


def load_recent_post_probs_v2(
    v2: psycopg.Connection, accts: list[str]
) -> dict[str, list[float]]:
    """Use functional index on lower(author_acct). Resilient against PG
    port-forward flaps via per-chunk reconnect."""
    if not accts:
        return {}
    out: dict[str, list[float]] = {a: [] for a in accts}
    CHUNK = 50
    log.info("  v2: lookup against lower(author_acct) for %d accts (functional idx)", len(accts))
    get_v2 = _reconnect_on_failure(V2_DSN, "v2")
    i = 0
    n_chunks = (len(accts) + CHUNK - 1) // CHUNK
    while i < len(accts):
        chunk = accts[i : i + CHUNK]
        try:
            conn = get_v2()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH targets(acct) AS (SELECT unnest(%s::text[]))
                    SELECT t.acct, ps.probability
                    FROM targets t
                    JOIN LATERAL (
                        SELECT p.id, p.posted_at
                        FROM posts p
                        WHERE lower(p.author_acct) = t.acct
                        ORDER BY p.posted_at DESC
                        LIMIT 10
                    ) p ON true
                    JOIN post_scores ps ON ps.post_id = p.id AND ps.posted_at = p.posted_at
                    """,
                    (chunk,),
                )
                for acct, prob in cur:
                    if len(out[acct]) < 10:
                        out[acct].append(float(prob))
            if (i // CHUNK) % 5 == 0:
                log.info("    v2 chunk %d/%d, hits=%d",
                         i // CHUNK + 1, n_chunks,
                         sum(1 for v in out.values() if v))
            i += CHUNK
        except (psycopg.OperationalError, psycopg.InterfaceError) as e:
            log.warning("  v2 chunk %d/%d connection error: %s — reconnecting",
                        i // CHUNK + 1, n_chunks, e)
            try:
                if not (out and out.values()):
                    pass
            except Exception:
                pass
            # Force reconnect on next iteration
            try:
                if "conn" in dir():
                    conn.close()
            except Exception:
                pass
            # Reset state in get_v2 closure
            get_v2 = _reconnect_on_failure(V2_DSN, "v2")
            time.sleep(2)
            # Don't increment i — retry the chunk
    log.info("  v2 total hits: %d/%d", sum(1 for v in out.values() if v), len(accts))
    return {a: ps for a, ps in out.items() if ps}


def load_recent_v1_posts_content(
    v1: psycopg.Connection, accts: list[str]
) -> dict[str, list[str]]:
    """v1 posts is 14M rows w/ no functional index on lower(). For each acct,
    we'd hit a seq scan. This function is now a no-op stub: we don't materialize
    classifier probabilities for v1-only accts in this validation. The original
    compute_score also falls back to 0.5 when there are no v2 probs, so this
    matches production behavior.
    """
    return {}


def load_recent_v1_post_scores_canonical(
    v1: psycopg.Connection, original_accts: list[str]
) -> dict[str, list[float]]:
    """Last 10 v1.post_scores.score per acct, normalized to [0, 1]. Resilient."""
    if not original_accts:
        return {}
    out: dict[str, list[float]] = {}
    CHUNK = 50
    log.info("  v1: indexed lookup against author_acct (original case) for %d accts", len(original_accts))
    get_v1 = _reconnect_on_failure(V1_DSN, "v1")
    i = 0
    n_chunks = (len(original_accts) + CHUNK - 1) // CHUNK
    while i < len(original_accts):
        chunk = original_accts[i : i + CHUNK]
        try:
            conn = get_v1()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH targets(acct) AS (SELECT unnest(%s::text[]))
                    SELECT t.acct, ps.score
                    FROM targets t
                    JOIN LATERAL (
                        SELECT id, posted_at FROM posts
                        WHERE author_acct = t.acct
                        ORDER BY posted_at DESC LIMIT 10
                    ) p ON true
                    JOIN post_scores ps ON ps.post_id = p.id
                    """,
                    (chunk,),
                )
                for original_acct, score in cur:
                    key = original_acct.lower()
                    out.setdefault(key, [])
                    if len(out[key]) < 10:
                        out[key].append(min(float(score or 0) / 100.0, 1.0))
            if (i // CHUNK) % 5 == 0:
                log.info("    v1 chunk %d/%d, hits=%d",
                         i // CHUNK + 1, n_chunks,
                         sum(1 for v in out.values() if v))
            i += CHUNK
        except (psycopg.OperationalError, psycopg.InterfaceError) as e:
            log.warning("  v1 chunk %d/%d conn error: %s — reconnect", i // CHUNK + 1, n_chunks, e)
            get_v1 = _reconnect_on_failure(V1_DSN, "v1")
            time.sleep(2)
    log.info("  v1 hits: %d/%d", sum(1 for v in out.values() if v), len(original_accts))
    return out


def strip_html(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"<[^>]+>", " ", s).strip()


def load_v1_profiles(
    v1: psycopg.Connection, accts: list[str]
) -> dict[str, dict]:
    if not accts:
        return {}
    out: dict[str, dict] = {}
    with v1.cursor() as cur:
        cur.execute(
            """
            SELECT lower(acct), bio, post_count, last_post_at
            FROM profiles
            WHERE lower(acct) = ANY(%s)
            """,
            (accts,),
        )
        for a, bio, pc, lpa in cur:
            out[a] = {"bio": bio or "", "post_count": pc or 0, "last_post_at": lpa}
    return out


# --- Component computation --------------------------------------------------


def predict_post_prob_from_classifier(scorer, embedding: np.ndarray) -> float:
    """Use the trained classifier to score a single post embedding."""
    if not scorer._is_fit:
        return 0.5
    try:
        X = embedding.reshape(1, -1)
        return float(scorer.classifier.predict_proba(X)[0, 1])
    except Exception:
        return 0.5


def compute_components(
    bio_text: str,
    recent_post_probs: list[float],
    user_centroid: np.ndarray | None,
) -> tuple[float, float, float]:
    """Return (mean_prob, centroid_sim, topic_bonus)."""
    mean_prob = float(np.mean(recent_post_probs)) if recent_post_probs else 0.5

    centroid_sim = 0.5
    if user_centroid is not None and bio_text:
        try:
            bio_emb = embed(bio_text)
            cs = cosine_similarity(bio_emb, user_centroid)
            centroid_sim = (cs + 1.0) / 2.0
        except Exception:
            pass

    topic_bonus, _matched = _topic_bonus(bio_text)
    return mean_prob, centroid_sim, topic_bonus


# --- Main -------------------------------------------------------------------


async def gather_positives(
    client: httpx.AsyncClient, tim_id: str, v1: psycopg.Connection, v2: psycopg.Connection
) -> list[dict]:
    """Returns list of {acct, bio, recent_post_probs} for positives."""
    profiles = await fetch_followings_with_profiles(client, tim_id)
    log.info("got %d following profiles", len(profiles))

    accts: list[str] = []  # lowered
    accts_original: list[str] = []  # canonical case for indexed lookup
    bios: dict[str, str] = {}
    for p in profiles:
        api_acct = (p.get("acct") or "")
        if not api_acct:
            continue
        # Local accts come without instance suffix
        if "@" not in api_acct:
            api_acct_full = f"{api_acct}@holm.community"
        else:
            api_acct_full = api_acct
        lowered = _normalize(api_acct_full)
        accts.append(lowered)
        accts_original.append(api_acct_full)
        bios[lowered] = strip_html(p.get("note") or "")

    # v2 probs via functional lower() index (covers v2-known accts)
    log.info("loading recent post probs from v2 for %d positive accts", len(accts))
    v2_probs = load_recent_post_probs_v2(v2, accts)
    log.info("v2 had probs for %d/%d positive accts", len(v2_probs), len(accts))

    # v1 score backfill via indexed original-case lookup, only for accts not in v2
    missing_original = [
        accts_original[i] for i, a in enumerate(accts) if a not in v2_probs
    ]
    log.info("loading v1 scores for %d missing positives", len(missing_original))
    v1_scores = load_recent_v1_post_scores_canonical(v1, missing_original)
    log.info("v1 had scores for %d/%d missing", len(v1_scores), len(missing_original))

    rows: list[dict] = []
    for acct in accts:
        recent = v2_probs.get(acct) or v1_scores.get(acct) or []
        rows.append({
            "acct": acct,
            "bio": bios.get(acct, ""),
            "recent_probs": recent[:10],
            "v1_post_content": [],
        })
    return rows


def gather_negatives(
    v1: psycopg.Connection, v2: psycopg.Connection, n_target: int, exclude: set[str]
) -> list[dict]:
    """Random sample of accts from follow_tracking where followed_back=0.

    Pulls (lowered_acct, original_acct, bio) so we can use the original
    casing for indexed v1.posts lookups. Excludes any acct in `exclude`.
    """
    rng = random.Random(42)
    log.info("sampling negatives from follow_tracking where followed_back=0")

    # Resilient single-shot query
    get_v1 = _reconnect_on_failure(V1_DSN, "v1")
    all_rows = None
    for attempt in range(8):
        try:
            conn = get_v1()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (lower(ft.acct))
                        lower(ft.acct) AS lkey,
                        ft.acct        AS original_acct,
                        pr.bio         AS bio
                    FROM follow_tracking ft
                    JOIN profiles pr ON lower(pr.acct) = lower(ft.acct)
                    WHERE ft.followed_back = 0
                      AND pr.bio IS NOT NULL AND pr.bio != ''
                    """
                )
                all_rows = cur.fetchall()
            break
        except (psycopg.OperationalError, psycopg.InterfaceError) as e:
            log.warning("  follow_tracking query attempt %d failed: %s — reconnect", attempt + 1, e)
            get_v1 = _reconnect_on_failure(V1_DSN, "v1")
            time.sleep(2)
    if all_rows is None:
        raise RuntimeError("could not pull follow_tracking after retries")
    # Filter out positives
    filtered = [(lk, oa, b) for (lk, oa, b) in all_rows if lk not in exclude]
    log.info("eligible negative pool: %d", len(filtered))
    rng.shuffle(filtered)
    sampled = filtered[:n_target]
    log.info("sampled %d negatives", len(sampled))

    sampled_lowered = [r[0] for r in sampled]
    sampled_original = [r[1] for r in sampled]
    bios = {r[0]: r[2] for r in sampled}

    # v2 probs (lowered) — uses functional index we created
    v2_probs = load_recent_post_probs_v2(v2, sampled_lowered)
    # v1 score backfill (original cased — uses existing index)
    v1_scores = load_recent_v1_post_scores_canonical(v1, sampled_original)

    rows: list[dict] = []
    for acct_lower in sampled_lowered:
        # Prefer v2 calibrated, fall back to v1 score
        recent = v2_probs.get(acct_lower) or v1_scores.get(acct_lower) or []
        rows.append({
            "acct": acct_lower,
            "bio": strip_html(bios.get(acct_lower, "")),
            "recent_probs": recent[:10],
            "v1_post_content": [],
        })
    return rows


def materialize_post_probs(
    rows: list[dict], scorer, batch_size: int = 256
) -> None:
    """For accts where we only have v1 post content, embed and run through
    classifier to get a probability. Mutates `rows` in-place: sets `mean_prob`.
    """
    from fedi_studio.services.embedder import embed_batch

    # Find rows missing v2 probs
    needs_embedding: list[tuple[int, list[str]]] = []
    for i, r in enumerate(rows):
        if not r["recent_probs"] and r["v1_post_content"]:
            needs_embedding.append((i, r["v1_post_content"]))

    log.info("embedding posts for %d accts with v1-only content", len(needs_embedding))
    # Flatten content to embed in batches
    all_texts: list[str] = []
    text_to_acct_idx: list[int] = []  # which acct each text belongs to
    for acct_i, contents in needs_embedding:
        for c in contents:
            all_texts.append(strip_html(c)[:1000])
            text_to_acct_idx.append(acct_i)

    if not all_texts:
        return

    # Embed in batches
    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(all_texts), batch_size):
        chunk = all_texts[i : i + batch_size]
        emb = embed_batch(chunk)
        for j in range(emb.shape[0]):
            all_embeddings.append(emb[j])

    # Run through classifier
    if scorer._is_fit:
        try:
            X = np.vstack(all_embeddings)
            probs = scorer.classifier.predict_proba(X)[:, 1]
        except Exception as e:
            log.warning("classifier predict_proba failed: %s — using 0.5", e)
            probs = np.full(len(all_embeddings), 0.5)
    else:
        log.warning("classifier not fit; using 0.5 for v1-only posts")
        probs = np.full(len(all_embeddings), 0.5)

    # Aggregate per acct
    per_acct_probs: dict[int, list[float]] = {}
    for prob, acct_i in zip(probs, text_to_acct_idx):
        per_acct_probs.setdefault(acct_i, []).append(float(prob))

    for acct_i, plist in per_acct_probs.items():
        rows[acct_i]["recent_probs"] = plist[:10]


def evaluate_weights(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, label: str
) -> dict:
    """X is (N, 3) = mean_prob, centroid_sim, topic_bonus. y is 0/1."""
    scores = X @ w
    auc = roc_auc_score(y, scores)
    yhat = (scores >= 0.5).astype(int)
    p = precision_score(y, yhat, zero_division=0)
    r = recall_score(y, yhat, zero_division=0)
    cm = confusion_matrix(y, yhat)
    tn, fp, fn_, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    fpr = fp / max(fp + tn, 1)
    return {
        "label": label,
        "weights": w.tolist(),
        "auc": auc,
        "precision@0.5": p,
        "recall@0.5": r,
        "fpr@0.5": fpr,
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn_), "TP": int(tp)},
        "score_stats": {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "pos_mean": float(scores[y == 1].mean()) if (y == 1).any() else 0.0,
            "neg_mean": float(scores[y == 0].mean()) if (y == 0).any() else 0.0,
        },
    }


def _connect_with_retry(dsn: str, label: str):
    for attempt in range(8):
        try:
            return psycopg.connect(dsn, connect_timeout=15)
        except psycopg.OperationalError as e:
            log.warning("connect to %s failed (attempt %d): %s", label, attempt + 1, e)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"failed to connect to {label} after retries")


async def main_async(args):
    log.info("loading scorer + centroid")
    from fedi_studio.services.scorer import Scorer
    scorer = Scorer.load_or_initialize("models/scorer_v1.pkl")
    log.info("scorer fit=%s", scorer._is_fit)
    v1 = _connect_with_retry(V1_DSN, "v1")
    v2 = _connect_with_retry(V2_DSN, "v2")
    user_centroid = load_user_centroid(v2)
    log.info("centroid: %s, dim=%s", user_centroid is not None, None if user_centroid is None else user_centroid.shape)

    headers = {
        "Authorization": f"Bearer {MASTODON_TOKEN}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as client:
        who, _ = await _get_json(client, f"{MASTODON_URL}/api/v1/accounts/verify_credentials")
        if not isinstance(who, dict):
            raise RuntimeError("verify_credentials failed")
        tim_id = who["id"]
        log.info("tim id=%s following=%d", tim_id, who.get("following_count"))

        log.info("=== positives ===")
        positives = await gather_positives(client, tim_id, v1, v2)

    log.info("=== negatives ===")
    pos_set = {p["acct"] for p in positives}
    negatives = gather_negatives(v1, v2, n_target=args.n_neg, exclude=pos_set)

    # No materialization needed: positives use v2_probs (calibrated) or v1
    # post_scores.score (already pre-computed), same as production fallback.

    # Compute components
    log.info("computing components for %d pos + %d neg", len(positives), len(negatives))
    Xp, Xn = [], []
    pos_keep, neg_keep = [], []
    for r in positives:
        # Drop accts with no posts at all - we have no signal to evaluate them on
        if not r["recent_probs"] and not r["bio"]:
            continue
        mp, cs, tb = compute_components(r["bio"], r["recent_probs"], user_centroid)
        Xp.append([mp, cs, tb])
        pos_keep.append(r["acct"])
    for r in negatives:
        if not r["recent_probs"] and not r["bio"]:
            continue
        mp, cs, tb = compute_components(r["bio"], r["recent_probs"], user_centroid)
        Xn.append([mp, cs, tb])
        neg_keep.append(r["acct"])

    Xp = np.array(Xp)
    Xn = np.array(Xn)
    X = np.vstack([Xp, Xn])
    y = np.array([1] * len(Xp) + [0] * len(Xn))
    log.info("final dataset: %d positives, %d negatives", len(Xp), len(Xn))

    # --- Univariate AUCs ---
    print("\n=== Per-component univariate AUCs ===")
    for i, name in enumerate(["mean_prob", "centroid_sim", "topic_bonus"]):
        try:
            auc_i = roc_auc_score(y, X[:, i])
        except Exception as e:
            auc_i = float("nan")
        print(f"  {name}: AUC = {auc_i:.4f}, mean(pos)={X[y==1, i].mean():.4f}, mean(neg)={X[y==0, i].mean():.4f}")

    # --- Current weights ---
    print("\n=== Current weights (0.55, 0.30, 0.15) ===")
    w_current = np.array([0.55, 0.30, 0.15])
    res_cur = evaluate_weights(X, y, w_current, "current")
    print(json.dumps(res_cur, indent=2))

    # --- Logistic regression to find optimal weights ---
    print("\n=== Fitting logistic regression ===")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X, y)
    print(f"  raw coefficients: {lr.coef_[0]}")
    print(f"  intercept: {lr.intercept_[0]}")
    # Convert to weights normalized to sum=1 to keep score in similar [0,1] scale
    raw_coefs = lr.coef_[0]
    pos_coefs = np.clip(raw_coefs, a_min=0, a_max=None)
    if pos_coefs.sum() > 0:
        w_norm = pos_coefs / pos_coefs.sum()
    else:
        w_norm = w_current
    print(f"  normalized non-negative weights (sum=1): {w_norm}")

    # Evaluate normalized weights
    res_norm = evaluate_weights(X, y, w_norm, "lr_normalized")
    print(json.dumps(res_norm, indent=2))

    # Also evaluate using LR predicted probability directly (not just weighted sum)
    lr_proba = lr.predict_proba(X)[:, 1]
    auc_lr_direct = roc_auc_score(y, lr_proba)
    print(f"\n=== LR direct predict_proba AUC: {auc_lr_direct:.4f} ===")

    # --- Decision ---
    delta_auc = res_norm["auc"] - res_cur["auc"]
    delta_lr = auc_lr_direct - res_cur["auc"]
    print(f"\n=== Decision ===")
    print(f"  current AUC = {res_cur['auc']:.4f}")
    print(f"  LR-normalized weighted AUC = {res_norm['auc']:.4f} (delta = {delta_auc:+.4f})")
    print(f"  LR direct AUC = {auc_lr_direct:.4f} (delta = {delta_lr:+.4f})")
    if delta_auc > 0.05:
        print("  >>> DEPLOY new weights (delta > 0.05)")
    else:
        print("  >>> KEEP current weights (delta <= 0.05)")

    # Save full result for later use
    out = {
        "n_pos": len(Xp),
        "n_neg": len(Xn),
        "univariate_aucs": {
            "mean_prob": float(roc_auc_score(y, X[:, 0])),
            "centroid_sim": float(roc_auc_score(y, X[:, 1])),
            "topic_bonus": float(roc_auc_score(y, X[:, 2])),
        },
        "current": res_cur,
        "lr_normalized": res_norm,
        "lr_direct_auc": float(auc_lr_direct),
        "lr_raw_coefs": lr.coef_[0].tolist(),
        "lr_intercept": float(lr.intercept_[0]),
    }
    out_path = Path("/tmp/weights_validation.json")
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")

    # Save the labeled data so we can re-fit without re-fetching
    np.savez(
        "/tmp/weights_validation_data.npz",
        X=X,
        y=y,
        pos_accts=np.array(pos_keep, dtype=object),
        neg_accts=np.array(neg_keep, dtype=object),
    )
    print("Wrote /tmp/weights_validation_data.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neg", type=int, default=3000)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
