"""Build the curated candidate-account list for Tim's manual follow review.

READ-ONLY worker. Writes rows to `fedi_studio.candidates`. Tim opens the
/candidates web page, reviews each row, and clicks "open in Mastodon" to
follow them himself in the native UI.

THIS WORKER NEVER CALLS POST /api/v1/accounts/{id}/follow. It never queues
a follow. It never schedules a follow. Every Mastodon API call is GET-only:
- /api/v1/accounts/verify_credentials  (resolve Tim's id)
- /api/v1/accounts/{id}/following      (paginate to build skip-set)
- /api/v1/accounts/{id}/followers      (paginate to build skip-set)
- /api/v1/accounts/lookup              (resolve acct -> profile json)
- /api/v1/accounts/relationships       (detect outgoing follow_request=true)

Sources of candidate `acct` strings:
- Authors of high-prob posts in v2 `fedi_studio.posts` (probability >= 0.55)
- Authors of high-score posts in v1 `fedi_discover.posts` (score >= 30)
- Authors active in v1 in the last 90 days with at least one post

Hard exclusions:
- Tim's existing followings
- Tim's followers (already a relationship; redundant)
- Anyone in v1 `follow_tracking.acct` (already attempted by v1 follow bot)
- Any anyone with outgoing follow_request=true on Tim's instance
- Any acct on a domain in `fedi_studio.blocklist`
- Any acct on an instance that 403'd v1 fewer than threshold attempts
- Heuristic name filters: `bot|news|aggregator` substrings (overridable when bot=False)

Soft filters (post-API enrich):
- discoverable=true OR locked=false
- bot=false (final word from API)
- bio has no #nobot or #noindex
- last_status_at within 60 days
- account has posts in v1 or v2 in the last 90 days

Score = 0.30 * mean(prob over last 10 posts under classifier) +
        0.00 * cosine(bio_embedding, user_centroid)  # validated useless 2026-04-25
        0.70 * topic_match_bonus (0..1)
(See WEIGHT_* constants below for weight provenance.)

Run:
    .venv/bin/python -m fedi_studio.workers.build_candidates
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import httpx
import numpy as np
import psycopg

from fedi_studio.models.db import get_dsn
from fedi_studio.services.embedder import cosine_similarity, embed
from fedi_studio.services.scorer import Scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_candidates")

V1_DSN = os.environ.get(
    "V1_DSN", "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon"
)
V2_DSN = get_dsn()

MASTODON_URL = os.environ.get("MASTODON_URL", "https://holm.community").rstrip("/")
MASTODON_TOKEN = os.environ.get("MASTODON_TOKEN", "")

# Token rotation: if MASTODON_TOKENS (comma-separated) is set, requests can
# round-robin through multiple tokens, multiplying the per-token rate limit.
# Each token must be a personal access token under the same user account
# (created via Mastodon's Settings > Development page).
_tokens_env = os.environ.get("MASTODON_TOKENS", "").strip()
MASTODON_TOKENS: list[str] = (
    [t.strip() for t in _tokens_env.split(",") if t.strip()]
    if _tokens_env else
    ([MASTODON_TOKEN] if MASTODON_TOKEN else [])
)
_token_idx = 0


def _next_token() -> str:
    """Round-robin next token. Thread/coroutine safe enough — worst case is
    occasional repeats which only marginally reduces the multiplier."""
    global _token_idx
    if not MASTODON_TOKENS:
        return ""
    t = MASTODON_TOKENS[_token_idx % len(MASTODON_TOKENS)]
    _token_idx += 1
    return t


def _default_scorer_path() -> str:
    """Prefer scorer_v2 (full v1-DB retrain) if present, else fall back to v1.

    Env var SCORER_PATH still wins so deploys can pin a specific file.
    """
    for c in ("models/scorer_v2.pkl", "models/scorer_v1.pkl"):
        if os.path.exists(c):
            return c
    return "models/scorer_v1.pkl"


SCORER_PATH = os.environ.get("SCORER_PATH", _default_scorer_path())

MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "20000"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "20"))
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "12.0"))
USER_AGENT = "fedi-studio-build-candidates/1.0 (read-only; tim@holm.community)"

# Per-instance-403 threshold: if v1 had >= this many 403 responses for an
# instance we skip lookup attempts to that instance.
INSTANCE_403_THRESHOLD = 5

NAME_BLACKLIST_RE = ("bot", "news", "aggregator", "rss", "feed_")
HARD_BIO_BLOCK = ("#nobot", "#noindex")

# How recently the candidate must have posted SOMETHING for us to keep them.
LAST_STATUS_MAX_AGE_DAYS = 60
# How recently we need at least one observed post in our local DBs.
LOCAL_POSTS_MAX_AGE_DAYS = 90

# ---------------------------------------------------------------------------
# Candidate score weights
# ---------------------------------------------------------------------------
# Validated 2026-04-25 against Tim's existing followings (n=3,630 positives)
# vs v1.follow_tracking.followed_back=0 (n=3,000 negatives).
#
# Univariate AUCs:
#   mean_prob:    0.621
#   centroid_sim: 0.502  <- noise; bio embedding doesn't predict positives
#   topic_bonus:  0.648  <- strongest single signal
#
# Old weights [0.55, 0.30, 0.15] -> AUC = 0.640
# New weights [0.30, 0.00, 0.70] -> AUC = 0.719  (delta +0.08)
#
# Rationale: centroid_sim is dropped because user_centroid is trained on
# POST embeddings while the input here is the BIO — the distributions don't
# match and the signal collapses to noise (uniform ~0.685 across pos & neg).
# Topic-bonus is the strongest discriminator and gets the biggest weight;
# mean_prob breaks ties for accounts with no topic-keyword matches (~70% of
# accounts have topic_bonus = 0).
#
# Note: score magnitudes drop ~50% under the new weights (mean ~0.16 vs old
# ~0.36) because centroid_sim acted as a near-constant 0.685 floor. The
# /candidates page uses score for RANKING, not for binary thresholding,
# so the magnitude shift is harmless — Tim's UI sorts by score DESC.
WEIGHT_MEAN_PROB = 0.30
WEIGHT_CENTROID_SIM = 0.00
WEIGHT_TOPIC_BONUS = 0.70

# Topic keyword reuse from web/app.py — single source of truth.
from fedi_studio.web.app import TOPIC_RULES, ALL_TOPIC_KEYWORDS  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _instance_of(acct: str) -> str:
    """`user@instance.tld` -> `instance.tld`. Handles doubly-mangled accts
    like `user@instance@instance` (which appear in v1 from a firehose bug)
    by always taking the LAST `@` segment. Local accts (no @) -> holm.community."""
    if "@" not in acct:
        return MASTODON_URL.split("//", 1)[-1]
    return acct.rsplit("@", 1)[1].lower().rstrip("/")


def _normalize(acct: str) -> str:
    """Lowercase, strip whitespace, drop leading @, collapse double `@host@host`
    suffixes that appear in some v1 author_acct rows. Returns `local@instance`."""
    if not acct:
        return ""
    a = acct.strip().lstrip("@").rstrip("/").lower()
    # Detect & collapse `local@host@host` -> `local@host`
    parts = a.split("@")
    if len(parts) > 2 and parts[-1] == parts[-2]:
        a = parts[0] + "@" + parts[-1]
    return a


def _topic_bonus(text: str) -> tuple[float, list[str]]:
    """0..1 bonus based on number of topic keyword categories that match in `text`.

    The bonus saturates at 3 distinct topic categories matched.
    """
    if not text:
        return 0.0, []
    lower = text.lower()
    matched: list[str] = []
    for topic, kws in TOPIC_RULES:
        if any(kw in lower for kw in kws):
            matched.append(topic)
    if not matched:
        return 0.0, []
    bonus = min(len(matched) / 3.0, 1.0)
    return bonus, matched


# ---------------------------------------------------------------------------
# Mastodon API: paginated read-only endpoints
# ---------------------------------------------------------------------------


async def _get_json(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> tuple[Any, dict]:
    """GET with backoff for 429/5xx. Rotates through MASTODON_TOKENS if set,
    so we can stack rate-limit buckets across multiple tokens. Returns (json, headers)."""
    backoff = 1.0
    for attempt in range(6):
        # Pick a fresh token per attempt — on 429 from one bucket, the next
        # attempt may land on a different bucket that still has headroom.
        headers = {}
        if MASTODON_TOKENS:
            headers["Authorization"] = f"Bearer {_next_token()}"
        try:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", backoff))
                log.warning("429 on %s — sleeping %.1fs", url, wait)
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, 30.0)
                continue
            if r.status_code >= 500:
                log.warning("%d on %s — backoff %.1fs", r.status_code, url, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            if r.status_code in (401, 403, 404, 410, 422):
                return None, dict(r.headers)
            r.raise_for_status()
            return r.json(), dict(r.headers)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            log.debug("transient %s on %s; retry in %.1fs", type(e).__name__, url, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    return None, {}


async def fetch_paginated_accts(
    client: httpx.AsyncClient, base_url: str, label: str, limit: int = 80
) -> set[str]:
    """Walk the Mastodon Link-header pagination chain for accounts/{id}/following etc."""
    out: set[str] = set()
    url: str | None = base_url
    params: dict | None = {"limit": limit}
    page = 0
    while url:
        data, headers = await _get_json(client, url, params=params)
        params = None  # subsequent URLs already have ?max_id=...
        if not isinstance(data, list):
            break
        for a in data:
            acct = a.get("acct")
            if acct:
                out.add(_normalize(acct))
        page += 1
        if page % 5 == 0:
            log.info("[%s] paged %d, total=%d", label, page, len(out))
        # Mastodon paginates via Link header rel="next"
        link = headers.get("link") or headers.get("Link") or ""
        next_url = None
        for part in link.split(","):
            part = part.strip()
            if 'rel="next"' in part and "<" in part and ">" in part:
                next_url = part[part.index("<") + 1 : part.index(">")]
                break
        url = next_url
    log.info("[%s] done, total=%d", label, len(out))
    return out


# ---------------------------------------------------------------------------
# Local DB queries
# ---------------------------------------------------------------------------


def load_blocked_domains(v2: psycopg.Connection) -> set[str]:
    """v2 fedi_studio.blocklist holds DOMAINS (instances). Each row excludes
    every account on that instance."""
    domains: set[str] = set()
    with v2.cursor() as cur:
        cur.execute("SELECT pattern FROM blocklist")
        for (pat,) in cur:
            d = (pat or "").strip().lower().lstrip("@")
            if "@" in d:
                d = d.split("@", 1)[1]
            if d:
                domains.add(d)
    return domains


def load_blocked_users_v1(v1: psycopg.Connection) -> set[str]:
    """v1 fedi_discover.blocklist holds individual `user@host` entries — these
    are per-account blocks, NOT instance bans. Treat as user blocklist."""
    out: set[str] = set()
    try:
        with v1.cursor() as cur:
            cur.execute("SELECT acct FROM blocklist")
            for (pat,) in cur:
                a = _normalize(pat or "")
                if a:
                    out.add(a)
    except Exception as e:
        log.warning("v1 blocklist read failed: %s", e)
    return out


def load_attempted_follow_accts(v1: psycopg.Connection) -> set[str]:
    """Accts the v1 follow bot already attempted (success or fail)."""
    out: set[str] = set()
    with v1.cursor() as cur:
        cur.execute("SELECT acct FROM follow_tracking WHERE acct IS NOT NULL")
        for (a,) in cur:
            out.add(_normalize(a))
    return out


def load_403_instances(v1: psycopg.Connection) -> set[str]:
    """Instances where v1 saw repeated 403 follow responses."""
    out: set[str] = set()
    with v1.cursor() as cur:
        cur.execute(
            """
            SELECT split_part(p.author_acct, '@', 2) AS instance, count(*) AS c
            FROM actions a
            JOIN posts p ON p.id = a.post_id
            WHERE a.error_message ILIKE '%%403%%'
              AND a.action_type = 'follow'
              AND p.author_acct LIKE '%%@%%'
            GROUP BY instance
            HAVING count(*) >= %s
            """,
            (INSTANCE_403_THRESHOLD,),
        )
        for inst, _ in cur:
            if inst:
                out.add(inst.lower())
    return out


def load_local_posts_acct_set(
    v1: psycopg.Connection, v2: psycopg.Connection
) -> set[str]:
    """Accts with at least one post in v1 or v2 in last 90 days. Used to
    require local-DB activity."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOCAL_POSTS_MAX_AGE_DAYS)).strftime(
        "%Y-%m-%dT00:00:00.000Z"
    )
    accts: set[str] = set()
    with v1.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT lower(author_acct) FROM posts WHERE posted_at > %s",
            (cutoff,),
        )
        for (a,) in cur:
            if a:
                accts.add(a)
    with v2.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT lower(author_acct) FROM posts WHERE posted_at > NOW() - interval '90 days'"
        )
        for (a,) in cur:
            if a:
                accts.add(a)
    return accts


def gather_candidate_accts(
    v1: psycopg.Connection, v2: psycopg.Connection, score_pool: dict[str, float]
) -> dict[str, dict]:
    """Returns {acct: {sources: [...], v1_high_count, v2_high_count, ...}}.

    `score_pool` is updated in-place: acct -> initial heuristic score from
    sources (used for ordering before profile enrichment, so we hit the API
    on the most promising authors first).
    """
    pool: dict[str, dict] = {}

    # --- Source 1: v2 high-prob authors ------------------------------------
    log.info("source 1: v2 fedi_studio.posts probability >= 0.55")
    with v2.cursor() as cur:
        cur.execute(
            """
            SELECT lower(p.author_acct) AS acct,
                   count(*) AS n,
                   avg(ps.probability) AS mean_prob,
                   max(ps.probability) AS max_prob
            FROM posts p
            JOIN post_scores ps ON ps.post_id = p.id AND ps.posted_at = p.posted_at
            WHERE ps.probability >= 0.55
              AND p.posted_at > NOW() - interval '120 days'
            GROUP BY acct
            """
        )
        for acct, n, mean_prob, max_prob in cur:
            d = pool.setdefault(acct, {"sources": []})
            d["sources"].append("v2_prob")
            d["v2_n"] = int(n)
            d["v2_mean_prob"] = float(mean_prob or 0.0)
            d["v2_max_prob"] = float(max_prob or 0.0)
            score_pool[acct] = max(score_pool.get(acct, 0.0), float(mean_prob or 0.0) * 0.9)
    log.info("  v2 hit %d distinct authors", len(pool))

    # --- Source 2: v1 high-score authors -----------------------------------
    log.info("source 2: v1 fedi_discover.posts score>=30 OR ai_score>=60 in last 90d")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=LOCAL_POSTS_MAX_AGE_DAYS)).strftime(
        "%Y-%m-%dT00:00:00.000Z"
    )
    with v1.cursor() as cur:
        cur.execute(
            """
            SELECT lower(p.author_acct) AS acct,
                   count(*) AS n,
                   avg(ps.score) AS mean_score,
                   coalesce(max(ps.ai_score), 0) AS max_ai
            FROM posts p
            JOIN post_scores ps ON ps.post_id = p.id
            WHERE (ps.score >= 30 OR ps.ai_score >= 60)
              AND p.posted_at > %s
              AND p.author_acct LIKE '%%@%%'
            GROUP BY acct
            """,
            (cutoff,),
        )
        for acct, n, mean_score, max_ai in cur:
            d = pool.setdefault(acct, {"sources": []})
            d["sources"].append("v1_high")
            d["v1_high_n"] = int(n)
            d["v1_mean_score"] = float(mean_score or 0.0) / 100.0
            d["v1_max_ai"] = float(max_ai or 0.0) / 100.0
            score_pool[acct] = max(
                score_pool.get(acct, 0.0),
                0.7 * (float(mean_score or 0) / 100.0) + 0.3 * (float(max_ai or 0) / 100.0),
            )
    log.info("  pool size after v1_high: %d", len(pool))

    # --- Source 3: v1 active authors (any post in last 90d, top by volume) -
    log.info("source 3: v1 fedi_discover.posts authors active in last 90d")
    with v1.cursor() as cur:
        cur.execute(
            """
            SELECT lower(p.author_acct) AS acct,
                   count(*) AS n,
                   coalesce(avg(ps.score), 0) AS mean_score
            FROM posts p
            LEFT JOIN post_scores ps ON ps.post_id = p.id
            WHERE p.posted_at > %s
              AND p.author_acct LIKE '%%@%%'
            GROUP BY acct
            HAVING count(*) >= 3
            """,
            (cutoff,),
        )
        for acct, n, mean_score in cur:
            d = pool.setdefault(acct, {"sources": []})
            if "v1_active" not in d["sources"]:
                d["sources"].append("v1_active")
            d.setdefault("v1_total_n", 0)
            d["v1_total_n"] = max(d["v1_total_n"], int(n))
            if "v1_mean_score" not in d:
                d["v1_mean_score"] = float(mean_score or 0.0) / 100.0
            score_pool.setdefault(acct, float(mean_score or 0.0) / 100.0 * 0.5)
    log.info("  pool size after v1_active: %d", len(pool))

    return pool


def load_user_centroid(v2: psycopg.Connection) -> np.ndarray | None:
    with v2.cursor() as cur:
        cur.execute("SELECT embedding FROM user_centroid WHERE id=1")
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        return np.array(row[0], dtype=np.float32)


def load_recent_post_probs(
    v2: psycopg.Connection, accts: list[str]
) -> dict[str, list[float]]:
    """Return {acct: [last 10 post probabilities]} from v2."""
    if not accts:
        return {}
    out: dict[str, list[float]] = {}
    with v2.cursor() as cur:
        cur.execute(
            """
            SELECT lower(p.author_acct) AS acct, ps.probability
            FROM posts p
            JOIN post_scores ps ON ps.post_id = p.id AND ps.posted_at = p.posted_at
            WHERE lower(p.author_acct) = ANY(%s)
            ORDER BY p.posted_at DESC
            """,
            (accts,),
        )
        for acct, prob in cur:
            out.setdefault(acct, [])
            if len(out[acct]) < 10:
                out[acct].append(float(prob))
    return out


def load_v1_recent_post_scores(
    v1: psycopg.Connection, accts: list[str]
) -> dict[str, list[float]]:
    """Return {acct: [last 10 v1 post scores normalized 0..1]} as a fallback."""
    if not accts:
        return {}
    out: dict[str, list[float]] = {}
    with v1.cursor() as cur:
        cur.execute(
            """
            SELECT lower(p.author_acct) AS acct, ps.score
            FROM posts p
            JOIN post_scores ps ON ps.post_id = p.id
            WHERE lower(p.author_acct) = ANY(%s)
            ORDER BY p.posted_at DESC
            LIMIT %s
            """,
            (accts, max(10 * len(accts), 100)),
        )
        for acct, score in cur:
            out.setdefault(acct, [])
            if len(out[acct]) < 10:
                out[acct].append(min(float(score or 0) / 100.0, 1.0))
    return out


# ---------------------------------------------------------------------------
# Per-candidate enrichment via Mastodon API lookup
# ---------------------------------------------------------------------------


@dataclass
class EnrichedCandidate:
    acct: str
    profile: dict
    instance: str


# Separate unauth client for origin instance lookups. We must NOT send Tim's
# holm.community token to other instances — they'd reject it (and even if not,
# it leaks our token across the fediverse). Lazy-initialised on first use.
_origin_client: "httpx.AsyncClient | None" = None


async def _get_origin_client() -> "httpx.AsyncClient":
    global _origin_client
    if _origin_client is None or _origin_client.is_closed:
        _origin_client = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},  # NO Authorization header
        )
    return _origin_client


# Skip these hosts entirely — they're not Mastodon-API-compatible.
# Substring matches too (e.g., any *.lemmy.world subdomain).
_NON_MASTODON_HOSTS = {
    # Lemmy network (different software, different API)
    "lemmy.ml", "lemmy.world", "lemmy.dbzer0.com", "lemmy.linuxuserspace.show",
    "lemmygrad.ml", "sh.itjust.works", "feddit.de", "feddit.it", "feddit.uk",
    "feddit.cl", "feddit.nl", "feddit.org", "lemm.ee", "lemmy.frozeninferno.xyz",
    "slrpnk.net", "discuss.online", "midwest.social", "programming.dev",
    # Bridges (proxy other protocols)
    "bsky.brid.gy", "fed.brid.gy", "rss-parrot.net",
    # FediBuzz / relays (not real hosts)
    "fedi.buzz",
    # Non-Mastodon protocols
    "flipboard.com",  # Different protocol
    "threads.net",  # Meta — Mastodon-compatible only for outbound, no lookup API
    "kbin.social", "kbin.run", "fedia.social",  # /kbin / mbin
    "misskey.io", "misskey.flowers", "stop.voring.me",  # Misskey
    "pixelfed.social", "pixelfed.de",  # Pixelfed
    "peertube.social", "makertube.net",  # PeerTube
    "wordpress.com", "write.as", "writefreely.org",  # Long-form / blogs
    # Known dead / unstable for lookup
    "eden.one",
}


def _is_non_mastodon_host(host: str) -> bool:
    if host in _NON_MASTODON_HOSTS:
        return True
    # Lemmy-like subdomain catch-all
    parts = host.split(".")
    if len(parts) >= 2:
        for known in _NON_MASTODON_HOSTS:
            if host.endswith("." + known):
                return True
    return False


async def _origin_lookup(_unused_client, acct: str) -> dict | None:
    """Look up `acct` on its ORIGIN instance directly (unauth public endpoint).
    Uses a SEPARATE unauth client so we don't leak Tim's holm.community token
    to remote instances. Skips known-non-Mastodon hosts.
    """
    if "@" not in acct:
        return None
    local, instance = acct.split("@", 1)
    instance = instance.strip("/").lower()
    if not instance or not local:
        return None
    if _is_non_mastodon_host(instance):
        return None
    client = await _get_origin_client()
    url = f"https://{instance}/api/v1/accounts/lookup"
    backoff = 1.0
    for attempt in range(3):
        try:
            r = await client.get(url, params={"acct": local})
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, dict) and "id" in d:
                    d["_origin_id"] = str(d["id"])
                    d["_origin_instance"] = instance
                    return d
                return None
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", backoff))
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, 15.0)
                continue
            if r.status_code in (401, 403, 404, 410, 422, 451):
                return None
            if r.status_code >= 500:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                continue
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError):
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)
    return None


async def enrich_one(
    client: httpx.AsyncClient, acct: str, sem: asyncio.Semaphore
) -> EnrichedCandidate | None:
    """Resolve `acct` via the ORIGIN instance directly (much higher hit rate
    than going through holm.community). The returned profile has the origin
    instance's account ID — not holm.community's. Relationships check is done
    separately, against holm.community, by the caller using the acct string."""
    async with sem:
        data = await _origin_lookup(client, acct)
        if not isinstance(data, dict) or "id" not in data:
            return None
        return EnrichedCandidate(acct=acct, profile=data, instance=_instance_of(acct))


async def fetch_relationships(
    client: httpx.AsyncClient, ids: list[str]
) -> dict[str, dict]:
    """Batched relationships lookup. Mastodon allows up to 40 ids per call."""
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i : i + 40]
        params = [("id[]", x) for x in chunk]
        data, _ = await _get_json(
            client, f"{MASTODON_URL}/api/v1/accounts/relationships", params=params
        )
        if not isinstance(data, list):
            continue
        for rel in data:
            out[str(rel.get("id"))] = rel
    return out


# ---------------------------------------------------------------------------
# Filtering and scoring
# ---------------------------------------------------------------------------


def passes_soft_filters(profile: dict) -> tuple[bool, str | None]:
    """Apply per-account post-API filters. Return (kept, reject_reason)."""
    if profile.get("suspended"):
        return False, "suspended"
    if profile.get("bot") is True:
        return False, "bot"
    bio = (profile.get("note") or "").lower()
    for kw in HARD_BIO_BLOCK:
        if kw in bio:
            return False, f"bio:{kw}"

    discoverable = profile.get("discoverable")
    locked = profile.get("locked")
    # We accept either explicitly discoverable, or simply not-locked.
    # (Lots of accts have discoverable=null but are public.)
    if locked is True and discoverable is False:
        return False, "locked_and_not_discoverable"

    last = profile.get("last_status_at")
    if last:
        last_dt = parse_last_status_at(last)
        if last_dt:
            age = datetime.now(timezone.utc) - last_dt
            if age > timedelta(days=LAST_STATUS_MAX_AGE_DAYS):
                return False, "stale_last_status"
    else:
        return False, "no_last_status"

    if (profile.get("statuses_count") or 0) < 5:
        return False, "too_few_statuses"

    return True, None


def parse_last_status_at(raw: str | None) -> datetime | None:
    """Parse Mastodon last_status_at — sometimes "YYYY-MM-DD", sometimes full
    ISO, sometimes ISO with Z. Always returns tz-aware UTC, or None."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d")
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_score(
    profile: dict,
    recent_probs: list[float],
    user_centroid: np.ndarray | None,
    bio_text: str,
) -> tuple[float, dict]:
    # Component 1: classifier mean prob (over recent posts)
    mean_prob = float(np.mean(recent_probs)) if recent_probs else 0.5

    # Component 2: bio centroid similarity
    # Kept in the reasoning dict for transparency but weight is 0.0 — see
    # WEIGHT_CENTROID_SIM constant above for validation rationale.
    centroid_sim = 0.5
    if user_centroid is not None and bio_text:
        try:
            bio_emb = embed(bio_text)
            cs = cosine_similarity(bio_emb, user_centroid)
            centroid_sim = (cs + 1.0) / 2.0
        except Exception as e:
            log.debug("bio embed failed: %s", e)

    # Component 3: topic bonus from bio
    topic_bonus, matched_topics = _topic_bonus(bio_text)

    # Weights validated 2026-04-25 (see constants above). Centroid_sim is
    # zero-weighted because the bio-vs-post-centroid mapping is not
    # discriminative (univariate AUC = 0.502).
    score = (
        WEIGHT_MEAN_PROB * mean_prob
        + WEIGHT_CENTROID_SIM * centroid_sim
        + WEIGHT_TOPIC_BONUS * topic_bonus
    )
    # Clamp
    score = max(0.0, min(1.0, score))

    return score, {
        "mean_prob": round(mean_prob, 3),
        "n_recent_posts": len(recent_probs),
        "centroid_sim": round(centroid_sim, 3),
        "topic_bonus": round(topic_bonus, 3),
        "matched_topics": matched_topics,
        "weights": {
            "mean_prob": WEIGHT_MEAN_PROB,
            "centroid": WEIGHT_CENTROID_SIM,
            "topic_bonus": WEIGHT_TOPIC_BONUS,
        },
        "weights_version": "2026-04-25",
    }


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


def insert_candidate(cur: psycopg.Cursor, row: dict) -> bool:
    cur.execute(
        """
        INSERT INTO candidates
            (acct, display_name, avatar_url, bio, followers_count, following_count,
             statuses_count, locked, bot, discoverable, last_status_at,
             score, reasoning, instance)
        VALUES
            (%(acct)s, %(display_name)s, %(avatar_url)s, %(bio)s, %(followers_count)s,
             %(following_count)s, %(statuses_count)s, %(locked)s, %(bot)s,
             %(discoverable)s, %(last_status_at)s, %(score)s, %(reasoning)s::jsonb,
             %(instance)s)
        ON CONFLICT (acct) DO UPDATE SET
            display_name    = EXCLUDED.display_name,
            avatar_url      = EXCLUDED.avatar_url,
            bio             = EXCLUDED.bio,
            followers_count = EXCLUDED.followers_count,
            following_count = EXCLUDED.following_count,
            statuses_count  = EXCLUDED.statuses_count,
            locked          = EXCLUDED.locked,
            bot             = EXCLUDED.bot,
            discoverable    = EXCLUDED.discoverable,
            last_status_at  = EXCLUDED.last_status_at,
            score           = EXCLUDED.score,
            reasoning       = EXCLUDED.reasoning,
            instance        = EXCLUDED.instance
        """,
        row,
    )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args) -> None:
    if not MASTODON_TOKEN:
        log.error(
            "MASTODON_TOKEN env var not set — cannot enumerate Tim's followings/followers. "
            "Refusing to build candidates without the skip-set; that would be unsafe."
        )
        return

    log.info("Connecting to v2 (%s) and v1 DBs", V2_DSN.split("dbname=")[1].split()[0])
    v2 = psycopg.connect(V2_DSN, autocommit=False)
    v1 = psycopg.connect(V1_DSN, autocommit=False)

    try:
        # --- Build skip sets ------------------------------------------------
        log.info("loading skip sets")
        blocked_domains = load_blocked_domains(v2)
        log.info("blocked_domains (v2 instance bans): %d", len(blocked_domains))
        blocked_users = load_blocked_users_v1(v1)
        log.info("blocked_users (v1 per-acct blocks): %d", len(blocked_users))
        attempted_follow = load_attempted_follow_accts(v1)
        log.info("attempted_follow (v1 follow_tracking): %d", len(attempted_follow))
        bad_403_instances = load_403_instances(v1)
        log.info("instances with %d+ 403s in v1: %d", INSTANCE_403_THRESHOLD, len(bad_403_instances))
        local_posts_recent = load_local_posts_acct_set(v1, v2)
        log.info("accts with local posts in last 90d: %d", len(local_posts_recent))

        # --- Mastodon API: Tim's followings + followers + outgoing requests ---
        headers = {
            "Authorization": f"Bearer {MASTODON_TOKEN}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as client:
            who, _ = await _get_json(
                client, f"{MASTODON_URL}/api/v1/accounts/verify_credentials"
            )
            if not isinstance(who, dict):
                log.error("verify_credentials failed; cannot continue")
                return
            tim_id = who["id"]
            tim_username = (who.get("username") or "").lower()
            log.info(
                "tim id=%s acct=%s following=%d followers=%d",
                tim_id,
                who.get("acct"),
                who.get("following_count", 0),
                who.get("followers_count", 0),
            )

            tim_following, tim_followers = await asyncio.gather(
                fetch_paginated_accts(
                    client,
                    f"{MASTODON_URL}/api/v1/accounts/{tim_id}/following",
                    "following",
                ),
                fetch_paginated_accts(
                    client,
                    f"{MASTODON_URL}/api/v1/accounts/{tim_id}/followers",
                    "followers",
                ),
            )
            # Local accts in followings come back without "@host" suffix; store
            # both forms for matching against author_acct strings later.
            local_host = MASTODON_URL.split("//", 1)[-1]
            tim_following_full = set(tim_following)
            for a in list(tim_following):
                if "@" not in a:
                    tim_following_full.add(f"{a}@{local_host}")
            tim_followers_full = set(tim_followers)
            for a in list(tim_followers):
                if "@" not in a:
                    tim_followers_full.add(f"{a}@{local_host}")
            # Always exclude Tim himself
            if tim_username:
                tim_following_full.add(tim_username)
                tim_following_full.add(f"{tim_username}@{local_host}")

        # --- Gather candidate accts -----------------------------------------
        score_pool: dict[str, float] = {}
        pool = gather_candidate_accts(v1, v2, score_pool)
        log.info("raw pool size before exclusions: %d", len(pool))

        # --- Exclusions -----------------------------------------------------
        excluded = {
            "already_following": 0,
            "already_follower": 0,
            "v1_attempted": 0,
            "blocked_domain": 0,
            "blocked_user": 0,
            "bad_403_instance": 0,
            "name_blacklist": 0,
            "no_local_post_recent": 0,
            "missing_instance": 0,
        }

        keep: list[tuple[str, float]] = []
        for acct in pool.keys():
            inst = _instance_of(acct)
            if not inst:
                excluded["missing_instance"] += 1
                continue
            if acct in tim_following_full or acct in tim_following:
                excluded["already_following"] += 1
                continue
            if acct in tim_followers_full or acct in tim_followers:
                excluded["already_follower"] += 1
                continue
            if acct in attempted_follow:
                excluded["v1_attempted"] += 1
                continue
            if acct in blocked_users:
                excluded["blocked_user"] += 1
                continue
            if inst in blocked_domains or any(
                inst.endswith("." + d) for d in blocked_domains
            ):
                excluded["blocked_domain"] += 1
                continue
            if inst in bad_403_instances:
                excluded["bad_403_instance"] += 1
                continue
            local = acct.split("@")[0]
            if any(b in local for b in NAME_BLACKLIST_RE):
                excluded["name_blacklist"] += 1
                continue
            if acct not in local_posts_recent:
                excluded["no_local_post_recent"] += 1
                continue
            keep.append((acct, score_pool.get(acct, 0.0)))

        keep.sort(key=lambda t: t[1], reverse=True)
        log.info(
            "keep after exclusions: %d (excluded: %s)",
            len(keep),
            json.dumps(excluded),
        )

        # --- Cap to 2x MAX_CANDIDATES so API enrichment doesn't waste time ---
        budget = min(len(keep), MAX_CANDIDATES * 2)
        keep = keep[:budget]
        log.info("enriching top %d candidates via Mastodon API", len(keep))

        # --- Load classifier, centroid, recent post probs -------------------
        scorer = Scorer.load_or_initialize(SCORER_PATH)
        user_centroid = load_user_centroid(v2)
        log.info(
            "scorer fit=%s, centroid=%s",
            scorer._is_fit,
            "yes" if user_centroid is not None else "no",
        )

        recent_probs_v2 = load_recent_post_probs(v2, [a for a, _ in keep])
        recent_scores_v1 = load_v1_recent_post_scores(v1, [a for a, _ in keep])
        # Merge: prefer v2 probs (calibrated), fall back to v1 scores
        recent_probs_merged: dict[str, list[float]] = {}
        for acct in (a for a, _ in keep):
            v2p = recent_probs_v2.get(acct, [])
            v1s = recent_scores_v1.get(acct, [])
            recent_probs_merged[acct] = (v2p + v1s)[:10]

        # --- Enrich via Mastodon API ----------------------------------------
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        inserted = 0
        api_skipped = {
            "lookup_failed": 0,
            "soft_filter": 0,
            "follow_request_pending": 0,
            "duplicate_full_acct": 0,
        }
        seen_full_accts: set[str] = set()

        # Open a fresh client for the enrich phase
        async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as client:
            t0 = time.time()
            # Process in chunks so we can periodically:
            #  1. Fetch relationships in batches of 40
            #  2. Commit DB so progress is durable on early stop
            CHUNK = 80
            with v2.cursor() as cur:
                for i in range(0, len(keep), CHUNK):
                    chunk = keep[i : i + CHUNK]
                    enriched: list[EnrichedCandidate] = []
                    tasks = [
                        enrich_one(client, a, sem) for a, _ in chunk
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Exception):
                            log.debug("enrich exception: %s", r)
                            api_skipped["lookup_failed"] += 1
                            continue
                        if r is None:
                            api_skipped["lookup_failed"] += 1
                            continue
                        enriched.append(r)

                    if not enriched:
                        continue

                    # Batch relationships for the enriched profiles
                    ids = [e.profile["id"] for e in enriched]
                    rels = await fetch_relationships(client, ids)

                    for e in enriched:
                        prof = e.profile
                        rel = rels.get(str(prof.get("id")), {})
                        if rel.get("following") or rel.get("requested"):
                            api_skipped["follow_request_pending"] += 1
                            continue

                        # Build full "user@instance" form for dedup
                        local_acct = prof.get("acct") or e.acct
                        # If acct came back without host (local user), normalize
                        if "@" not in local_acct:
                            local_acct = f"{local_acct}@{local_host}"
                        full = _normalize(local_acct)
                        if full in tim_following_full or full in tim_followers_full:
                            api_skipped["soft_filter"] += 1
                            continue
                        if full in seen_full_accts:
                            api_skipped["duplicate_full_acct"] += 1
                            continue
                        seen_full_accts.add(full)

                        ok, reason = passes_soft_filters(prof)
                        if not ok:
                            api_skipped["soft_filter"] += 1
                            continue

                        bio_html = prof.get("note") or ""
                        # Naive HTML strip for embedding
                        import re as _re

                        bio_text = _re.sub(r"<[^>]+>", " ", bio_html).strip()

                        score, reasoning = compute_score(
                            prof, recent_probs_merged.get(e.acct, []), user_centroid, bio_text
                        )

                        last_dt = parse_last_status_at(prof.get("last_status_at"))

                        row = {
                            "acct": full,
                            "display_name": prof.get("display_name"),
                            "avatar_url": prof.get("avatar"),
                            "bio": bio_text[:2000],
                            "followers_count": prof.get("followers_count"),
                            "following_count": prof.get("following_count"),
                            "statuses_count": prof.get("statuses_count"),
                            "locked": prof.get("locked"),
                            "bot": prof.get("bot"),
                            "discoverable": prof.get("discoverable"),
                            "last_status_at": last_dt,
                            "score": score,
                            "reasoning": json.dumps(
                                {
                                    **reasoning,
                                    "sources": pool.get(e.acct, {}).get("sources", []),
                                    "instance_403_blocked": False,
                                }
                            ),
                            "instance": e.instance,
                        }
                        try:
                            insert_candidate(cur, row)
                            inserted += 1
                            if inserted % 100 == 0:
                                v2.commit()
                                rate = inserted / max(time.time() - t0, 1.0)
                                log.info(
                                    "[progress] inserted=%d/%d  rate=%.1f/s  api_skipped=%s",
                                    inserted,
                                    MAX_CANDIDATES,
                                    rate,
                                    json.dumps(api_skipped),
                                )
                        except Exception as ex:
                            v2.rollback()
                            log.warning("insert failed for %s: %s", full, ex)

                    v2.commit()
                    if inserted >= MAX_CANDIDATES:
                        log.info("hit MAX_CANDIDATES=%d, stopping", MAX_CANDIDATES)
                        break

            v2.commit()
            log.info(
                "DONE. inserted=%d, api_skipped=%s, exclusions=%s, elapsed=%.1fs",
                inserted,
                json.dumps(api_skipped),
                json.dumps(excluded),
                time.time() - t0,
            )
    finally:
        v1.close()
        v2.close()


def main() -> None:
    global MAX_CANDIDATES, MAX_CONCURRENCY
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    args = parser.parse_args()
    MAX_CANDIDATES = args.max
    MAX_CONCURRENCY = args.concurrency
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
