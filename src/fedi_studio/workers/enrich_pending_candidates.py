"""Continuously drain `candidates_pending` (populated by firehose) into `candidates`.

This is the auto-growth pipeline. The firehose inserts every newly-seen author into
`candidates_pending` (one row per unique acct). This daemon polls the queue, applies
the same exclusion + lookup + scoring pipeline as `build_candidates.py` (whose helpers
we import), and inserts qualifying accounts into `candidates`. Every author Tim sees
on /candidates was added by this loop, not by a one-shot run.

READ-ONLY against Mastodon. Reuses `build_candidates`'s GET-only API surface
(verify_credentials, accounts/lookup, accounts/relationships, accounts/{id}/following,
accounts/{id}/followers). Never calls POST /follow.

Run:
    MASTODON_TOKEN=... .venv/bin/python -m fedi_studio.workers.enrich_pending_candidates
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import psycopg

from fedi_studio.models.db import get_dsn
from fedi_studio.services.scorer import Scorer
from fedi_studio.workers.build_candidates import (
    HARD_BIO_BLOCK,
    HTTP_TIMEOUT,
    INSTANCE_403_THRESHOLD,
    NAME_BLACKLIST_RE,
    USER_AGENT,
    EnrichedCandidate,
    _get_json,
    _instance_of,
    _normalize,
    compute_score,
    enrich_one,
    fetch_paginated_accts,
    fetch_relationships,
    insert_candidate,
    load_403_instances,
    load_blocked_domains,
    load_blocked_users_v1,
    load_attempted_follow_accts,
    load_recent_post_probs,
    load_user_centroid,
    load_v1_recent_post_scores,
    parse_last_status_at,
    passes_soft_filters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enrich_candidates")

V1_DSN = os.environ.get(
    "V1_DSN",
    "host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon",
)
V2_DSN = get_dsn()

MASTODON_URL = os.environ.get("MASTODON_URL", "https://holm.community").rstrip("/")
MASTODON_TOKEN = os.environ.get("MASTODON_TOKEN", "")


def _default_scorer_path() -> str:
    """Prefer scorer_v2 (full v1-DB retrain) if present, else fall back to v1."""
    for c in ("models/scorer_v2.pkl", "models/scorer_v1.pkl"):
        if os.path.exists(c):
            return c
    return "models/scorer_v1.pkl"


SCORER_PATH = os.environ.get("SCORER_PATH", _default_scorer_path())

POLL_INTERVAL_S = int(os.environ.get("ENRICH_POLL_S", "60"))
BATCH_SIZE = int(os.environ.get("ENRICH_BATCH", "40"))
SKIPSET_REFRESH_S = int(os.environ.get("SKIPSET_REFRESH_S", "1800"))  # 30 min
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))

_running = True


def _stop(*_):
    global _running
    _running = False
    log.info("received stop signal")


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def fetch_pending_batch(v2: psycopg.Connection, n: int) -> list[tuple[int, str, str | None, str | None]]:
    """FOR UPDATE SKIP LOCKED makes this safe for parallel replicas: each
    enricher reserves its own slice with no overlap."""
    with v2.cursor() as cur:
        cur.execute(
            """
            SELECT id, acct, source_post_uri, holm_account_id
            FROM candidates_pending
            WHERE enriched_at IS NULL
            ORDER BY first_seen_at
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (n,),
        )
        return [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]


def mark_pending_outcome(
    v2: psycopg.Connection, ids: list[int], outcome: str
) -> None:
    if not ids:
        return
    with v2.cursor() as cur:
        cur.execute(
            """
            UPDATE candidates_pending
            SET enriched_at = NOW(), enriched_outcome = %s
            WHERE id = ANY(%s)
            """,
            (outcome, ids),
        )
    v2.commit()


def acct_already_in_candidates(v2: psycopg.Connection, accts: list[str]) -> set[str]:
    if not accts:
        return set()
    with v2.cursor() as cur:
        cur.execute(
            "SELECT acct FROM candidates WHERE acct = ANY(%s)", (accts,)
        )
        return {r[0] for r in cur.fetchall()}


async def fetch_tim_skipset(client: httpx.AsyncClient) -> tuple[set[str], set[str], str, str]:
    """Returns (following_full, followers_full, tim_id, tim_username)."""
    who, _ = await _get_json(client, f"{MASTODON_URL}/api/v1/accounts/verify_credentials")
    if not isinstance(who, dict):
        raise RuntimeError("verify_credentials failed")
    tim_id = str(who["id"])
    tim_username = (who.get("username") or "").lower()
    log.info(
        "tim id=%s acct=%s following=%d followers=%d",
        tim_id, who.get("acct"), who.get("following_count", 0), who.get("followers_count", 0),
    )
    following, followers = await asyncio.gather(
        fetch_paginated_accts(
            client, f"{MASTODON_URL}/api/v1/accounts/{tim_id}/following", "following"
        ),
        fetch_paginated_accts(
            client, f"{MASTODON_URL}/api/v1/accounts/{tim_id}/followers", "followers"
        ),
    )
    local_host = MASTODON_URL.split("//", 1)[-1]
    f_full = set(following)
    for a in list(following):
        if "@" not in a:
            f_full.add(f"{a}@{local_host}")
    f2_full = set(followers)
    for a in list(followers):
        if "@" not in a:
            f2_full.add(f"{a}@{local_host}")
    if tim_username:
        f_full.add(tim_username)
        f_full.add(f"{tim_username}@{local_host}")
    return f_full, f2_full, tim_id, tim_username


async def enrich_batch(
    v1: psycopg.Connection,
    v2: psycopg.Connection,
    pending: list[tuple[int, str, str | None, str | None]],
    skip_following: set[str],
    skip_followers: set[str],
    skip_attempted: set[str],
    skip_blocked_users: set[str],
    skip_blocked_domains: set[str],
    skip_403_instances: set[str],
    scorer: Scorer | None,
    user_centroid: np.ndarray | None,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> tuple[int, dict[str, int]]:
    """Enrich a batch. Returns (inserted_count, outcome_breakdown).

    Optimization: if holm_account_id is pre-populated (from follow-graph
    crawler), skip the /accounts/lookup call and fetch relationships directly.
    """
    if not pending:
        return 0, {}
    counts = {
        "already_in_candidates": 0,
        "already_following": 0,
        "already_follower": 0,
        "v1_attempted": 0,
        "blocked_user": 0,
        "blocked_domain": 0,
        "bad_403_instance": 0,
        "name_blacklist": 0,
        "missing_instance": 0,
        "lookup_failed": 0,
        "soft_filter_rejected": 0,
        "requested_outgoing": 0,
        "inserted": 0,
    }
    pending_by_acct: dict[str, list[int]] = {}
    account_ids_for_rel: dict[str, str] = {}  # acct.lower() -> holm_account_id
    for pid, acct, _src, holm_id in pending:
        pending_by_acct.setdefault(acct.lower(), []).append(pid)
        if holm_id:
            account_ids_for_rel[acct.lower()] = holm_id

    accts = list(pending_by_acct.keys())

    # Hard skip: already-inserted candidates
    already = acct_already_in_candidates(v2, accts)
    survivors: list[str] = []
    for a in accts:
        if a in already:
            counts["already_in_candidates"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:already_in_candidates")
            continue
        # Tim
        if a in skip_following:
            counts["already_following"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:already_following")
            continue
        if a in skip_followers:
            counts["already_follower"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:already_follower")
            continue
        if a in skip_attempted:
            counts["v1_attempted"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:v1_attempted")
            continue
        if a in skip_blocked_users:
            counts["blocked_user"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:blocked_user")
            continue
        inst = _instance_of(a)
        if not inst:
            counts["missing_instance"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:missing_instance")
            continue
        if inst in skip_blocked_domains or any(
            inst.endswith("." + d) for d in skip_blocked_domains
        ):
            counts["blocked_domain"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:blocked_domain")
            continue
        if inst in skip_403_instances:
            counts["bad_403_instance"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:bad_403_instance")
            continue
        local = a.split("@")[0]
        if any(b in local for b in NAME_BLACKLIST_RE):
            counts["name_blacklist"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "excluded:name_blacklist")
            continue
        survivors.append(a)

    if not survivors:
        return 0, counts

    # Enrich profiles
    enriched: list[EnrichedCandidate] = []
    results = await asyncio.gather(
        *(enrich_one(client, a, sem) for a in survivors), return_exceptions=True
    )
    for a, res in zip(survivors, results):
        if isinstance(res, EnrichedCandidate):
            enriched.append(res)
        else:
            counts["lookup_failed"] += 1
            mark_pending_outcome(v2, pending_by_acct[a], "lookup_failed")

    if not enriched:
        return 0, counts

    # Soft filter
    keepers: list[EnrichedCandidate] = []
    for ec in enriched:
        ok, reason = passes_soft_filters(ec.profile)
        if not ok:
            counts["soft_filter_rejected"] += 1
            mark_pending_outcome(
                v2, pending_by_acct[ec.acct.lower()], f"excluded:soft_{reason}"
            )
            continue
        keepers.append(ec)

    if not keepers:
        return 0, counts

    # Relationships check: ONLY meaningful for IDs that holm.community recognizes.
    # Since enrich_one() now does origin-instance lookup, the profile['id'] is the
    # ORIGIN instance's account id — not holm.community's. Holm's relationships
    # endpoint would 404 on these. We rely on the precomputed skip_following /
    # skip_followers / skip_attempted sets (built at startup from Tim's actual
    # followings list) to dedupe already-related accounts.
    final: list[EnrichedCandidate] = list(keepers)

    if not final:
        return 0, counts

    # Score
    accts_lower = [ec.acct.lower() for ec in final]
    v2_probs = load_recent_post_probs(v2, accts_lower)
    v1_scores = load_v1_recent_post_scores(v1, accts_lower)

    inserted = 0
    with v2.cursor() as cur:
        for ec in final:
            recent = v2_probs.get(ec.acct.lower()) or v1_scores.get(ec.acct.lower()) or []
            bio_text = (ec.profile.get("note") or "").strip()
            score, reasoning = compute_score(ec.profile, recent, user_centroid, bio_text)
            last_status_at = parse_last_status_at(ec.profile.get("last_status_at"))
            row = {
                "acct": ec.acct.lower(),
                "display_name": ec.profile.get("display_name"),
                "avatar_url": ec.profile.get("avatar_static") or ec.profile.get("avatar"),
                "bio": bio_text or None,
                "followers_count": ec.profile.get("followers_count"),
                "following_count": ec.profile.get("following_count"),
                "statuses_count": ec.profile.get("statuses_count"),
                "locked": ec.profile.get("locked"),
                "bot": ec.profile.get("bot"),
                "discoverable": ec.profile.get("discoverable"),
                "last_status_at": last_status_at,
                "score": score,
                "reasoning": json.dumps(reasoning),
                "instance": ec.instance,
            }
            try:
                if insert_candidate(cur, row):
                    inserted += 1
                    counts["inserted"] += 1
                    mark_pending_outcome(v2, pending_by_acct[ec.acct.lower()], "inserted")
            except Exception as e:
                log.debug("insert_candidate failed for %s: %s", ec.acct, e)
                mark_pending_outcome(v2, pending_by_acct[ec.acct.lower()], "insert_failed")
    v2.commit()
    return inserted, counts


def load_skip_sets(v1: psycopg.Connection, v2: psycopg.Connection) -> dict[str, Any]:
    return {
        "blocked_domains": load_blocked_domains(v2),
        "blocked_users": load_blocked_users_v1(v1),
        "attempted_follow": load_attempted_follow_accts(v1),
        "bad_403_instances": load_403_instances(v1),
    }


async def main_async() -> None:
    if not MASTODON_TOKEN:
        log.error("MASTODON_TOKEN not set; refusing to run without skip-set lookups")
        return

    log.info("connecting v1=%s v2=%s",
             V1_DSN.split("dbname=")[1].split()[0],
             V2_DSN.split("dbname=")[1].split()[0])
    v1 = psycopg.connect(V1_DSN, autocommit=False)
    v2 = psycopg.connect(V2_DSN, autocommit=False)

    log.info("loading scorer + centroid")
    scorer = None
    try:
        scorer = Scorer.load_or_initialize(SCORER_PATH)
    except Exception as e:
        log.warning("scorer load failed: %s", e)
    user_centroid = load_user_centroid(v2)
    log.info("scorer=%s centroid_dim=%s",
             scorer is not None, None if user_centroid is None else len(user_centroid))

    log.info("loading skip sets")
    skipsets = load_skip_sets(v1, v2)
    log.info("blocked_domains=%d blocked_users=%d attempted_follow=%d bad_403=%d",
             len(skipsets["blocked_domains"]),
             len(skipsets["blocked_users"]),
             len(skipsets["attempted_follow"]),
             len(skipsets["bad_403_instances"]))

    last_skipset_refresh = time.time()

    headers = {
        "Authorization": f"Bearer {MASTODON_TOKEN}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT) as client:
        skip_following, skip_followers, _, _ = await fetch_tim_skipset(client)
        log.info("tim_following=%d tim_followers=%d", len(skip_following), len(skip_followers))

        cycle = 0
        total_inserted = 0
        while _running:
            cycle += 1
            t0 = time.time()
            try:
                pending = fetch_pending_batch(v2, BATCH_SIZE)
                if not pending:
                    log.info("cycle %d: queue empty, sleeping %ds", cycle, POLL_INTERVAL_S)
                    await asyncio.sleep(POLL_INTERVAL_S)
                    continue

                inserted, counts = await enrich_batch(
                    v1, v2, pending,
                    skip_following, skip_followers,
                    skipsets["attempted_follow"], skipsets["blocked_users"],
                    skipsets["blocked_domains"], skipsets["bad_403_instances"],
                    scorer, user_centroid, client, sem,
                )
                total_inserted += inserted
                elapsed = time.time() - t0
                log.info(
                    "cycle %d: pending=%d inserted=%d total_inserted=%d "
                    "excl{following=%d follower=%d v1_attempted=%d soft=%d lookup=%d} "
                    "elapsed=%.1fs",
                    cycle, len(pending), inserted, total_inserted,
                    counts.get("already_following", 0),
                    counts.get("already_follower", 0),
                    counts.get("v1_attempted", 0),
                    counts.get("soft_filter_rejected", 0),
                    counts.get("lookup_failed", 0),
                    elapsed,
                )

                # Periodic skip-set refresh
                if time.time() - last_skipset_refresh > SKIPSET_REFRESH_S:
                    log.info("refreshing skip sets")
                    skipsets = load_skip_sets(v1, v2)
                    skip_following, skip_followers, _, _ = await fetch_tim_skipset(client)
                    last_skipset_refresh = time.time()

                if _running:
                    await asyncio.sleep(2)  # quick yield between cycles
            except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                # PG connection dropped (port-forward flap, server restart). Reconnect.
                log.warning("cycle %d DB connection lost: %s — reconnecting", cycle, e)
                try:
                    v1.close()
                except Exception:
                    pass
                try:
                    v2.close()
                except Exception:
                    pass
                await asyncio.sleep(5)
                while _running:
                    try:
                        v1 = psycopg.connect(V1_DSN, autocommit=False)
                        v2 = psycopg.connect(V2_DSN, autocommit=False)
                        log.info("reconnected to v1+v2")
                        break
                    except Exception as re:
                        log.warning("reconnect failed: %s — retrying in 10s", re)
                        await asyncio.sleep(10)
            except Exception as e:
                log.warning("cycle %d error: %s", cycle, e)
                await asyncio.sleep(POLL_INTERVAL_S)

    v1.close()
    v2.close()
    log.info("clean shutdown")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
