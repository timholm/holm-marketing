"""fedi-studio web app: read-only consumption surface for Tim's morning catch-up.

Routes:
    GET  /                    -> redirect to /today
    GET  /today               -> top scored posts from last 24h
    POST /feedback/bookmark   -> add to reading_queue (no Mastodon side-effect)
    POST /feedback/dismiss    -> mark dismissed + partial_fit(label=0)
    POST /feedback/read       -> mark read + partial_fit(label=1)
    GET  /draft               -> draft-assist composer (Phase 3.3)
    POST /draft               -> critique pasted text, return warnings
    GET  /weekly              -> Sunday and Friday ritual assistants (Phase 3.2)
    GET  /intro-wizard        -> form for pinned introduction post (Phase 3.1)
    POST /intro-wizard        -> generate intro draft + bio + featured tags
    GET  /healthz             -> {ok: true}

Auth: HTTP Basic. Set FEDI_STUDIO_USER and FEDI_STUDIO_PASSWORD env vars.

Design rules (non-negotiable):
- This server NEVER calls Mastodon's like/boost/follow endpoints.
- Bookmark/dismiss/read are LOCAL events. Tim manually engages on Mastodon.
- Every "engage" button in the UI opens Mastodon's compose/post URL in a new tab.
- /draft, /weekly, /intro-wizard ONLY produce drafts. Tim copies to Mastodon manually.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from fedi_studio.models.db import get_conn, init_pool
from fedi_studio.services.scorer import Scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fedi_studio.web")

# Templates
TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _strip_html(s: str | None) -> str:
    """Render filter: strip HTML tags, decode common entities."""
    if not s:
        return ""
    import re
    out = re.sub(r"<br\s*/?>", "\n", s)
    out = re.sub(r"</p>", "\n", out)
    out = re.sub(r"<[^>]+>", "", out)
    return (
        out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
        .strip()
    )


templates.env.filters["clean"] = _strip_html


def _holm_url(local_id: str | None, fallback_url: str | None) -> str:
    """Build a URL that opens the post on holm.community so Tim is logged in
    and can like/boost natively. Falls back to original URL if no local_id."""
    base = MASTODON_URL.rstrip("/")
    if local_id:
        return f"{base}/web/statuses/{local_id}"
    if fallback_url:
        # authorize_interaction lets Mastodon resolve a remote post on Tim's instance
        from urllib.parse import quote
        return f"{base}/authorize_interaction?uri={quote(fallback_url)}"
    return base


templates.env.filters["holm_url"] = _holm_url

# Auth
security = HTTPBasic()
AUTH_USER = os.environ.get("FEDI_STUDIO_USER", "tim")
AUTH_PASS = os.environ.get("FEDI_STUDIO_PASSWORD", "")
MASTODON_URL = os.environ.get("MASTODON_URL", "https://holm.community")

# Persisted scorer (in-memory for now; later: pickle to disk after each fit)
_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Constant-time auth check."""
    if not AUTH_PASS:
        # Dev mode: no password set, allow all
        return "dev"
    user_ok = secrets.compare_digest(credentials.username, AUTH_USER)
    pass_ok = secrets.compare_digest(credentials.password, AUTH_PASS)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ---------------------------------------------------------------------------
# Topic rules: HARD FILTER. Posts that don't match ANY topic are not shown.
# Better to show 5 relevant posts than 100 random ones.
# ---------------------------------------------------------------------------
TOPIC_RULES = [
    (
        "Off-grid & Homestead",
        [
            "off-grid", "offgrid", "off grid", "homestead", "homesteading",
            "cabin build", "tiny house", "tinyhouse", "rural living",
            "rainwater", "rain water", "cistern", "greywater", "grey water",
            "incinolet", "compost toilet", "composting toilet", "humanure",
            "earthship", "rammed earth", "compressed earth", "thinshell",
            "ferrocement", "natural building", "straw bale", "cob house",
            "self-suffic", "self suffic", "homestead",
        ],
    ),
    (
        "Solar & Energy",
        [
            "solar panel", "solar panels", "photovoltaic", "lifepo4",
            "battery bank", "off-grid power", "off grid power",
            "charge controller", "mppt", "victron", "ecoflow",
            "wind turbine", "micro hydro", "energy storage",
        ],
    ),
    (
        "Solarpunk",
        ["solarpunk", "solar punk", "solar-punk", "solarpunkstation"],
    ),
    (
        "Tech & Self-host",
        [
            "kubernetes", "k8s", "k3s", "kubectl", "docker compose",
            "selfhost", "self-host", "self host", "homelab", "home lab",
            "raspberry pi", "rpi", "linux server", "sysadmin", "devops",
            "argocd", "argo cd", "flux cd", "longhorn", "ceph",
            "nixos", "proxmox", "talos", "gitops",
            "mastodon admin", "matrix server",
        ],
    ),
    (
        "Garden & Growing",
        [
            "permaculture", "food forest", "raised bed", "raised beds",
            "hydroponic", "aquaponic", "no-till", "no till",
            "garden bed", "vegetable garden", "veggie garden",
            "greenhouse", "polytunnel", "cold frame",
            "saving seeds", "seed saving", "seed starting",
            "potato harvest", "tomato harvest",
        ],
    ),
    (
        "Vehicles & Conversion",
        [
            "schoolbus", "school bus", "skoolie", "bus conversion",
            "van life", "vanlife", "van conversion", "campervan", "camper van",
            "rv life", "rvlife", "tiny home on wheels",
        ],
    ),
    (
        "Maker & DIY",
        [
            "3d print", "3d-print", "3dprint",
            "woodwork", "wood working",
            "metalwork", "welding", "tig weld", "mig weld", "arc weld",
            "arduino", "esp32", "esp8266", "microcontroller",
            "cnc mill", "cnc router", "cnc machine",
        ],
    ),
    (
        "Mesh & Radio",
        [
            "lora", "meshtastic", "mesh network", "mesh networking",
            "ham radio", "amateur radio", "ham operator",
            "winlink", "aprs ", "p25 radio",
        ],
    ),
    (
        "Arizona & Desert",
        [
            "sonoran", "mojave", "arizona desert", "tucson",
            "saguaro", "ocotillo", "desert living", "desert garden",
        ],
    ),
]

# Flatten for SQL ILIKE filter
ALL_TOPIC_KEYWORDS = sorted({kw for _, kws in TOPIC_RULES for kw in kws})


def classify_topic(content: str, tags: list[str]) -> str | None:
    """Return the first topic whose keywords match in content or tags.
    Returns None if no topic matches — caller MUST drop these from display."""
    text = (content or "").lower() + " " + " ".join(tags or []).lower()
    for topic, kws in TOPIC_RULES:
        if any(kw in text for kw in kws):
            return topic
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="fedi-studio", description="Mastodon reading assistant. Not a bot.")


@app.on_event("startup")
async def startup() -> None:
    init_pool()
    log.info("DB pool initialized")


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/today", status_code=302)


@app.get("/today", response_class=HTMLResponse)
async def today(request: Request, _user: str = Depends(require_auth)) -> HTMLResponse:
    """Morning catch-up: only posts that match Tim's interests, grouped by topic.

    Filtering happens in two passes:
        1. SQL: ILIKE ANY across all topic keywords (cheap fanout filter)
        2. Python: classify_topic() picks the actual topic and DROPS no-match posts
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)

    # Build ILIKE-ANY clause for SQL prefilter. Matches in content OR tag arrays.
    ilike_terms = [f"%{kw}%" for kw in ALL_TOPIC_KEYWORDS]

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                p.id, p.uri, p.url, p.author_acct, p.content, p.tags,
                p.posted_at, p.media_count, p.favourites_count, p.reblogs_count,
                p.local_id, p.media_attachments, p.account_avatar, p.account_display_name,
                p.sensitive,
                ps.probability, ps.reasoning,
                (SELECT id FROM reading_queue rq WHERE rq.post_uri = p.uri AND rq.dismissed_at IS NULL) AS bookmarked,
                EXISTS (SELECT 1 FROM events e WHERE e.target_uri = p.uri AND e.event_type = 'dismiss') AS dismissed,
                EXISTS (SELECT 1 FROM events e WHERE e.target_uri = p.uri AND e.event_type = 'read') AS read
            FROM posts p
            JOIN post_scores ps ON ps.post_id = p.id AND ps.posted_at = p.posted_at
            WHERE p.posted_at >= %s
              AND (
                lower(p.content) LIKE ANY(%s)
                OR EXISTS (
                    SELECT 1 FROM unnest(p.tags) t WHERE lower(t) LIKE ANY(%s)
                )
              )
            ORDER BY ps.probability DESC, p.posted_at DESC
            LIMIT 200
            """,
            (cutoff, ilike_terms, ilike_terms),
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur]

    # Drop dismissed; classify; drop unmatched (defensive — SQL already filtered)
    visible = [r for r in rows if not r["dismissed"]]
    grouped: dict[str, list[dict]] = {}
    for r in visible:
        topic = classify_topic(r["content"], r["tags"])
        if topic is None:
            continue  # Should not happen given SQL filter, but be safe
        grouped.setdefault(topic, []).append(r)

    # Stable topic order: follow TOPIC_RULES sequence
    ordered_topics = [t for t, _ in TOPIC_RULES if t in grouped]
    grouped_ordered = [(t, grouped[t]) for t in ordered_topics]
    total_visible = sum(len(v) for v in grouped.values())

    return templates.TemplateResponse(
        request=request,
        name="today.html",
        context={
            "grouped": grouped_ordered,
            "total": total_visible,
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
        },
    )


@app.post("/feedback/bookmark")
async def bookmark(
    post_uri: str = Form(...),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Add to reading_queue. No Mastodon-side effect."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO reading_queue (post_uri, rank)
            VALUES (%s, 0)
            ON CONFLICT (post_uri) DO UPDATE SET dismissed_at = NULL
            """,
            (post_uri,),
        )
        cur.execute(
            "INSERT INTO events (event_type, target_type, target_uri) VALUES ('bookmark', 'post', %s)",
            (post_uri,),
        )
        conn.commit()
    return HTMLResponse(
        '<button class="btn btn-active" disabled>★ saved</button>'
    )


@app.post("/feedback/dismiss")
async def dismiss(
    post_uri: str = Form(...),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Negative feedback: don't show similar, train classifier with label=0."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO events (event_type, target_type, target_uri) VALUES ('dismiss', 'post', %s)",
            (post_uri,),
        )
        # Learn: pull the embedding and train
        cur.execute(
            "SELECT embedding, author_acct FROM posts WHERE uri = %s LIMIT 1",
            (post_uri,),
        )
        row = cur.fetchone()
        if row and row[0]:
            emb = np.array(row[0], dtype=np.float32)
            get_scorer().partial_fit(emb, label=0, author_acct=row[1])
        conn.commit()
    return HTMLResponse(
        '<div class="post-dismissed">dismissed (will see less of this)</div>',
        headers={"HX-Reswap": "outerHTML"},
    )


@app.post("/feedback/read")
async def mark_read(
    post_uri: str = Form(...),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Positive feedback: Tim engaged with this on Mastodon."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO events (event_type, target_type, target_uri) VALUES ('read', 'post', %s)",
            (post_uri,),
        )
        cur.execute(
            "SELECT embedding, author_acct FROM posts WHERE uri = %s LIMIT 1",
            (post_uri,),
        )
        row = cur.fetchone()
        if row and row[0]:
            emb = np.array(row[0], dtype=np.float32)
            get_scorer().partial_fit(emb, label=1, author_acct=row[1])
        conn.commit()
    return HTMLResponse(
        '<button class="btn btn-active" disabled>✓ read</button>'
    )


# ---------------------------------------------------------------------------
# /people: relationship view with Ollama drafts
# ---------------------------------------------------------------------------

# Markers from Tim's April 11, 2026 public apology post. Any mention whose
# parent_content contains one of these substrings is treated as a reply to
# that single one-time event and routed to a collapsed "Apology thread"
# section so it doesn't drown out real ongoing relationships.
APOLOGY_MARKERS = [
    "owe an apology",
    "apologies",
    "follow activity",
    "didn't understand the norms",
    "followed far too many",
    "i've stopped completely",
    "i was new to mastodon",
]


def _is_apology_thread(parent_content: str | None) -> bool:
    """True if the mention's parent_content quotes Tim's apology post."""
    if not parent_content:
        return False
    text = parent_content.lower()
    return any(m in text for m in APOLOGY_MARKERS)


@app.get("/people", response_class=HTMLResponse)
async def people(request: Request, _user: str = Depends(require_auth)) -> HTMLResponse:
    """Show every person who has interacted with Tim, grouped by interaction.

    Each row: their last mention, Tim's parent (if any), and the auto-draft
    (if on-topic). Tim approves or rejects per-row.

    The April 11 apology thread is filtered out of the main buckets and
    collected into its own collapsed `<details>` section. Data is preserved.
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                m.id, m.created_at, m.kind, m.author_acct, m.author_display_name,
                m.author_avatar, m.post_url, m.post_local_id, m.post_content,
                m.parent_content, m.visibility,
                d.id AS draft_id, d.draft_text, d.rationale, d.on_topic,
                d.topic_match, d.posted_at, d.rejected_at,
                (SELECT count(*) FROM mentions m2 WHERE m2.author_acct = m.author_acct) AS interaction_count
            FROM mentions m
            LEFT JOIN mention_drafts d
                ON d.mention_id = m.id AND d.mention_created_at = m.created_at
            WHERE m.kind IN ('mention', 'reply', 'dm')
            ORDER BY m.created_at DESC
            LIMIT 100
            """
        )
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur]

    # Bucket: on-topic drafts, off-topic drafts, no-draft (refused), already handled.
    # Replies to the April 11 apology post are split off into apology_thread.
    on_topic_ready: list[dict] = []
    off_topic_ready: list[dict] = []
    refused_hostile: list[dict] = []
    refused_other: list[dict] = []  # apology guard, ollama errors, etc.
    handled: list[dict] = []
    apology_thread: list[dict] = []

    for r in rows:
        if _is_apology_thread(r.get("parent_content")):
            r["apology_thread"] = True
            apology_thread.append(r)
            continue
        if r["posted_at"] or r["rejected_at"]:
            handled.append(r)
            continue
        rat = (r["rationale"] or "").lower()
        if r["draft_text"]:
            if r["on_topic"]:
                on_topic_ready.append(r)
            else:
                off_topic_ready.append(r)
        elif "hostile" in rat or "refused_topic" in rat:
            refused_hostile.append(r)
        else:
            refused_other.append(r)

    return templates.TemplateResponse(
        request=request,
        name="people.html",
        context={
            "on_topic_ready": on_topic_ready,
            "off_topic_ready": off_topic_ready,
            "refused_hostile": refused_hostile,
            "refused_other": refused_other,
            "handled": handled,
            "apology_thread": apology_thread,
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
        },
    )


@app.post("/people/approve")
async def approve_draft(
    draft_id: int = Form(...),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Mark draft posted. (Tim posts manually on Mastodon — we just track it.)"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE mention_drafts SET posted_at = NOW() WHERE id = %s",
            (draft_id,),
        )
        conn.commit()
    return HTMLResponse(
        '<div class="row-handled">marked posted</div>',
        headers={"HX-Reswap": "outerHTML"},
    )


@app.post("/people/reject")
async def reject_draft(
    draft_id: int = Form(...),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE mention_drafts SET rejected_at = NOW() WHERE id = %s",
            (draft_id,),
        )
        conn.commit()
    return HTMLResponse(
        '<div class="row-handled">rejected</div>',
        headers={"HX-Reswap": "outerHTML"},
    )


@app.post("/people/regenerate")
async def regenerate_draft(
    mention_id: int = Form(...),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Re-run the drafter on a mention. Replaces the existing draft row."""
    from fedi_studio.services.drafter import draft_reply

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT created_at, author_acct, post_content, parent_content
            FROM mentions WHERE id = %s
            """,
            (mention_id,),
        )
        row = cur.fetchone()
        if not row:
            return HTMLResponse("<div>not found</div>", status_code=404)
        created_at, author, content, parent = row

        d = draft_reply(content or "", parent or "", author or "")

        # Delete prior draft, insert new
        cur.execute(
            "DELETE FROM mention_drafts WHERE mention_id = %s AND mention_created_at = %s",
            (mention_id, created_at),
        )
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

    return HTMLResponse(
        f'<div class="draft-text">{d.text or "(no draft — " + (d.refused_reason or "refused") + ")"}</div>',
        headers={"HX-Reswap": "outerHTML"},
    )


# ---------------------------------------------------------------------------
# /relationships: per-person aggregate view across mentions/DMs/follows
# ---------------------------------------------------------------------------

# Mastodon DB connection (separate cluster from fedi_studio).
# Optional: if unreachable we degrade gracefully and mark mutual=None.
MASTODON_DSN = os.environ.get(
    "MASTODON_DSN",
    "host=localhost port=30141 dbname=mastodon user=mastodon password=mastodon",
)


def _fetch_mutuals_from_mastodon(accts: list[str]) -> dict[str, bool]:
    """Return {author_acct: is_mutual} for each acct in the input list.

    Mutual = Tim follows them AND they follow Tim back.
    Acct format from mentions: "user@host" (remote) or "user" (local).

    Best-effort: returns {} if the Mastodon DB is unreachable.
    """
    if not accts:
        return {}
    try:
        import psycopg

        with psycopg.connect(MASTODON_DSN, connect_timeout=2) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM accounts WHERE username = 'tim' AND domain IS NULL AND id > 0"
            )
            row = cur.fetchone()
            if not row:
                return {}
            tim_id = row[0]

            # Resolve each acct into an account id
            split = []
            for a in accts:
                if "@" in a:
                    user, _, host = a.partition("@")
                    split.append((a, user, host))
                else:
                    split.append((a, a, None))

            id_to_acct: dict[int, str] = {}
            for acct, user, host in split:
                if host is None:
                    cur.execute(
                        "SELECT id FROM accounts WHERE username = %s AND domain IS NULL",
                        (user,),
                    )
                else:
                    cur.execute(
                        "SELECT id FROM accounts WHERE username = %s AND domain = %s",
                        (user, host),
                    )
                r = cur.fetchone()
                if r:
                    id_to_acct[r[0]] = acct
            if not id_to_acct:
                return {}

            ids = list(id_to_acct.keys())
            # Tim follows them
            cur.execute(
                "SELECT target_account_id FROM follows WHERE account_id = %s AND target_account_id = ANY(%s)",
                (tim_id, ids),
            )
            tim_follows = {r[0] for r in cur}
            # They follow Tim
            cur.execute(
                "SELECT account_id FROM follows WHERE target_account_id = %s AND account_id = ANY(%s)",
                (tim_id, ids),
            )
            follows_tim = {r[0] for r in cur}

            mutuals_ids = tim_follows & follows_tim
            return {id_to_acct[i]: True for i in mutuals_ids} | {
                id_to_acct[i]: False for i in id_to_acct if i not in mutuals_ids
            }
    except Exception as e:  # pragma: no cover - exercised only when DB up
        log.warning("mastodon mutual lookup failed: %s", e)
        return {}


@app.get("/relationships", response_class=HTMLResponse)
async def relationships(
    request: Request, _user: str = Depends(require_auth)
) -> HTMLResponse:
    """Per-person aggregate view of every interaction Tim has had.

    Excludes Tim's own account and welcomebot. Ranks by:
      1. has_pending_draft DESC (queue work first)
      2. mutual DESC (real relationships next)
      3. last_interaction_at DESC (recency)
    """
    excluded = ["tim@holm.community", "welcomebot@holm.community"]

    with get_conn() as conn:
        cur = conn.cursor()
        # Aggregate per author
        cur.execute(
            """
            SELECT
                m.author_acct,
                MAX(m.author_display_name) AS display_name,
                MAX(m.author_avatar)       AS avatar,
                MIN(m.created_at)          AS first_at,
                MAX(m.created_at)          AS last_at,
                COUNT(*) FILTER (WHERE m.kind IN ('mention', 'reply')) AS total_mentions,
                COUNT(*) FILTER (WHERE m.kind = 'dm')                  AS total_dms,
                COUNT(*) FILTER (WHERE m.kind = 'follow')              AS total_follows,
                BOOL_OR(EXISTS (
                    SELECT 1 FROM mention_drafts d
                    WHERE d.mention_id = m.id
                      AND d.mention_created_at = m.created_at
                      AND d.posted_at IS NULL
                      AND d.rejected_at IS NULL
                )) AS has_pending_draft
            FROM mentions m
            WHERE m.author_acct <> ALL(%s)
            GROUP BY m.author_acct
            ORDER BY MAX(m.created_at) DESC
            """,
            (excluded,),
        )
        cols = [c.name for c in cur.description]
        people_rows = [dict(zip(cols, r)) for r in cur]

        # For each person grab last 3 mention/DM rows with drafts
        accts = [p["author_acct"] for p in people_rows]
        recent: dict[str, list[dict]] = {a: [] for a in accts}
        topics: dict[str, list[str]] = {a: [] for a in accts}
        if accts:
            cur.execute(
                """
                SELECT
                    m.id, m.created_at, m.kind, m.author_acct,
                    m.post_url, m.post_local_id, m.post_content, m.parent_content,
                    d.id AS draft_id, d.draft_text, d.on_topic, d.topic_match,
                    d.posted_at AS draft_posted_at, d.rejected_at AS draft_rejected_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.author_acct
                        ORDER BY m.created_at DESC
                    ) AS rn
                FROM mentions m
                LEFT JOIN mention_drafts d
                    ON d.mention_id = m.id AND d.mention_created_at = m.created_at
                WHERE m.author_acct = ANY(%s)
                  AND m.kind IN ('mention', 'reply', 'dm')
                """,
                (accts,),
            )
            mcols = [c.name for c in cur.description]
            for r in cur:
                row = dict(zip(mcols, r))
                if row["rn"] <= 3:
                    recent.setdefault(row["author_acct"], []).append(row)

            # Up to 5 most recent posts per author for topic extraction
            cur.execute(
                """
                SELECT author_acct, tags
                FROM (
                    SELECT
                        author_acct, tags,
                        ROW_NUMBER() OVER (
                            PARTITION BY author_acct ORDER BY posted_at DESC
                        ) AS rn
                    FROM posts
                    WHERE author_acct = ANY(%s)
                ) t
                WHERE rn <= 5 AND tags IS NOT NULL
                """,
                (accts,),
            )
            for acct, tags in cur:
                if not tags:
                    continue
                topics.setdefault(acct, []).extend(tags)

    # Compress topic lists: dedupe, keep top 5 by frequency
    for acct, tlist in topics.items():
        if not tlist:
            continue
        counts: dict[str, int] = {}
        for t in tlist:
            counts[t] = counts.get(t, 0) + 1
        topics[acct] = [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])][:5]

    # Mutuals (best-effort: Mastodon DB may be unreachable)
    mutuals_map = _fetch_mutuals_from_mastodon(accts)

    # Decorate and rank
    for p in people_rows:
        acct = p["author_acct"]
        p["recent"] = recent.get(acct, [])
        p["topics_we_share"] = topics.get(acct, [])
        p["mutual"] = mutuals_map.get(acct)  # True / False / None (unknown)

    def sort_key(p: dict) -> tuple:
        return (
            0 if p["has_pending_draft"] else 1,
            0 if p["mutual"] else 1,
            -p["last_at"].timestamp(),
        )

    people_rows.sort(key=sort_key)

    return templates.TemplateResponse(
        request=request,
        name="relationships.html",
        context={
            "people": people_rows,
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
            "mutuals_available": bool(mutuals_map),
        },
    )


# ---------------------------------------------------------------------------
# /stats: dev dashboard
# ---------------------------------------------------------------------------


@app.get("/stats", response_class=HTMLResponse)
async def stats(request: Request, _user: str = Depends(require_auth)) -> HTMLResponse:
    """Dev dashboard: post counts, score distribution, top authors/tags, partition sizes."""
    with get_conn() as conn:
        cur = conn.cursor()

        # Post totals + windowed counts in a single query
        cur.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE posted_at >= NOW() - INTERVAL '24 hours') AS last_24h,
                COUNT(*) FILTER (WHERE posted_at >= NOW() - INTERVAL '7 days')   AS last_7d,
                COUNT(*) FILTER (WHERE posted_at >= NOW() - INTERVAL '30 days')  AS last_30d
            FROM posts
            """
        )
        post_totals = dict(zip([c.name for c in cur.description], cur.fetchone()))

        # Daily ingest series for last 30 days (line chart)
        cur.execute(
            """
            SELECT date_trunc('day', posted_at) AS day, COUNT(*) AS n
            FROM posts
            WHERE posted_at >= NOW() - INTERVAL '30 days'
            GROUP BY 1
            ORDER BY 1
            """
        )
        daily_series = [(r[0], r[1]) for r in cur]

        # Score distribution histogram (10 buckets, [0.0, 1.0])
        cur.execute(
            """
            SELECT width_bucket(probability, 0.0, 1.0, 10) AS bucket, COUNT(*)
            FROM post_scores
            GROUP BY 1
            ORDER BY 1
            """
        )
        score_buckets_raw = {b: c for b, c in cur if b is not None}
        # Normalize: width_bucket returns 1..10 inclusive plus 11 for >=1.0
        score_histogram = []
        for b in range(1, 11):
            count = score_buckets_raw.get(b, 0)
            if b == 10:
                # roll the [1.0] overflow bucket into the last bin
                count += score_buckets_raw.get(11, 0)
            lo = (b - 1) / 10
            hi = b / 10
            score_histogram.append({"lo": lo, "hi": hi, "count": count})
        max_hist = max((h["count"] for h in score_histogram), default=1) or 1

        # Top 10 authors by post count in last 7 days
        cur.execute(
            """
            SELECT author_acct, COUNT(*) AS n
            FROM posts
            WHERE posted_at >= NOW() - INTERVAL '7 days'
            GROUP BY author_acct
            ORDER BY n DESC
            LIMIT 10
            """
        )
        top_authors = [(r[0], r[1]) for r in cur]

        # Top 10 tags (last 7 days)
        cur.execute(
            """
            SELECT t.tag, COUNT(*) AS n
            FROM (
                SELECT unnest(tags) AS tag
                FROM posts
                WHERE posted_at >= NOW() - INTERVAL '7 days'
            ) t
            GROUP BY t.tag
            ORDER BY n DESC
            LIMIT 10
            """
        )
        top_tags = [(r[0], r[1]) for r in cur]

        # Partition sizes for posts_YYYY_MM (just the table itself, not indexes)
        cur.execute(
            """
            SELECT relname, pg_relation_size(oid) AS bytes
            FROM pg_class
            WHERE relkind = 'r'
              AND relname ~ '^posts_[0-9]{4}_[0-9]{2}$'
            ORDER BY relname
            """
        )
        partitions = [(r[0], int(r[1])) for r in cur]

        # Mention/draft counters
        cur.execute("SELECT COUNT(*) FROM mentions")
        total_mentions = cur.fetchone()[0]
        cur.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE posted_at IS NOT NULL) AS posted,
                COUNT(*) FILTER (WHERE rejected_at IS NOT NULL) AS rejected,
                COUNT(*) FILTER (WHERE posted_at IS NULL AND rejected_at IS NULL) AS pending
            FROM mention_drafts
            """
        )
        draft_counts = dict(zip([c.name for c in cur.description], cur.fetchone()))

        # Training labels available
        cur.execute(
            """
            SELECT COUNT(*) FROM events
            WHERE event_type IN ('like_v1', 'read', 'bookmark')
            """
        )
        training_labels = cur.fetchone()[0]

    # Build a tiny inline SVG line chart for the 30-day series
    line_svg = _line_chart_svg(daily_series)

    return templates.TemplateResponse(
        request=request,
        name="stats.html",
        context={
            "post_totals": post_totals,
            "score_histogram": score_histogram,
            "max_hist": max_hist,
            "top_authors": top_authors,
            "top_tags": top_tags,
            "partitions": partitions,
            "total_mentions": total_mentions,
            "draft_counts": draft_counts,
            "training_labels": training_labels,
            "line_svg": line_svg,
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
        },
    )


def _line_chart_svg(series: list[tuple]) -> str:
    """Render a tiny inline SVG line chart from (date, count) pairs.

    Empty series returns a 'no data' placeholder. No external deps.
    """
    if not series:
        return '<div class="chart-empty">no data in window</div>'
    width, height, pad = 600, 120, 24
    counts = [c for _, c in series]
    max_y = max(counts) or 1
    n = len(series)
    if n == 1:
        # Single point: draw a dot
        cx, cy = width / 2, height - pad
        return (
            f'<svg viewBox="0 0 {width} {height}" class="line-chart" '
            f'preserveAspectRatio="none">'
            f'<circle cx="{cx}" cy="{cy}" r="3" />'
            f'<text x="{cx}" y="{cy - 6}" font-size="10" text-anchor="middle">{counts[0]}</text>'
            f"</svg>"
        )
    pts = []
    for i, (_, c) in enumerate(series):
        x = pad + (width - 2 * pad) * (i / (n - 1))
        y = height - pad - (height - 2 * pad) * (c / max_y)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (
        f'<svg viewBox="0 0 {width} {height}" class="line-chart">'
        f'<polyline fill="none" stroke="currentColor" stroke-width="2" points="{poly}" />'
        f'<text x="{pad}" y="{pad - 6}" font-size="10">peak: {max_y}</text>'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# /draft: draft-assist composer (Phase 3.3)
#
# Tim pastes text. We critique. We do NOT rewrite. We do NOT post.
# Output: warnings panel + clean copy of the original text + copy-to-clipboard.
# ---------------------------------------------------------------------------

# Food keyword list (short, curated): triggers a CW prompt
FOOD_KEYWORDS = ("bacon", "meat", "eat", "dinner", "recipe")

# Politics keywords: reuse REFUSE_TOPICS from drafter.py for consistency
# (imported lazily inside the route to avoid import-time coupling)


@app.get("/draft", response_class=HTMLResponse)
async def draft_form(request: Request, _user: str = Depends(require_auth)) -> HTMLResponse:
    """Blank composer with explainer."""
    return templates.TemplateResponse(
        request=request,
        name="draft.html",
        context={
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
            "text": "",
            "warnings": None,
            "checked": False,
        },
    )


@app.post("/draft", response_class=HTMLResponse)
async def draft_check(
    request: Request,
    text: str = Form(""),
    has_image: str = Form(""),
    alt_text: str = Form(""),
    reply_url: str = Form(""),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Critique only. Never rewrites. Returns warnings list + the same text."""
    from fedi_studio.services.drafter import REFUSE_TOPICS, _enforce_no_apology

    warnings: list[dict[str, str]] = []
    lower = (text or "").lower()

    # 1. Image attached without alt-text
    if has_image == "yes":
        if not alt_text.strip():
            warnings.append({
                "level": "high",
                "code": "alt_missing",
                "msg": "Image attached without alt-text. Mastodon culture expects alt for accessibility.",
            })
        elif len(alt_text.strip()) < 8:
            warnings.append({
                "level": "med",
                "code": "alt_short",
                "msg": "Alt-text is very short. Aim for a sentence describing what's visible and why it matters.",
            })

    # 2. Reply URL pasted: warn if older than 72h (best-effort: scan posts table)
    if reply_url.strip():
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT posted_at FROM posts WHERE url = %s OR uri = %s LIMIT 1",
                (reply_url.strip(), reply_url.strip()),
            )
            row = cur.fetchone()
            if row and row[0]:
                age = datetime.now(timezone.utc) - row[0]
                if age > timedelta(hours=72):
                    warnings.append({
                        "level": "med",
                        "code": "old_thread",
                        "msg": f"Replying to a thread {age.days}d {age.seconds//3600}h old. Replies on stale threads often go unread; consider quoting instead.",
                    })
            else:
                # URL pasted but unknown locally — flag stub
                warnings.append({
                    "level": "low",
                    "code": "old_thread_unknown",
                    "msg": "Reply URL pasted but the post isn't in local DB. Can't check age automatically; eyeball the timestamp before replying.",
                })

    # 3. Hashtag count
    hashtags = re.findall(r"#\w+", text or "", flags=re.UNICODE)
    if len(hashtags) > 5:
        warnings.append({
            "level": "med",
            "code": "too_many_tags",
            "msg": f"{len(hashtags)} hashtags in this post. More than 5 reads as marketing; pick the 3-5 that actually matter.",
        })

    # 4. Deadname: stub. Requires user-config list.
    # TODO: read deadname list from a config table or env var.
    # For now, only fire if a hardcoded sentinel `__DEADNAME__` appears so
    # the field stays present and operators see the placeholder existed.
    if "__deadname__" in lower:
        warnings.append({
            "level": "high",
            "code": "deadname_stub",
            "msg": "Deadname check is a stub: no list configured. Add user-config deadnames to enable.",
        })

    # 5. Politics or food without CW
    # Use word-boundary matching for short words to avoid false positives
    # like "eat" matching "great" or "meat" matching "treatment".
    has_cw = "cw:" in lower or "cw " in lower or "content warning" in lower

    def _word_hit(words: tuple[str, ...], hay: str) -> str | None:
        for kw in words:
            # Multi-word keywords stay as substring; single short words use \b.
            if " " in kw or len(kw) > 8:
                if kw in hay:
                    return kw
            else:
                if re.search(rf"\b{re.escape(kw)}\b", hay):
                    return kw
        return None

    politics_hit = _word_hit(REFUSE_TOPICS, lower)
    food_hit = _word_hit(FOOD_KEYWORDS, lower)
    if politics_hit and not has_cw:
        warnings.append({
            "level": "high",
            "code": "cw_politics",
            "msg": f"Mentions politics ('{politics_hit}') without a content warning. Many users hide politics; add 'CW: politics' before the body.",
        })
    if food_hit and not has_cw:
        warnings.append({
            "level": "med",
            "code": "cw_food",
            "msg": f"Mentions food ('{food_hit}') without a content warning. Common Mastodon courtesy: 'CW: food' for posts about meals or recipes.",
        })

    # 6. Apology language via existing guard
    if text and not _enforce_no_apology(text):
        warnings.append({
            "level": "high",
            "code": "apology",
            "msg": "Apology / fault-admission language detected. The standing rule is: do not apologize publicly. Reword as a confident statement of what you'll do next.",
        })

    return templates.TemplateResponse(
        request=request,
        name="draft.html",
        context={
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
            "text": text,
            "warnings": warnings,
            "checked": True,
            "has_image": has_image,
            "alt_text": alt_text,
            "reply_url": reply_url,
        },
    )


# ---------------------------------------------------------------------------
# /weekly: ritual assistants (Phase 3.2)
#
# Two side-by-side panels:
#   - #SolarPunkSunday: top 3 of Tim's photo posts last 7d, 3 caption suggestions each
#   - #FollowFriday:    top 3 accounts Tim has interacted with most last 7d
# ---------------------------------------------------------------------------

# Account format used for "Tim" varies between local and federated views.
# Match liberally so we handle 'tim', 'tim@holm.community', and full URI forms.
TIM_ACCT_PATTERNS = ("tim", "tim@holm.community")


def _ollama_generate(prompt: str, system: str | None = None, temp: float = 0.7, num: int = 200) -> str | None:
    """Tiny one-shot wrapper around Ollama. Returns None on any error/timeout.

    Reuses the same OLLAMA_URL/MODEL constants the drafter uses.
    """
    import httpx
    from fedi_studio.services.drafter import MODEL, OLLAMA_URL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "stream": False,
                "messages": messages,
                "options": {"temperature": temp, "num_predict": num},
            },
            timeout=45,
        )
        if r.status_code != 200:
            log.warning("ollama generate HTTP %d: %s", r.status_code, r.text[:200])
            return None
        return (r.json().get("message", {}) or {}).get("content", "") or None
    except Exception as e:
        log.warning("ollama generate exception: %s", e)
        return None


def _split_caption_options(raw: str) -> list[str]:
    """Best-effort: split a model response into 3 caption alternatives.

    Accepts numbered lists, bullets, blank-line separated, or one-per-line.
    Trims quotes, hashtags-inside-caption discouraged but kept if present.
    """
    if not raw:
        return []
    # Strip <think>…</think> first
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    cleaned: list[str] = []
    for ln in lines:
        # Strip leading list markers: "1.", "1)", "-", "*", "•"
        ln = re.sub(r"^\s*(?:\d+[\.)]\s+|[-*•]\s+)", "", ln)
        # Strip surrounding quotes
        if len(ln) > 2 and ln[0] in '"\'' and ln[-1] in '"\'':
            ln = ln[1:-1].strip()
        if ln and not ln.lower().startswith(("here are", "here's", "sure", "okay")):
            cleaned.append(ln)
    return cleaned[:3]


@app.get("/weekly", response_class=HTMLResponse)
async def weekly(request: Request, _user: str = Depends(require_auth)) -> HTMLResponse:
    """Weekly rituals: SolarPunkSunday photo captions + FollowFriday recs.

    Both sections degrade gracefully when there is no data or Ollama is down.
    """
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    # --- SolarPunkSunday: Tim's photo posts last 7d ---
    photo_posts: list[dict] = []
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, uri, url, content, posted_at,
                   media_count, media_attachments, favourites_count,
                   reblogs_count, local_id, account_avatar, account_display_name
            FROM posts
            WHERE (author_acct = ANY(%s) OR author_acct ILIKE 'tim@%%holm.community%%')
              AND posted_at >= %s
              AND media_count > 0
            ORDER BY (favourites_count + reblogs_count) DESC, posted_at DESC
            LIMIT 3
            """,
            (list(TIM_ACCT_PATTERNS), week_ago),
        )
        cols = [d.name for d in cur.description]
        photo_posts = [dict(zip(cols, r)) for r in cur]

    # Generate 3 caption suggestions per photo (one Ollama call per post)
    caption_system = (
        "You write casual one-line solarpunk captions for Mastodon photos. "
        "No hashtags inside the caption. No apology language. Lowercase fine. "
        "Output exactly 3 alternatives, each on its own numbered line. Nothing else."
    )
    for p in photo_posts:
        clean = _strip_html(p["content"])[:400]
        prompt = (
            f"Photo posted by Tim on {p['posted_at'].strftime('%a')}.\n"
            f"Tim's text alongside the photo:\n{clean or '(no caption yet)'}\n\n"
            "Output 3 alternative captions. Each one a single line, casual, "
            "solarpunk voice, no hashtags inside the caption text."
        )
        raw = _ollama_generate(prompt, system=caption_system, temp=0.8, num=180)
        p["caption_options"] = _split_caption_options(raw or "")
        if not p["caption_options"]:
            p["caption_options"] = []
            p["captions_error"] = "Ollama unavailable or returned no usable lines."

    # --- FollowFriday: top 3 accounts Tim interacted with last 7d ---
    excluded = ["tim@holm.community", "welcomebot@holm.community"]
    ff_candidates: list[dict] = []
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                m.author_acct,
                MAX(m.author_display_name) AS display_name,
                MAX(m.author_avatar)       AS avatar,
                COUNT(*) AS interaction_count,
                MAX(m.created_at) AS last_at
            FROM mentions m
            WHERE m.created_at >= %s
              AND m.author_acct <> ALL(%s)
              AND m.kind IN ('mention', 'reply')
            GROUP BY m.author_acct
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC, MAX(m.created_at) DESC
            LIMIT 3
            """,
            (week_ago, excluded),
        )
        cols = [d.name for d in cur.description]
        ff_candidates = [dict(zip(cols, r)) for r in cur]

        # Also fetch up to 5 recent tags per candidate for "topic" hint
        if ff_candidates:
            accts = [c["author_acct"] for c in ff_candidates]
            cur.execute(
                """
                SELECT author_acct, tags FROM (
                    SELECT author_acct, tags,
                        ROW_NUMBER() OVER (PARTITION BY author_acct ORDER BY posted_at DESC) AS rn
                    FROM posts WHERE author_acct = ANY(%s)
                ) t WHERE rn <= 5 AND tags IS NOT NULL
                """,
                (accts,),
            )
            tag_map: dict[str, list[str]] = {a: [] for a in accts}
            for acct, tags in cur:
                if tags:
                    tag_map[acct].extend(tags)
            for c in ff_candidates:
                tlist = tag_map.get(c["author_acct"], [])
                counts: dict[str, int] = {}
                for t in tlist:
                    counts[t] = counts.get(t, 0) + 1
                c["topics"] = [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])][:3]

    # Per-person one-sentence recommendations
    rec_system = (
        "You draft one-sentence #FollowFriday recommendations for Mastodon. "
        "Casual, lowercase fine, no apology language, no marketing tone. "
        "Output ONE sentence only. No quotes, no preamble, no @-mention."
    )
    for c in ff_candidates:
        topics = c.get("topics") or []
        topic_blurb = ", ".join(topics) if topics else "interesting things"
        prompt = (
            f"Draft a one-sentence #FollowFriday recommendation for "
            f"@{c['author_acct']} who posts about {topic_blurb}. "
            "Keep it warm but specific. Tim is the author of the recommendation."
        )
        raw = _ollama_generate(prompt, system=rec_system, temp=0.7, num=120)
        if raw:
            line = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
            # Strip leading list markers / quotes
            line = re.sub(r"^\s*(?:\d+[\.)]\s+|[-*•]\s+)", "", line).strip()
            if len(line) > 2 and line[0] in '"\'' and line[-1] in '"\'':
                line = line[1:-1].strip()
            # Take only the first non-empty line
            line = next((ln for ln in line.splitlines() if ln.strip()), "")
            c["recommendation"] = line[:200]
        else:
            c["recommendation"] = ""

    # Bundle the 3 recs into one ready-to-paste toot
    if ff_candidates and any(c.get("recommendation") for c in ff_candidates):
        bundle_lines = ["#FollowFriday — folks I've been talking with this week:"]
        for c in ff_candidates:
            if c.get("recommendation"):
                bundle_lines.append(f"@{c['author_acct']} — {c['recommendation']}")
        ff_bundle = "\n\n".join(bundle_lines)
    else:
        ff_bundle = ""

    return templates.TemplateResponse(
        request=request,
        name="weekly.html",
        context={
            "now": now,
            "mastodon_url": MASTODON_URL,
            "photo_posts": photo_posts,
            "ff_candidates": ff_candidates,
            "ff_bundle": ff_bundle,
        },
    )


# ---------------------------------------------------------------------------
# /intro-wizard: pinned introduction post helper (Phase 3.1)
# ---------------------------------------------------------------------------

# Mapping from TOPIC_RULES topic name to a short label and the seed hashtag
# used in the intro draft. Order matches /today's topic order.
INTRO_INTERESTS = [
    ("offgrid",    "Off-grid & Homestead", "#OffGrid #Homesteading"),
    ("solar",      "Solar & Energy",       "#Solar #Solarpunk"),
    ("solarpunk",  "Solarpunk",            "#Solarpunk"),
    ("tech",       "Tech & Self-host",     "#SelfHosted #Kubernetes"),
    ("garden",     "Garden & Growing",     "#Permaculture #Garden"),
    ("vehicles",   "Vehicles & Conversion","#Skoolie #VanLife"),
    ("maker",      "Maker & DIY",          "#Maker #DIY"),
    ("mesh",       "Mesh & Radio",         "#Meshtastic #LoRa"),
    ("arizona",    "Arizona & Desert",     "#Arizona #Sonoran"),
]


@app.get("/intro-wizard", response_class=HTMLResponse)
async def intro_wizard_form(request: Request, _user: str = Depends(require_auth)) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="intro_wizard.html",
        context={
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
            "interests": INTRO_INTERESTS,
            "submitted": False,
        },
    )


@app.post("/intro-wizard", response_class=HTMLResponse)
async def intro_wizard_submit(
    request: Request,
    interests: list[str] = Form(default=[]),
    extra_notes: str = Form(""),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Generate intro draft + bio + featured tags from selected interests."""
    from fedi_studio.services.drafter import _enforce_no_apology

    selected_labels: list[str] = []
    seed_tags: list[str] = []
    for code, label, tags in INTRO_INTERESTS:
        if code in interests:
            selected_labels.append(label)
            seed_tags.extend(tags.split())
    # Dedup tags preserving order
    seen = set()
    seed_tags = [t for t in seed_tags if not (t in seen or seen.add(t))]

    # ---- (a) Intro post draft ----
    interests_blurb = ", ".join(selected_labels) if selected_labels else "off-grid life and tinkering"
    extra = extra_notes.strip()[:240]
    intro_system = (
        "You draft a single Mastodon #introduction post on behalf of Tim. "
        "Strict rules: under 500 characters, no apology language, no marketing tone, "
        "lowercase fine, no @-mentions, end with 3-5 CamelCased hashtags on a final line. "
        "Mastodon-native voice: warm, curious, specific. No emoji unless distinctive. "
        "Output ONLY the post text, nothing else."
    )
    intro_user = (
        f"Tim is in Arizona, runs a small Kubernetes cluster on Raspberry Pis, "
        f"and is building toward an off-grid cabin. His stated interests: {interests_blurb}. "
        f"{('Extra notes from Tim: ' + extra) if extra else ''}\n\n"
        "Draft Tim's pinned #introduction post. Under 500 characters. "
        "End with 3-5 CamelCased hashtags on the final line."
    )
    raw_intro = _ollama_generate(intro_user, system=intro_system, temp=0.6, num=380)
    if raw_intro:
        intro_text = re.sub(r"<think>.*?</think>", "", raw_intro, flags=re.DOTALL | re.IGNORECASE).strip()
        # Strip surrounding quotes
        if len(intro_text) > 2 and intro_text[0] in '"\'' and intro_text[-1] in '"\'':
            intro_text = intro_text[1:-1].strip()
        # Cap to 500 chars
        if len(intro_text) > 500:
            intro_text = intro_text[:497] + "..."
        # Apology guard: invalidate if violates
        if not _enforce_no_apology(intro_text):
            intro_text = ""
            intro_error = "Generated draft contained apology language; rejected."
        else:
            intro_error = ""
    else:
        intro_text = ""
        intro_error = "Ollama unavailable; try again or hand-write the draft."

    # ---- (b) Featured hashtag suggestions ----
    # Combine selected seed tags, fall back to defaults if user picked nothing.
    if seed_tags:
        featured_tags = seed_tags[:5]
    else:
        featured_tags = ["#OffGrid", "#Homesteading", "#Solarpunk", "#SelfHosted", "#Arizona"]
    featured_tags = featured_tags[:5]

    # ---- (c) Bio rewrite ----
    bio_system = (
        "You rewrite Mastodon profile bios. Strict rules: under 220 characters, "
        "no apology language, no marketing tone, no emoji unless distinctive, "
        "MUST include the phrase 'automated tools: fedi-studio (reading assistant only)'. "
        "Output ONLY the bio text, nothing else."
    )
    bio_user = (
        f"Tim's interests: {interests_blurb}. Arizona, off-grid in progress, "
        f"runs a small kubernetes cluster on raspberry pis. "
        "Write a Mastodon bio under 220 chars that mentions those interests and "
        "includes verbatim: 'automated tools: fedi-studio (reading assistant only)'."
    )
    raw_bio = _ollama_generate(bio_user, system=bio_system, temp=0.5, num=180)
    if raw_bio:
        bio_text = re.sub(r"<think>.*?</think>", "", raw_bio, flags=re.DOTALL | re.IGNORECASE).strip()
        if len(bio_text) > 2 and bio_text[0] in '"\'' and bio_text[-1] in '"\'':
            bio_text = bio_text[1:-1].strip()
        if len(bio_text) > 220:
            bio_text = bio_text[:217] + "..."
        if not _enforce_no_apology(bio_text):
            bio_text = ""
            bio_error = "Generated bio contained apology language; rejected."
        else:
            bio_error = ""
    else:
        bio_text = ""
        bio_error = "Ollama unavailable."

    return templates.TemplateResponse(
        request=request,
        name="intro_wizard.html",
        context={
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
            "interests": INTRO_INTERESTS,
            "submitted": True,
            "selected": interests,
            "selected_labels": selected_labels,
            "extra_notes": extra,
            "intro_text": intro_text,
            "intro_error": intro_error,
            "featured_tags": featured_tags,
            "bio_text": bio_text,
            "bio_error": bio_error,
        },
    )


# ---------------------------------------------------------------------------
# /candidates: read-only follow-suggestion list
#
# Tim manually reviews each row and clicks "open in Mastodon" to follow them
# in the native UI. THIS ROUTE NEVER CALLS THE FOLLOW API. The decision
# endpoints (/candidates/{id}/decision) only update the local row.
# ---------------------------------------------------------------------------


@app.get("/candidates", response_class=HTMLResponse)
async def candidates_page(
    request: Request,
    page: int = 1,
    instance: str | None = None,
    min_score: float = 0.0,
    q: str | None = None,
    show: str = "pending",  # 'pending' | 'reviewed' | 'all'
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Paginated candidates view, ordered by score DESC where reviewed=false."""
    page = max(page, 1)
    per_page = 50
    offset = (page - 1) * per_page

    where: list[str] = []
    params: list = []

    if show == "pending":
        where.append("reviewed = FALSE")
    elif show == "reviewed":
        where.append("reviewed = TRUE")
    # else 'all': no filter

    if instance:
        where.append("instance = %s")
        params.append(instance.strip().lower())
    if min_score > 0:
        where.append("score >= %s")
        params.append(min_score)
    if q:
        where.append("acct ILIKE %s")
        params.append(f"%{q.strip().lower()}%")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with get_conn() as conn:
        cur = conn.cursor()
        # Counts (always over the entire table for the header)
        cur.execute(
            """
            SELECT
                count(*) FILTER (WHERE TRUE)                   AS total,
                count(*) FILTER (WHERE reviewed = TRUE)        AS reviewed,
                count(*) FILTER (WHERE decision = 'followed')  AS followed,
                count(*) FILTER (WHERE decision = 'skipped')   AS skipped,
                count(*) FILTER (WHERE reviewed = FALSE)       AS remaining
            FROM candidates
            """
        )
        counts_row = cur.fetchone()
        counts = {
            "total": counts_row[0],
            "reviewed": counts_row[1],
            "followed": counts_row[2],
            "skipped": counts_row[3],
            "remaining": counts_row[4],
        }

        # Available instances for the filter dropdown (top 30 by row count)
        cur.execute(
            """
            SELECT instance, count(*) AS n
            FROM candidates
            WHERE instance IS NOT NULL
            GROUP BY instance
            ORDER BY n DESC
            LIMIT 30
            """
        )
        instance_options = [(r[0], r[1]) for r in cur]

        cur.execute(
            f"""
            SELECT id, acct, display_name, avatar_url, bio, followers_count,
                   following_count, statuses_count, locked, bot, discoverable,
                   last_status_at, score, reasoning, instance, reviewed,
                   reviewed_at, decision, created_at
            FROM candidates
            {where_sql}
            ORDER BY score DESC NULLS LAST, id ASC
            LIMIT %s OFFSET %s
            """,
            (*params, per_page, offset),
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur]

        # Filtered total for pagination
        cur.execute(
            f"SELECT count(*) FROM candidates {where_sql}",
            params,
        )
        filtered_total = cur.fetchone()[0]

    total_pages = max((filtered_total + per_page - 1) // per_page, 1)

    return templates.TemplateResponse(
        request=request,
        name="candidates.html",
        context={
            "rows": rows,
            "counts": counts,
            "filtered_total": filtered_total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "instance": instance or "",
            "min_score": min_score,
            "q": q or "",
            "show": show,
            "instance_options": instance_options,
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
        },
    )


@app.post("/candidates/{candidate_id}/decision", response_class=HTMLResponse)
async def candidate_decision(
    candidate_id: int,
    decision: str = Form(...),
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Record Tim's decision. THIS DOES NOT CALL THE MASTODON FOLLOW API.

    decision must be one of: 'followed', 'skipped'. The card is replaced
    in-place with a confirmation chip; Tim's actual follow happens in the
    Mastodon UI on holm.community.
    """
    if decision not in ("followed", "skipped"):
        raise HTTPException(status_code=400, detail="bad decision")

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE candidates
            SET reviewed = TRUE,
                reviewed_at = NOW(),
                decision = %s
            WHERE id = %s
            RETURNING acct
            """,
            (decision, candidate_id),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        # Audit log only — local event, NOT a Mastodon call
        cur.execute(
            "INSERT INTO events (event_type, target_type, target_id, payload) "
            "VALUES (%s, %s, %s, %s::jsonb)",
            (
                f"candidate_{decision}",
                "candidate",
                candidate_id,
                f'{{"acct": "{row[0]}"}}',
            ),
        )
        conn.commit()

    label = "✓ followed" if decision == "followed" else "✗ skipped"
    cls = "btn btn-active" if decision == "followed" else "btn btn-dismiss"
    return HTMLResponse(
        f'<div class="candidate-decided">{label}</div>'
        f'<style>#candidate-{candidate_id}{{opacity:.45}}</style>'
    )


# ---------------------------------------------------------------------------
# /review: single-candidate detail review page (one at a time)
# ---------------------------------------------------------------------------


@app.get("/review", response_class=HTMLResponse)
async def review_page(
    request: Request,
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """Fetch the highest-scoring unreviewed candidate and render a full review card."""
    with get_conn() as conn:
        cur = conn.cursor()

        # Get queue status for header
        cur.execute(
            """
            SELECT
                count(*) FILTER (WHERE reviewed = FALSE) AS pending,
                count(*) FILTER (WHERE reviewed = TRUE)  AS reviewed,
                count(*) FILTER (WHERE reviewed = FALSE AND created_at::date = CURRENT_DATE) AS today_reviewed
            FROM candidates
            """
        )
        queue_row = cur.fetchone()
        queue_pending = queue_row[0] if queue_row else 0
        queue_reviewed = queue_row[1] if queue_row else 0
        queue_today = queue_row[2] if queue_row else 0

        # Pick the next unreviewed candidate (highest score)
        cur.execute(
            """
            SELECT id, acct, display_name, avatar_url, bio, followers_count,
                   following_count, statuses_count, locked, bot, discoverable,
                   last_status_at, score, reasoning, instance, created_at
            FROM candidates
            WHERE reviewed = FALSE
            ORDER BY score DESC NULLS LAST, id ASC
            LIMIT 1
            """
        )
        candidate_row = cur.fetchone()

        if candidate_row is None:
            # No unreviewed candidates
            return templates.TemplateResponse(
                request=request,
                name="review_empty.html",
                context={
                    "now": datetime.now(timezone.utc),
                    "mastodon_url": MASTODON_URL,
                },
            )

        cols = [d.name for d in cur.description]
        candidate = dict(zip(cols, candidate_row))

        # Pull recent posts for this candidate (if available in v2 table)
        cur.execute(
            """
            SELECT id, content, posted_at
            FROM posts
            WHERE author_acct = %s
            ORDER BY posted_at DESC
            LIMIT 5
            """,
            (candidate["acct"],),
        )
        recent_posts = [
            dict(zip([d.name for d in cur.description], r))
            for r in cur
        ]

        # Total candidate count
        cur.execute("SELECT count(*) FROM candidates")
        total_candidates = cur.fetchone()[0]

        return templates.TemplateResponse(
            request=request,
            name="review.html",
            context={
                "candidate": candidate,
                "recent_posts": recent_posts,
                "queue_pending": queue_pending,
                "queue_reviewed": queue_reviewed,
                "queue_total": total_candidates,
                "queue_today": queue_today,
                "now": datetime.now(timezone.utc),
                "mastodon_url": MASTODON_URL,
            },
        )


@app.get("/review/empty", response_class=HTMLResponse)
async def review_empty_page(
    request: Request,
    _user: str = Depends(require_auth),
) -> HTMLResponse:
    """HTMX swap target: when queue is exhausted, show completion message."""
    return templates.TemplateResponse(
        request=request,
        name="review_empty.html",
        context={
            "now": datetime.now(timezone.utc),
            "mastodon_url": MASTODON_URL,
        },
    )


# Static files (CSS) — optional, fall back gracefully
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
