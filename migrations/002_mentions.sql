-- 002: mentions and relationships
--
-- Tracks every interaction Tim is a party to:
--   - someone @-mentioned him
--   - someone replied to one of his posts
--   - someone DMed him
--   - someone followed him (informational only)
--
-- The mentions table is partitioned like posts. The "people" view aggregates
-- by author so the UI can show a relationship-per-person summary.

-- ----------------------------------------------------------------------------
-- mentions: every inbound interaction
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mentions (
    id              BIGSERIAL,
    notification_id TEXT,                         -- Mastodon notification id (string in API)
    kind            TEXT NOT NULL,                -- mention, reply, dm, follow, mention_followed_thread
    author_acct     TEXT NOT NULL,                -- the OTHER person
    author_display_name TEXT,
    author_avatar   TEXT,
    post_uri        TEXT,                         -- their post that mentions Tim
    post_url        TEXT,
    post_local_id   TEXT,                         -- Mastodon ID on holm.community
    post_content    TEXT,                         -- HTML-stripped
    post_html       TEXT,                         -- original HTML for display
    in_reply_to_uri TEXT,                         -- Tim's post they replied to (if any)
    parent_content  TEXT,                         -- snippet of Tim's post for context
    visibility      TEXT,                         -- public, unlisted, private, direct
    created_at      TIMESTAMPTZ NOT NULL,
    seen_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE IF NOT EXISTS mentions_2026_03 PARTITION OF mentions FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
CREATE TABLE IF NOT EXISTS mentions_2026_04 PARTITION OF mentions FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE IF NOT EXISTS mentions_2026_05 PARTITION OF mentions FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

CREATE INDEX IF NOT EXISTS idx_mentions_author ON mentions(author_acct, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mentions_kind_time ON mentions(kind, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_mentions_uniq ON mentions(notification_id, created_at) WHERE notification_id IS NOT NULL;

-- ----------------------------------------------------------------------------
-- mention_drafts: Ollama-generated draft replies for review
-- Tim never auto-posts. He clicks "copy & open" and posts manually.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mention_drafts (
    id              BIGSERIAL PRIMARY KEY,
    mention_id      BIGINT,
    mention_created_at TIMESTAMPTZ,
    draft_text      TEXT NOT NULL,
    rationale       TEXT,                         -- why the model picked this angle
    on_topic        BOOLEAN NOT NULL DEFAULT FALSE,
    topic_match     TEXT,                         -- which topic matched (or null)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    posted_at       TIMESTAMPTZ,                  -- when Tim confirmed
    rejected_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_drafts_pending ON mention_drafts(created_at DESC)
    WHERE posted_at IS NULL AND rejected_at IS NULL;

INSERT INTO migrations (version) VALUES (2) ON CONFLICT DO NOTHING;
