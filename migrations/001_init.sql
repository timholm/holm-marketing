-- fedi-studio schema, initial migration
-- Postgres 16+ with pgvector extension
--
-- Design notes:
-- - posts is partitioned by month on posted_at; retention drops old partitions
-- - BRIN index on posted_at: matches append-ordered insertion, tiny size
-- - events is an immutable log; replaces denormalized signal columns
-- - No actions/queue table by design: the system never queues outbound engagement

-- pgvector not installed in current PG image; using real[] until we swap.
-- To migrate later:
--   CREATE EXTENSION vector;
--   ALTER TABLE posts ALTER COLUMN embedding TYPE vector(256) USING embedding::real[]::vector(256);
--   CREATE INDEX idx_posts_embedding ON posts USING hnsw (embedding vector_cosine_ops);

-- ----------------------------------------------------------------------------
-- profiles: minimal account metadata, scoring lives in events
-- ----------------------------------------------------------------------------
CREATE TABLE profiles (
    id              BIGSERIAL PRIMARY KEY,
    acct            TEXT UNIQUE NOT NULL,
    display_name    TEXT,
    bio             TEXT,
    instance        TEXT NOT NULL,
    raw_data        JSONB DEFAULT '{}'::jsonb,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_profiles_instance ON profiles(instance);
CREATE INDEX idx_profiles_last_seen ON profiles(last_seen_at DESC);

-- ----------------------------------------------------------------------------
-- posts: partitioned by month on posted_at
-- ----------------------------------------------------------------------------
CREATE TABLE posts (
    id                  BIGSERIAL,
    uri                 TEXT NOT NULL,            -- canonical AP URL
    url                 TEXT,                     -- web URL (may differ)
    author_acct         TEXT NOT NULL,
    content             TEXT NOT NULL,
    content_hash        BYTEA NOT NULL,           -- md5 for dedup
    language            TEXT,                     -- ISO 639-1 from Mastodon API
    tags                TEXT[] DEFAULT '{}',
    media_count         INT DEFAULT 0,
    favourites_count    INT DEFAULT 0,
    reblogs_count       INT DEFAULT 0,
    in_reply_to_id      TEXT,                     -- parent URI for threads
    sensitive           BOOLEAN DEFAULT FALSE,
    posted_at           TIMESTAMPTZ NOT NULL,
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding           REAL[],                   -- Model2Vec potion-base-32M (256-dim)
    PRIMARY KEY (id, posted_at)
) PARTITION BY RANGE (posted_at);

-- BRIN index: tiny (<10MB even with 100M rows), perfect for append-ordered data
CREATE INDEX idx_posts_posted_at_brin ON posts USING BRIN (posted_at)
    WITH (pages_per_range = 32);

-- B-tree for author lookups
CREATE INDEX idx_posts_author ON posts(author_acct, posted_at DESC);

-- Hash-style index for dedup (using content_hash as bytea)
CREATE INDEX idx_posts_hash ON posts(content_hash);

-- Tags lookup (GIN)
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);

-- HNSW index on embeddings for nearest-neighbor (added when pgvector populates)
-- Created after data is loaded for better build performance:
-- CREATE INDEX idx_posts_embedding ON posts USING hnsw (embedding vector_cosine_ops);

-- Initial partitions for current and next month
-- Production rotation handled by a monthly job
CREATE TABLE posts_2026_04 PARTITION OF posts
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE posts_2026_05 PARTITION OF posts
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

-- ----------------------------------------------------------------------------
-- post_scores: scoring decoupled from post storage
-- ----------------------------------------------------------------------------
CREATE TABLE post_scores (
    post_id         BIGINT NOT NULL,
    posted_at       TIMESTAMPTZ NOT NULL,         -- denormalized for partition routing
    probability     REAL NOT NULL,                -- 0.0-1.0 calibrated score
    reasoning       JSONB DEFAULT '{}'::jsonb,    -- {alpha: 0.6, beta: 0.3, ...}
    scorer_version  TEXT NOT NULL,                -- for A/B and rollback
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (post_id, posted_at)
);

CREATE INDEX idx_post_scores_prob ON post_scores(probability DESC, posted_at DESC);

-- ----------------------------------------------------------------------------
-- author_priors: learned author affinity from Tim's history
-- ----------------------------------------------------------------------------
CREATE TABLE author_priors (
    acct            TEXT PRIMARY KEY,
    likes           INT NOT NULL DEFAULT 0,
    impressions     INT NOT NULL DEFAULT 0,
    prior           REAL NOT NULL DEFAULT 0.5,    -- Bayesian smoothed (likes+1)/(impressions+2)
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- user_centroid: Tim's preference embedding (single row table)
-- ----------------------------------------------------------------------------
CREATE TABLE user_centroid (
    id              INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    embedding       REAL[],
    based_on_likes  INT NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO user_centroid (id) VALUES (1);

-- ----------------------------------------------------------------------------
-- events: immutable log, replaces denormalized signal tables
-- Every meaningful thing Tim does or sees is an event.
-- ----------------------------------------------------------------------------
CREATE TABLE events (
    id              BIGSERIAL,
    event_type      TEXT NOT NULL,                -- read, dismiss, bookmark, like, boost, draft, post, ...
    target_type     TEXT NOT NULL,                -- post, profile, draft, ...
    target_id       BIGINT,
    target_uri      TEXT,                         -- denormalized for joining without target_id
    payload         JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE events_2026_04 PARTITION OF events
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');
CREATE TABLE events_2026_05 PARTITION OF events
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

CREATE INDEX idx_events_type_time ON events(event_type, created_at DESC);
CREATE INDEX idx_events_target ON events(target_type, target_id, created_at DESC);

-- ----------------------------------------------------------------------------
-- reading_queue: what Tim has bookmarked or wants to come back to
-- ----------------------------------------------------------------------------
CREATE TABLE reading_queue (
    id              BIGSERIAL PRIMARY KEY,
    post_uri        TEXT NOT NULL UNIQUE,         -- not post_id because partitioned
    rank            INT NOT NULL DEFAULT 0,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dismissed_at    TIMESTAMPTZ,
    read_at         TIMESTAMPTZ
);

CREATE INDEX idx_reading_queue_active ON reading_queue(rank DESC, added_at DESC)
    WHERE dismissed_at IS NULL AND read_at IS NULL;

-- ----------------------------------------------------------------------------
-- drafts: replies and posts Tim is working on
-- The tool drafts; Tim posts. posted_at is set if/when Tim copies to Mastodon.
-- ----------------------------------------------------------------------------
CREATE TABLE drafts (
    id              BIGSERIAL PRIMARY KEY,
    target_uri      TEXT,                         -- post being replied to (NULL for original posts)
    draft_text      TEXT NOT NULL,
    kind            TEXT NOT NULL,                -- reply, original, intro, follow_friday, ...
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    posted_at       TIMESTAMPTZ                   -- when Tim confirmed he posted it
);

CREATE INDEX idx_drafts_unposted ON drafts(created_at DESC) WHERE posted_at IS NULL;

-- ----------------------------------------------------------------------------
-- blocklist: domains and accts to never ingest
-- ----------------------------------------------------------------------------
CREATE TABLE blocklist (
    id              BIGSERIAL PRIMARY KEY,
    pattern         TEXT NOT NULL UNIQUE,         -- domain or @acct@host
    reason          TEXT,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed with known-bad instances
INSERT INTO blocklist (pattern, reason) VALUES
    ('poa.st', 'White supremacist instance'),
    ('kiwifarms.cc', 'Harassment platform'),
    ('poast.org', 'Hate speech instance'),
    ('freespeechextremist.com', 'Hate speech'),
    ('shitposter.club', 'Harassment'),
    ('shitposter.world', 'Harassment'),
    ('nicecrew.digital', 'Harassment'),
    ('detroitriotcity.com', 'Harassment'),
    ('aethy.com', 'Illegal content'),
    ('baraag.net', 'Illegal content');

-- ----------------------------------------------------------------------------
-- migrations: track applied migrations
-- ----------------------------------------------------------------------------
CREATE TABLE migrations (
    version         INT PRIMARY KEY,
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO migrations (version) VALUES (1);
