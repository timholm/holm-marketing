-- 003: candidate accounts
--
-- READ-ONLY follow-suggestion list. Tim manually reviews each row and clicks
-- through to Mastodon to follow. THIS APP NEVER CALLS THE FOLLOW API.
--
-- The build_candidates worker fans across multiple discovery sources
-- (v2 high-prob authors, v1 high-score authors, v1 active authors) and
-- writes the merged set here, dedup'd on `acct`. The /candidates web route
-- presents them paginated, ordered by score DESC where reviewed=false.

CREATE TABLE IF NOT EXISTS candidates (
    id                  BIGSERIAL PRIMARY KEY,
    acct                TEXT UNIQUE NOT NULL,
    display_name        TEXT,
    avatar_url          TEXT,
    bio                 TEXT,
    followers_count     INT,
    following_count     INT,
    statuses_count      INT,
    locked              BOOLEAN,
    bot                 BOOLEAN,
    discoverable        BOOLEAN,
    last_status_at      TIMESTAMPTZ,
    score               REAL,
    reasoning           JSONB DEFAULT '{}'::jsonb,
    instance            TEXT,
    reviewed            BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_at         TIMESTAMPTZ,
    decision            TEXT,                       -- 'followed' | 'skipped' | NULL
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates(score DESC) WHERE reviewed = FALSE;
CREATE INDEX IF NOT EXISTS idx_candidates_instance ON candidates(instance);
CREATE INDEX IF NOT EXISTS idx_candidates_decision ON candidates(decision);

INSERT INTO migrations (version) VALUES (3) ON CONFLICT DO NOTHING;
