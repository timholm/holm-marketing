-- candidates_pending: lightweight queue populated by firehose for fresh high-scoring authors.
-- Drained by enrich_pending_candidates worker which runs the full exclusion + lookup + scoring path
-- and INSERT-ON-CONFLICT-DO-NOTHING into `candidates`.

CREATE TABLE IF NOT EXISTS candidates_pending (
    id            BIGSERIAL PRIMARY KEY,
    acct          TEXT NOT NULL,
    source_post_uri TEXT,
    source_score  REAL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    enriched_at   TIMESTAMPTZ,
    enriched_outcome TEXT,   -- 'inserted' | 'excluded:<reason>' | 'lookup_failed' | NULL
    UNIQUE (acct)
);

CREATE INDEX IF NOT EXISTS idx_candidates_pending_unenriched
    ON candidates_pending(first_seen_at)
    WHERE enriched_at IS NULL;

INSERT INTO migrations (version) VALUES (4)
    ON CONFLICT (version) DO NOTHING;
