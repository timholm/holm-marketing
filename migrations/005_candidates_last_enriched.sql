-- 005: Add last_enriched_at to candidates for stale-refresh cycle
--
-- The enrich_pending_candidates daemon now tracks when each candidate
-- was last enriched (fetched from Mastodon API). This allows the daemon
-- to periodically re-enrich stale candidates that have aged > 60 days
-- without being checked, or newly marked as bot/suspended.

ALTER TABLE candidates ADD COLUMN IF NOT EXISTS last_enriched_at TIMESTAMPTZ DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_candidates_last_enriched
    ON candidates(last_enriched_at)
    WHERE reviewed = FALSE;

INSERT INTO migrations (version) VALUES (5)
    ON CONFLICT (version) DO NOTHING;
