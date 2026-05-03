-- 007: follow-graph crawling support
--
-- Add columns to track Mastodon account IDs and graph-crawl state.
-- This enables follow-graph crawler to seed from high-scoring candidates
-- and populate candidates_pending with their following/followers.

-- candidates table: store holm_account_id and graph_crawl timestamp
ALTER TABLE candidates
ADD COLUMN IF NOT EXISTS holm_account_id TEXT,
ADD COLUMN IF NOT EXISTS graph_crawled_at TIMESTAMPTZ;

-- Index for efficient cursor (next uncrawled seed, ordered by score DESC)
CREATE INDEX IF NOT EXISTS idx_candidates_graph_crawl
ON candidates (score DESC)
WHERE reviewed = FALSE AND graph_crawled_at IS NULL;

-- candidates_pending table: store holm_account_id to allow
-- enrich_pending_candidates to skip the /accounts/lookup step
ALTER TABLE candidates_pending
ADD COLUMN IF NOT EXISTS holm_account_id TEXT;

INSERT INTO migrations (version) VALUES (7)
ON CONFLICT (version) DO NOTHING;
