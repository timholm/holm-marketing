-- Add last_crawled_at column to v1 profiles table for lead crawler tracking.
-- Run against fedi_discover_full DB (v1).

ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS last_crawled_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;

-- Index for efficient cursor-based scanning (ordered by oldest-first)
CREATE INDEX IF NOT EXISTS idx_profiles_last_crawled
ON profiles (COALESCE(last_crawled_at, '1970-01-01') ASC)
WHERE is_active = 1;
