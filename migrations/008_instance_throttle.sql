-- Per-instance rate-limit throttle for parallel lead_crawler replicas.
-- Prevents 429 storms when multiple replicas hit the same instance.

CREATE TABLE IF NOT EXISTS instance_throttle (
    instance TEXT PRIMARY KEY,
    last_hit_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_instance_throttle_last_hit
ON instance_throttle (last_hit_at DESC);

INSERT INTO migrations (version) VALUES (8) ON CONFLICT DO NOTHING;
