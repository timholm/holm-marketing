# Follow-Graph Crawler: Deployment & Verification

## Overview
The follow-graph crawler bypasses the 50% `lookup_failed` waste from v1 crawling by seeding from Tim's high-scoring candidates and pulling their following/followers graphs. This expands the candidate pool with LIVE accounts.

## Files Created

### 1. Migration
- **`migrations/007_followgraph.sql`**
  - Adds `holm_account_id TEXT` and `graph_crawled_at TIMESTAMPTZ` to `candidates` table
  - Adds `holm_account_id TEXT` to `candidates_pending` table
  - Creates index `idx_candidates_graph_crawl` for efficient seed selection (score DESC, where reviewed=false AND graph_crawled_at IS NULL)

### 2. Worker
- **`src/fedi_studio/workers/follow_graph_crawler.py`**
  - Single-seed-at-a-time crawler (sequential, polite rate)
  - Fetches holm_account_id via `/accounts/lookup` if not cached
  - Concurrently fetches up to 5 pages (400 max) of both following and followers
  - Inserts unique accts into `candidates_pending` with source_post_uri `graph:from:{seed_acct}`
  - Updates `candidates.graph_crawled_at = NOW()` after each seed
  - Sleeps 2s between seeds to respect rate limits (~1100 calls/5min, under 1500 token limit)
  - Handles 429/5xx with backoff, treats 404/410 as terminal (marks crawled, continues)

### 3. K8s Deployment
- **`k8s/follow-graph-crawler-deployment.yaml`**
  - Single replica
  - Image: `192.168.8.197:30080/tim/fedi-studio:latest`
  - Command: `python -m fedi_studio.workers.follow_graph_crawler`
  - Resources: 100m CPU req / 500m limit, 256Mi mem req / 512Mi limit
  - PSA-restricted security context (read-only root, no privileges, seccomp)
  - Env vars: V1_DSN, V2_DSN, FEDI_STUDIO_DSN, MASTODON_URL, MASTODON_TOKEN (from secret)

- **`k8s/kustomization.yaml`**
  - Orchestrates deployment in fedi-discover namespace

### 4. Optional Enhancement
- **Modified `src/fedi_studio/workers/enrich_pending_candidates.py`**
  - Updated `fetch_pending_batch()` to retrieve `holm_account_id` from `candidates_pending`
  - Updated `enrich_batch()` signature and documentation to note ID optimization
  - Prepares pipeline to skip lookup step when ID is pre-cached (future enhancement)

## Deployment Steps

### Step 1: Apply Migration
```bash
# Copy migration to cluster
kubectl cp migrations/007_followgraph.sql fedi-discover/fedi-postgresql-0:/tmp/007.sql

# Apply it
kubectl exec -n fedi-discover fedi-postgresql-0 -- \
  psql -U mastodon -d fedi_studio -f /tmp/007.sql

# Verify columns exist
kubectl exec -n fedi-discover fedi-postgresql-0 -- \
  psql -U mastodon -d fedi_studio -c "
    SELECT column_name FROM information_schema.columns 
    WHERE table_name='candidates' AND column_name IN ('holm_account_id', 'graph_crawled_at');
  "
```

### Step 2: Build & Push Image
```bash
cd /Users/tim/Documents/Holm/tools/fedi-studio

# Build for ARM64 (Holm cluster architecture)
docker buildx build --platform=linux/arm64 \
  -f Dockerfile.fedi-studio \
  -t 192.168.8.197:30080/tim/fedi-studio:latest \
  --load .

# Push to local registry
docker push 192.168.8.197:30080/tim/fedi-studio:latest
```

### Step 3: Deploy K8s Manifest
```bash
# Apply deployment
kubectl apply -k k8s/

# Verify rollout
kubectl rollout status deploy/follow-graph-crawler -n fedi-discover --timeout=180s

# Check initial logs
kubectl logs -n fedi-discover -l app=follow-graph-crawler --tail=30 -f
```

### Step 4: Verify Operation (After 5 minutes)
```bash
# Check new pending entries from follow-graph
kubectl exec -n fedi-discover fedi-postgresql-0 -- \
  psql -U mastodon -d fedi_studio -c "
    SELECT COUNT(*) as new_pending 
    FROM candidates_pending 
    WHERE source_post_uri LIKE 'graph:%';
  "

# Sample the crawler logs
kubectl logs -n fedi-discover deploy/follow-graph-crawler --tail=60
```

## Performance Expectations

### Rate Limits
- Token auth: 1500 reqs / 5min (300 unauth)
- Per seed: ~1 lookup + ~10 paginated fetches = 11 API calls
- Safe throughput: ~100 seeds / 5min ≈ 1100 API calls total

### Yield Estimates
From a typical candidate (score >= 30):
- Following: avg 100-200 accounts
- Followers: avg 150-300 accounts
- Unique union: ~300-400 accts per seed (before dedup against existing)
- Expected new_pending: ~250-350 per seed after dedup

**Hourly throughput** (at 2s/seed):
- ~1800 seeds/hour
- At ~300 new pending per seed
- ≈ 540K new pending candidates/hour (before exclusions)

Actual insertion into `candidates` depends on `enrich_pending_candidates` daemon's processing rate (~40/batch × 60s poll = 40/min = 2.4K/hour max at current config).

## Troubleshooting

### Pod stuck pending or crashing
```bash
kubectl describe pod -n fedi-discover -l app=follow-graph-crawler
kubectl logs -n fedi-discover -l app=follow-graph-crawler --previous
```

### Database connection errors
- Check if `fedi-postgresql-0` is running: `kubectl get pods -n fedi-discover`
- Verify secret: `kubectl get secret -n fedi-discover fedi-discover-secrets -o jsonpath='{.data.mastodon-token}' | base64 -d`

### No new pending entries after 5 minutes
```bash
# Check if any seeds exist
kubectl exec -n fedi-discover fedi-postgresql-0 -- \
  psql -U mastodon -d fedi_studio -c "
    SELECT COUNT(*) FROM candidates WHERE reviewed=FALSE AND graph_crawled_at IS NULL;
  "

# If seeds exist, check logs for errors
kubectl logs -n fedi-discover deploy/follow-graph-crawler --tail=100
```

### 429 Retry-After backoff
Expected and handled. Logs show `429 Retry-After X.Xs` — crawler continues after wait.

### 404/410 on dead profiles
Expected for inactive/deleted accounts. Marked as crawled and skipped in future runs.

## Stopping & Cleanup

To pause crawling:
```bash
kubectl scale deploy/follow-graph-crawler -n fedi-discover --replicas=0
```

To resume:
```bash
kubectl scale deploy/follow-graph-crawler -n fedi-discover --replicas=1
```

To remove entirely:
```bash
kubectl delete -k k8s/ -n fedi-discover
# (Migration stays; data persists)
```
