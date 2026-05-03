#!/bin/bash
# Supervisor for all fedi-studio workers. Each worker runs in its own respawn loop.
# Logs go to /tmp/<worker>.log with a 50MB cap (logrotate-on-restart).
# Starts:  pf_pg_watchdog, web, firehose, sync_v1_to_v2, bulk_import_v1, score_all_loop, enrich_pending_candidates
# Stops:   pkill -f 'fedi-studio supervisor:'  (each child carries this tag in argv[0])

set -u
# Use absolute path so the supervisor works whether invoked directly,
# from launchd (where $0 may not be the script path), or via stdin (`bash <script`).
cd /Users/tim/Documents/Holm/tools/fedi-studio

UV=/Users/tim/.local/bin/uv
LOG_DIR=/tmp
LOG_CAP_BYTES=$((50 * 1024 * 1024))   # 50 MB
RESTART_BACKOFF_S=5

# Pull MASTODON_TOKEN once on supervisor start. Children inherit via env.
export MASTODON_TOKEN=$(kubectl get secret -n fedi-discover fedi-discover-secrets -o jsonpath='{.data.mastodon-token}' | base64 -d 2>/dev/null)
export V1_DSN="host=localhost port=30141 dbname=fedi_discover_full user=mastodon password=mastodon"
# Mastodon main DB (read-only access via per-row SELECT in /relationships).
# Port-forwarded by pf_mastodon worker below.
export MASTODON_DSN="host=localhost port=30142 dbname=mastodon user=mastodon password=mastodon"

if [ -z "$MASTODON_TOKEN" ]; then
    echo "WARNING: MASTODON_TOKEN unavailable — enrich daemon will not run" >&2
fi

# Rotate a log file if it's grown past LOG_CAP_BYTES.
rotate_if_big() {
    local f=$1
    if [ -f "$f" ]; then
        local sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$sz" -gt "$LOG_CAP_BYTES" ]; then
            mv "$f" "$f.1"
        fi
    fi
}

# Run a command, restart it on exit, sleep between restarts.
respawn() {
    local name=$1; shift
    local logf=$LOG_DIR/${name}.log
    while true; do
        rotate_if_big "$logf"
        echo "[$(date)] supervisor: starting $name" >> "$logf"
        # Tag the process so we can find it with pgrep
        # exec into the command preserving the tag
        bash -c "exec -a 'fedi-studio supervisor:$name' $*" >> "$logf" 2>&1
        echo "[$(date)] supervisor: $name exited rc=$?, restart in ${RESTART_BACKOFF_S}s" >> "$logf"
        sleep "$RESTART_BACKOFF_S"
    done
}

# Port-forward (separate watchdog: kubectl is itself the worker)
respawn pf_pg \
    "kubectl port-forward -n fedi-discover svc/fedi-postgresql 30141:5432" &

# Mastodon main DB port-forward (read-only; powers /relationships mutual badges).
respawn pf_mastodon \
    "kubectl port-forward -n mastodon svc/mastodon-postgresql 30142:5432" &

# Wait for PG to be reachable before starting DB-using workers
echo "supervisor: waiting for PG up..." >&2
until nc -z localhost 30141 2>/dev/null; do sleep 2; done
echo "supervisor: PG up" >&2

# Best-effort wait for Mastodon PG (don't block forever — degrades gracefully)
echo "supervisor: waiting for Mastodon PG up..." >&2
for _ in $(seq 1 15); do
    if nc -z localhost 30142 2>/dev/null; then echo "supervisor: Mastodon PG up" >&2; break; fi
    sleep 2
done

respawn web \
    "$UV run uvicorn fedi_studio.web.app:app --host 0.0.0.0 --port 8765" &

respawn firehose \
    "$UV run python -m fedi_studio.workers.firehose" &

respawn sync_v1 \
    "$UV run python -m fedi_studio.workers.sync_v1_to_v2" &

respawn bulk_import \
    "$UV run python -m fedi_studio.workers.bulk_import_v1" &

respawn score_all \
    "$(pwd)/scripts/run_score_all_loop.sh" &

if [ -n "$MASTODON_TOKEN" ]; then
    respawn enrich \
        "$UV run python -m fedi_studio.workers.enrich_pending_candidates" &
    respawn tim_outbox \
        "$UV run python -m fedi_studio.workers.pull_tim_outbox" &
fi

echo "supervisor: all children launched, pid=$$" >&2
# Wait forever (until pkill)
wait
