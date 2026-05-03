#!/bin/bash
# Continuous scorer: runs score_all every 60s. New posts get scored within a minute.
# Replaces the absent CronJob. Auto-restarts inner script on failure.
set -u
cd "$(dirname "$0")/.."

while true; do
    echo "[$(date)] starting score_all pass" >&2
    /Users/tim/.local/bin/uv run python -m fedi_studio.workers.score_all 2>&1 || true
    echo "[$(date)] pass complete, sleeping 60s" >&2
    sleep 60
done
