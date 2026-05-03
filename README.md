# fedi-studio

A reading and posting assistant for Mastodon. Companion to my own presence on holm.community.

**This is not a bot.** It does not auto-follow, auto-like, or auto-boost anything. Every action originates from a human click in the same session.

## What it does

- Listens to the Mastodon streaming API and stores posts of potential interest
- Scores posts using a personal classifier (trained on what I actually like)
- Builds a private morning catch-up digest
- Drafts replies, intro posts, and weekly ritual posts (#FollowFriday, #SolarPunkSunday)
- Tracks relationships so I notice when I haven't talked to someone in a while

## What it does NOT do

- Send a `Follow` activity automatically
- Send a `Like` activity automatically
- Send an `Announce` (boost) activity automatically
- Reply on my behalf
- Index or store posts from accounts that signal `#nobot`, `#noindex`, or `discoverable=false`

If you find this account interacting with you and you didn't initiate the interaction, that is a bug. File an issue.

## Architecture

See `/Users/tim/.claude/plans/fuzzy-weaving-map.md` for the full plan.

```
Mastodon SSE -> Rust listener -> NATS JetStream
                                       v
                               Python ingest worker
                               (embedding + dedup)
                                       v
                               Postgres 16 (rpi-12, NVMe)
                               - posts (partitioned by month)
                               - post_scores
                               - events (immutable log)
                                       v
                               Python scorer (Model2Vec + SGDClassifier)
                                       v
                               FastAPI + HTMX UI
                               (read-only consumption surface)
```

## Layout

```
fedi-studio/
  src/fedi_studio/         Python package
    web/                   FastAPI routes, HTMX templates
    workers/               Background workers (ingest, scorer, digest)
    services/              Business logic (scoring, dedup, embedding)
    models/                DB access layer
    api/                   API routes (read-only)
  migrations/              SQL schema migrations
  listener/                Rust SSE listener (separate cargo project)
  k8s/                     Kubernetes manifests
  scripts/                 One-off scripts (data import, debug)
  tests/                   pytest tests
```
