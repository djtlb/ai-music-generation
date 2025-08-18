# AI Music Generation Platform (Beta)

Full‑stack system for automated multi‑stage music generation, arrangement, mixing and progress tracking. 

## High‑Level Overview
Backend: FastAPI service orchestrating a multi‑stage pipeline (planning → generation → arrangement → rendering → mixing). Real‑time progress via WebSockets and aggregate REST endpoints. In‑memory orchestration (DB persistence planned).
Frontend: React + Vite + TypeScript UI triggering full‑song generation, showing stage progress and aggregate project status.
Infrastructure: Multi‑stage Docker build (frontend → Python runtime), optional Postgres + Redis via docker‑compose, dev orchestrator script for local rapid iteration (FAST_DEV mode).

## Quick Start (Local Dev)
Option A – Orchestrator (auto installs, runs backend + frontend):
1. Ensure Python 3.11+, Node 18+, Redis (optional), Postgres (optional if using compose) installed.
2. Copy `.env.example` to `.env` and adjust secrets.
3. Run: `npm run connect` (or `pnpm connect` if using pnpm) – this sets up venv, installs deps, launches backend & frontend with health polling.

Option B – Docker Compose:
1. Copy `.env.example` to `.env`.
2. `docker compose up --build` (uses multi‑stage Dockerfile, launches api + postgres + redis).
3. Frontend static build is baked into the image; access API at `http://localhost:8000` and bundled UI (if served) or run `npm run dev` separately for hot reload.

## Core Endpoints
POST /api/v1/music/generate/full-song → Start full pipeline (returns project_id).
GET  /api/v1/music/project/{project_id}/aggregate → Consolidated project + stage metadata + computed progress.
GET  /api/v1/music/project/{project_id}/status → Lightweight status (subset).
GET  /health, /health/ready → Liveness / readiness.
WebSocket: /ws → Subscribe with `{ "action":"subscribe", "project_id":"<id>" }` to receive events.

## WebSocket Event Types
stage.started, stage.progress, stage.completed, project.completed, project.failed.
Each event includes minimal envelope plus `project_id`, `stage`, and when applicable `progress` (0..1) or `error`.

## Auth (Development Mode)
In FAST_DEV the frontend retrieves a temporary dev JWT via `/api/v1/dev/token`. Do NOT expose this endpoint in production (disable FAST_DEV / set environment to production & gate dev routers accordingly).

## Environment Variables
Defined in `.env.example` (all prefixed with `AIMUSIC_`). Key vars:
- AIMUSIC_SECRET_KEY – JWT signing key (rotate for production)
- AIMUSIC_DATABASE_URL – Async SQLAlchemy database URL
- AIMUSIC_REDIS_URL – Redis instance (caching / future task coordination)
- AIMUSIC_ALLOWED_ORIGINS – Comma list for CORS
- AIMUSIC_FAST_DEV – Enables dev token endpoint & permissive defaults
- AIMUSIC_ENVIRONMENT – development | staging | production

Feature flags (all boolean): `AIMUSIC_ENABLE_BLOCKCHAIN`, `AIMUSIC_ENABLE_COLLABORATION`, `AIMUSIC_ENABLE_MARKETPLACE`, `AIMUSIC_ENABLE_ENTERPRISE`.

## Running Tests
Backend tests (selected): `python run_tests.py` or targeted scripts (see `tests/` / `test_*.py`). Add persistent DB & migrate state before enabling DB‑backed orchestration tests.

## Deployment (Beta)
1. Build image: `docker build -t aimusic:beta .`
2. Run with external Postgres + Redis (supply URLs via env vars; do not rely on in‑memory orchestrator for durability).
3. Set `AIMUSIC_ENVIRONMENT=production`, `AIMUSIC_FAST_DEV=false`, restrict `AIMUSIC_ALLOWED_ORIGINS`.
4. Provide strong `AIMUSIC_SECRET_KEY`.
5. Add reverse proxy (e.g., Nginx / Caddy) for TLS termination and caching of static assets.

## Production Hardening Roadmap
- [ ] Persist project + stage state (replace in‑memory) in Postgres tables.
- [ ] Background task queue (Celery / RQ / custom) for long stages & retry semantics.
- [ ] Replace dev JWT with real user auth (OAuth / email magic link / etc.).
- [ ] Add rate limiting + abuse detection (currently basic values in settings).
- [ ] Structured log export (e.g., to ELK / Loki) & error tracking (Sentry).
- [ ] Observability: metrics endpoint (Prometheus) + dashboards.
- [ ] CI/CD: build, test, scan, push image, migration step, deploy.
- [ ] Version metadata injection (commit hash, build timestamp endpoint).
- [ ] Security review (dependency scanning, secret scanning, CORS tighten, headers middleware).

See `BETA_LAUNCH_CHECKLIST.md` for current status tracking.

## Contributing
1. Fork / branch
2. Add or update tests for behavior changes
3. Run lint/tests locally
4. Submit PR with concise description + risk notes

## License
MIT (see `LICENSE`).

## Disclaimer
Beta software – expect breaking changes while persistence & authentication layers mature.

