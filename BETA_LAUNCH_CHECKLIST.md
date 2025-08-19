# Beta Launch Checklist

Legend: [ ] pending | [~] in progress | [x] done

## Core Functionality

- [x] Full-song generation endpoint
- [x] Aggregate project status endpoint
- [x] WebSocket real-time stage events
- [x] Frontend integration (trigger + progress UI)
- [x] In-memory orchestration layer
- [~] Persistent project + stage storage (Postgres)
- [ ] Background task queue (for long-running stages)

## Auth & Security

- [x] Dev JWT issuance (FAST_DEV only)
- [ ] Production auth (real users, token refresh)
- [ ] Rate limiting enforcement & tuning
- [ ] Harden CORS (production origins only)
- [ ] Secure secret management (vault / env provisioning)

## Observability & Ops

- [x] Structured logging
- [ ] Central log aggregation pipeline
- [ ] Metrics endpoint enabled + dashboard
- [ ] Error tracking (Sentry / equivalent)
- [ ] Health & readiness probes documented for orchestrators

## Deployment

- [x] Multi-stage Dockerfile
- [x] docker-compose for dev services
- [ ] Production image scan (vuln scanning CI step)
- [ ] Automated version / build metadata injection
- [ ] CD pipeline (build → test → push → deploy)

## Testing

- [x] Existing unit/integration tests runnable
- [ ] Add tests for WebSocket progress events
- [ ] Add tests for aggregate endpoint consistency
- [ ] Load test representative generation workflow

## Documentation

- [x] Beta README
- [x] Env vars documented
- [ ] API reference (OpenAPI curation / pruning)
- [ ] Architecture deep-dive (current doc refinement)

## Data & Persistence

- [x] Schema design for projects/stages (models + auto create)
- [ ] Migration scripts (pending Alembic setup)
- [ ] Backfill strategy (if pre-beta data retained)

## Performance

- [ ] Benchmark stage durations (baseline numbers)
- [ ] Identify and cache hot model assets
- [ ] Streaming partial audio output (future enhancement)

## Compliance & Legal

- [ ] Dependency license audit
- [ ] Content usage & model provenance statement
- [ ] Data retention & privacy policy draft

## Next Suggested Sequence

1. Design & implement persistence layer (projects, stages, events).
2. Introduce task queue for async stage execution + retries.
3. Add auth system (user accounts) and remove dev token outside development.
4. Instrument metrics + error tracking.
5. CI/CD pipeline & image scanning.
6. Harden security (CORS, headers, rate limits) & finalize docs.
