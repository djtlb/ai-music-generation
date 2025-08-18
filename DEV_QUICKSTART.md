# Dev Quickstart

## Fast Path (recommended)

```bash
npm run up
```

This launches:

- FAST_DEV backend (minimal) on <http://localhost:8000>
- Frontend (Vite) on <http://localhost:5173>
- Proxy (dynamic port, printed when ready)
- Waits for health of backend + frontend before declaring READY

## Full Backend

```bash
npm run up:full
```

Loads full backend (`backend.main:app`). May require full Python dependencies.

## Classic Simple Mode

```bash
./dev.sh            # fast (uses dev_app)
./dev.sh --no-fast  # full backend
```

## Switching Modes

FAST_DEV=0 forces full mode:

```bash
FAST_DEV=0 npm run up
```

## Health Endpoints

- Backend: <http://localhost:8000/health>
- Frontend root: <http://localhost:5173/>

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Backend not healthy | Check backend logs (prefixed [backend]) – missing deps? Run `pip install -r backend/requirements.txt` |
| Frontend not healthy | Ensure `node_modules` exists: `npm install` |
| Proxy missing | Wait a few seconds; it binds after backend/frontend start |
| Need full features | Use `npm run up:full` |

## Common Tasks

Run tests (Python):

```bash
pytest -q
```

Build frontend prod bundle:

```bash
npm run build
```

## Version Bump (manual)

```bash
npm run version:patch
```

## Next Enhancements (optional)

- Auto version bump on first healthy full boot
- Windows PowerShell wrapper
- Dependency auto-install inside orchestrator

