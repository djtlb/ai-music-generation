# Multi-stage Dockerfile for AI Music Generation Platform
# Stage 1: Frontend build (if frontend exists)
FROM node:20-alpine AS frontend
WORKDIR /app
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
# Install deps (tolerate whichever lockfile exists)
RUN if [ -f package-lock.json ]; then npm ci; elif [ -f pnpm-lock.yaml ]; then npm install -g pnpm && pnpm install --frozen-lockfile; elif [ -f yarn.lock ]; then yarn install --frozen-lockfile; else npm install; fi
COPY . .
RUN if [ -f vite.config.ts ]; then npm run build || echo "Skipping frontend build (dev placeholder)"; fi

# Stage 2: Backend / API image
FROM python:3.11-slim AS backend
ARG INSTALL_BUILD_DEPS=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps (audio libs minimal subset; extend as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        && if [ "$INSTALL_BUILD_DEPS" = "1" ]; then apt-get install -y --no-install-recommends build-essential; fi \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (split support)
COPY backend/requirements.txt backend/requirements.txt
COPY backend/requirements-prod.txt backend/requirements-prod.txt
RUN pip install --upgrade pip && pip install -r backend/requirements-prod.txt

# Copy application (only backend + built frontend assets)
COPY backend /app/backend
COPY --from=frontend /app/dist /app/dist
COPY .env.example /app/.env.example
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user
RUN useradd -m appuser
USER appuser

ENV ENV=production \
    LOG_LEVEL=INFO \
    FAST_DEV=0 \
    PORT=8000 \
    WORKERS=2

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
