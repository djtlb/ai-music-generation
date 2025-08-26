# 🛠️ System Architecture & Backend

- **Backend**: Python FastAPI (with some Flask legacy endpoints), running in Docker containers on a Hostinger VPS.
- **AI Engine**: Advanced music generation, stem splitting, and DJ drop creation using custom and open-source models.
- **Database**: PostgreSQL for user/projects, Redis for job queueing and real-time updates.
- **Supabase Edge Functions**: Secure proxy for frontend-backend communication, handling authentication and rate limiting.
- **Security**: All endpoints protected by API keys/tokens, CORS, and HTTPS. Stripe integration for subscription management.
- **DevOps**: Docker Compose for orchestration, nginx/Caddy for SSL and reverse proxy, structured logging, and health monitoring.

---

**How Everything Connects:**

- **Frontend** (Hostinger Website Builder)
  ⬇️ fetches
- **Supabase Edge Function** (adds auth, proxies)
  ⬇️ calls
- **FastAPI Backend** (AI, Flask legacy, Docker, DB, Redis)
  ⬇️ stores
- **Postgres/Redis** (data, jobs, cache)
  ⬆️
- **Stripe** (for payments/tier checks)

---

This block can be added to your documentation or system overview for a complete, professional description.
