# Deployment Guide — Railway

## Prerequisites

- [Railway account](https://railway.app) (free tier gives $5/month credit)
- [Railway CLI](https://docs.railway.app/guides/cli) installed: `npm install -g @railway/cli`
- Git repo pushed to GitHub (or deploy directly from CLI)

---

## 1. Install Railway CLI and log in

```bash
npm install -g @railway/cli
railway login
```

---

## 2. Create a new Railway project

```bash
cd /path/to/RouletteBias-AwareAgent
railway init          # creates a new project, links to this directory
```

Or go to [railway.app/new](https://railway.app/new) and connect your GitHub repo.

---

## 3. Add a PostgreSQL service

In the Railway dashboard:
1. Click **+ New** → **Database** → **Add PostgreSQL**
2. Railway automatically injects `DATABASE_URL` into all services in the same project

From CLI:
```bash
railway add --plugin postgresql
```

After provisioning, verify the variable is set:
```bash
railway variables | grep DATABASE_URL
```

---

## 4. Set required environment variables

```bash
railway variables set API_KEY=<your-secret-api-key>
railway variables set ANTHROPIC_API_KEY=<your-anthropic-key>
```

`DATABASE_URL` is injected automatically by the Postgres plugin — **do not set it manually**.

### Full list of environment variables

| Variable | Source | Description |
|----------|--------|-------------|
| `DATABASE_URL` | Railway Postgres plugin (auto) | PostgreSQL connection string |
| `API_KEY` | Set manually | Secret header value for `X-API-Key` auth |
| `ANTHROPIC_API_KEY` | Set manually | Anthropic API key for Claude calls |

---

## 5. Deploy

### Option A — from CLI

```bash
railway up
```

Railway detects `uv.lock` → uses nixpacks with uv → installs deps → runs `Procfile`:

```
web: uv run alembic upgrade head && uv run uvicorn roulette_agent.app:app --host 0.0.0.0 --port $PORT
```

### Option B — from GitHub

Connect your repo in the Railway dashboard. Every push to `main` auto-deploys.

---

## 6. Get the public URL

```bash
railway domain
```

Or in the dashboard: **Settings → Domains → Generate Domain**.

The URL looks like `https://<project-name>.up.railway.app`.

---

## 7. Smoke test

Replace `<URL>` and `<API_KEY>` with your values:

```bash
# Health check (no auth)
curl https://<URL>/health

# Create a session
curl -X POST https://<URL>/session/new \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "wheel_type": "american",
    "bankroll": 800,
    "bet_unit": 5,
    "excluded_dozens": [],
    "recent_history": [17,5,17,32,17,11,17,5,17,32,17,0,17,5,17,22,17,8,17,32],
    "external_stats": {"black_pct": 0.62, "odd_pct": 0.58},
    "external_stats_n_estimate": 200
  }'

# Save the session_id from the response, then:
SESSION_ID=<session_id_from_response>

curl -X POST https://<URL>/session/$SESSION_ID/spin \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{"result_number": 17}'

curl https://<URL>/session/$SESSION_ID/state \
  -H "X-API-Key: <API_KEY>"
```

---

## 8. View logs

```bash
railway logs
```

Or in the dashboard under **Deployments → View Logs**.

---

## Cost and usage notes

### Railway free tier
- Free tier provides **$5/month** credit (≈ 500 CPU-hours on the starter plan).
- The Postgres plugin also counts against credit.
- To avoid surprise charges: set a **spending limit** in Railway → Settings → Billing.
- If you exceed the free tier, either upgrade to the Hobby plan ($20/month) or pause the service.

### Anthropic API costs
- Each `/session/new` and `/spin` call makes **1–3 Claude API calls** (tool-use rounds).
- With `claude-sonnet-4-6`, estimate roughly **$0.01–0.05 per spin** depending on history length.
- Monitor usage at [console.anthropic.com](https://console.anthropic.com) → Usage.
- Set a **monthly spend limit** in the Anthropic console to cap costs.

### Security reminders
- **Never commit `.env`** — it is already in `.gitignore`.
- Rotate `API_KEY` and `ANTHROPIC_API_KEY` if they appear in logs or are shared.
- The Railway dashboard shows env vars — treat project access as sensitive.

---

## Local dev vs production differences

| | Local | Railway |
|--|-------|---------|
| Database | `sqlite:///./dev.db` | PostgreSQL (via `DATABASE_URL`) |
| Start command | `bash scripts/run_dev.sh` | `Procfile` (via Railway) |
| Migrations | `alembic upgrade head` (same) | Run automatically on start |
| Hot reload | `--reload` flag | No reload (production) |

---

## Rollback

```bash
railway rollback          # rolls back to the previous successful deployment
```

Or in the dashboard: **Deployments → previous deploy → Rollback**.
