#!/bin/bash
# Production start script for Railway.
# Waits for Postgres to accept connections before running Alembic migrations,
# then starts uvicorn. Railway starts app containers before the DB is fully
# ready, so without this wait the alembic connection times out on cold starts.
set -e

# ── 1. Wait for Postgres ─────────────────────────────────────────────────────
# Skip for SQLite (local dev). For Postgres, poll up to 60 s (30 × 2 s).
uv run python3 - <<'PYEOF'
import os, sys, time
from sqlalchemy import create_engine, text

raw = os.environ.get("DATABASE_URL", "")
if not raw or raw.startswith("sqlite"):
    print("SQLite / no URL — skipping Postgres wait.")
    sys.exit(0)

url = (raw
       .replace("postgresql://", "postgresql+psycopg://")
       .replace("postgres://",   "postgresql+psycopg://"))

for attempt in range(1, 31):
    try:
        engine = create_engine(url, pool_pre_ping=True,
                               connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"  Postgres ready (attempt {attempt})", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"  [{attempt}/30] not ready: {exc}", flush=True)
        time.sleep(2)

print("ERROR: Postgres never became ready after 60 s.", flush=True)
sys.exit(1)
PYEOF

# ── 2. Run Alembic migrations ─────────────────────────────────────────────────
echo "==> Running Alembic migrations..."
uv run alembic upgrade head

# ── 3. Start uvicorn ──────────────────────────────────────────────────────────
echo "==> Starting uvicorn on port ${PORT:-8000}..."
exec uv run uvicorn roulette_agent.app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"
