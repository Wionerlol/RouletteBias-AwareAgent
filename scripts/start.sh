#!/bin/bash
# Production start script for Railway.
# pip install . puts alembic/uvicorn directly on PATH — no uv run needed.
set -e

# ── 1. Wait for Postgres ─────────────────────────────────────────────────────
python3 - <<'PYEOF'
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
alembic upgrade head

# ── 3. Start uvicorn ──────────────────────────────────────────────────────────
echo "==> Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn roulette_agent.app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"
