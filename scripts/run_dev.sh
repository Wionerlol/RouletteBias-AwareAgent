#!/bin/bash
set -e

export DATABASE_URL=sqlite:///./dev.db
export API_KEY=dev-key-123
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

echo "Running Alembic migrations..."
uv run alembic upgrade head

echo "Starting Uvicorn on port 8000..."
uv run uvicorn roulette_agent.app:app --reload --port 8000
