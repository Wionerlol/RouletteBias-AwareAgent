FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached until pyproject.toml changes)
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Copy remaining runtime files
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/

CMD ["bash", "scripts/start.sh"]
