import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context

# Alembic Config object
config = context.config

# Set up logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import our models' metadata for autogenerate support
from roulette_agent.models import Base  # noqa: E402
target_metadata = Base.metadata


def _get_url() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite:///./dev.db")
    # Railway injects postgresql:// or postgres://, both route to psycopg2 by
    # default. Rewrite to psycopg3 driver so we don't need psycopg2 installed.
    url = url.replace("postgresql://", "postgresql+psycopg://")
    url = url.replace("postgres://", "postgresql+psycopg://")
    return url


def run_migrations_offline() -> None:
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _get_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    connectable = create_engine(url, poolclass=pool.NullPool, connect_args=connect_args)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # required for SQLite ALTER TABLE support
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
