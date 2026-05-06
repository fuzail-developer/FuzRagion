import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "pdf_tracker.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if DATABASE_URL.startswith("postgresql://"):
    parts = urlsplit(DATABASE_URL)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    # Render/managed Postgres commonly requires TLS; enforce when missing.
    if "sslmode" not in q:
        q["sslmode"] = "require"
    DATABASE_URL = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment)
    )

is_sqlite = DATABASE_URL.startswith("sqlite")
is_hosted = any(
    os.getenv(name)
    for name in ("RENDER", "RENDER_SERVICE_ID", "RAILWAY_ENVIRONMENT", "FLY_APP_NAME")
)
if is_hosted and is_sqlite:
    print(
        "Warning: Hosted deployment is using sqlite. Set DATABASE_URL to PostgreSQL to avoid data loss."
    )
engine_kwargs = {}
if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Avoid long hangs on first DB hit (observed on hosted deployments).
    engine_kwargs["connect_args"] = {"connect_timeout": 8}
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_recycle"] = 300

def _build_engine(url: str):
    return create_engine(url, **engine_kwargs)


engine = _build_engine(DATABASE_URL)

# Optional local-only fallback: if hosted DB is unreachable, switch to sqlite for dev.
allow_sqlite_fallback = os.getenv("ALLOW_SQLITE_FALLBACK", "false").lower() == "true"
if not is_sqlite and allow_sqlite_fallback and not is_hosted:
    try:
        with engine.connect() as _conn:
            pass
    except Exception as exc:
        fallback_url = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
        print(f"Warning: DATABASE_URL unreachable, falling back to sqlite. Error: {exc}")
        DATABASE_URL = fallback_url
        engine_kwargs = {"connect_args": {"check_same_thread": False}}
        engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
