"""Application configuration for the Flask web app."""

import os
import secrets
from datetime import timedelta
from pathlib import Path

# .../AI_Customer_BI_Project/backend/routes/routes/config.py
BASE_DIR = Path(__file__).resolve().parents[3]

INSTANCE_DIR = BASE_DIR / "instance"
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"


def _load_secret_key():
    """Use SECRET_KEY from the environment, or persist a generated one."""
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    key_file = INSTANCE_DIR / "secret_key.txt"

    if key_file.exists():
        value = key_file.read_text(encoding="utf-8").strip()
        if value:
            return value

    value = secrets.token_hex(32)
    key_file.write_text(value, encoding="utf-8")
    return value


def _normalize_database_url(url):
    """Map hosted database URLs to the SQLAlchemy driver we ship.

    Render/Heroku style URLs use the bare ``postgres://`` or
    ``postgresql://`` schemes, which SQLAlchemy would route to psycopg2.
    We install psycopg 3 instead, so pin the driver explicitly.
    """
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("mysql://"):
        return "mysql+pymysql://" + url[len("mysql://"):]
    return url


def _database_uri():
    """Prefer DATABASE_URL (Postgres/MySQL), otherwise SQLite."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return _normalize_database_url(env_url)

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    sqlite_path = (INSTANCE_DIR / "app.db").as_posix()
    return f"sqlite:///{sqlite_path}"


class Config:
    SECRET_KEY = _load_secret_key()

    SQLALCHEMY_DATABASE_URI = _database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    MAX_CONTENT_LENGTH = 4 * 1024 * 1024  # 4 MB uploads

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # Render sets RENDER=true and always serves HTTPS, so default to secure
    # cookies there while keeping plain HTTP working for local development.
    SESSION_COOKIE_SECURE = os.environ.get(
        "COOKIE_SECURE",
        "1" if os.environ.get("RENDER") else "0",
    ) == "1"
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)

    TEMPLATE_DIR = TEMPLATE_DIR
    STATIC_DIR = STATIC_DIR
    DATA_DIR = DATA_DIR
    MODELS_DIR = MODELS_DIR

    FIRST_USER_IS_ADMIN = True
    APP_NAME = "AI Customer & Business Intelligence"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


def get_config():
    env = os.environ.get("FLASK_ENV", "development").lower()
    return ProductionConfig if env == "production" else DevelopmentConfig
