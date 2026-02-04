"""Local JSON-based storage helpers for offline mode.

Simple, synchronous JSON storage for users and analyses used by the Flet app.
"""

import logging
import os
import json
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

# Try bcrypt if available, otherwise fall back to pbkdf2_hmac (secure fallback)
try:
    import bcrypt
    _HAS_BCRYPT = True
except Exception:
    import hashlib, binascii, secrets
    _HAS_BCRYPT = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
ANALYSES_FILE = os.path.join(DATA_DIR, "analyses.json")

os.makedirs(DATA_DIR, exist_ok=True)


def _read_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _hash_password(password: str) -> str:
    if _HAS_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    # pbkdf2_hmac fallback: store as algo$salt$hex
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"pbkdf2_sha256${salt}${binascii.hexlify(dk).decode()}"


def _check_password(password: str, hashed: str) -> bool:
    if _HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    # fallback verify pbkdf2_hmac
    try:
        algo, salt, dk_hex = hashed.split('$', 2)
        if algo != 'pbkdf2_sha256':
            return False
        dk = binascii.unhexlify(dk_hex)
        newdk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return secrets.compare_digest(dk, newdk)
    except Exception:
        return False


# Users API

def find_user_by_email(email: str):
    """Return user dict for a given email, or None if not found."""
    users = _read_json(USERS_FILE)
    for u in users:
        if u.get("email") == email:
            return u
    return None


def create_user(username: str, email: str, password: str):
    """Create a new user and persist into local JSON storage.

    Raises ValueError if user with email already exists.
    """
    if find_user_by_email(email):
        raise ValueError("Usuario ya existe")
    users = _read_json(USERS_FILE)
    user = {
        "id": str(uuid4()),
        "username": username,
        "email": email,
        "password": _hash_password(password),
        "created_at": datetime.utcnow().isoformat()
    }
    users.append(user)
    _write_json(USERS_FILE, users)
    logger.info("Usuario creado: %s", email)
    return user


def authenticate_user(email: str, password: str):
    """Authenticate a user using local JSON storage. Returns user dict on success, else None."""
    user = find_user_by_email(email)
    if not user:
        return None
    if not _check_password(password, user["password"]):
        return None
    logger.info("Usuario autenticado: %s", email)
    return user


# Analyses API

def add_analysis(user_id: str, username: str, model_name: str, file_path: str, result: dict):
    """Persist an analysis record in local JSON storage."""
    analyses = _read_json(ANALYSES_FILE)
    record = {
        "id": str(uuid4()),
        "user_id": user_id,
        "username": username,
        "model": model_name,
        "file": file_path,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }
    analyses.append(record)
    _write_json(ANALYSES_FILE, analyses)
    logger.info("Análisis agregado: %s (usuario=%s)", record['id'], username)
    return record


def get_user_history(user_id: str, limit: int = 100):
    """Return latest analyses for a user (most recent first)."""
    analyses = _read_json(ANALYSES_FILE)
    user_docs = [a for a in analyses if a.get("user_id") == user_id]
    return sorted(user_docs, key=lambda d: d.get("timestamp", ""), reverse=True)[:limit]
