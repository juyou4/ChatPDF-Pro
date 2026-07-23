import json
import os
import threading
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4


BASE_DIR = os.path.join("data", "config")
PROVIDER_FILE = os.path.join(BASE_DIR, "providers.json")
MODEL_FILE = os.path.join(BASE_DIR, "models.json")

# Dynamic provider/model settings are changed through separate HTTP requests.
# Keep each JSON file coherent even when a UI retries or two requests arrive at
# nearly the same time. ``os.replace`` makes readers see either the old full
# file or the new full file, never a partially written JSON document.
_STORE_LOCK = threading.RLock()


def _ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    with _STORE_LOCK:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _save_json(path: str, data: Dict[str, Any]):
    with _STORE_LOCK:
        _ensure_dir()
        target = Path(path)
        temp_path = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target)
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass


def _without_builtin_provider_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ignore legacy dynamic entries that try to replace a built-in provider.

    Custom providers are intentionally additive. Allowing ``openai`` (or any
    other built-in id) here turns a persisted settings record into a global
    endpoint override for future chats.
    """
    try:
        from models.provider_registry import PROVIDER_CONFIG
        builtin_ids = set(PROVIDER_CONFIG)
    except Exception:
        builtin_ids = set()
    return {
        str(provider_id): config
        for provider_id, config in data.items()
        if str(provider_id) not in builtin_ids and isinstance(config, dict)
    }


def _without_builtin_model_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep dynamic model records from replacing trusted registry metadata."""
    try:
        from models.model_registry import EMBEDDING_MODELS
        builtin_ids = set(EMBEDDING_MODELS)
    except Exception:
        builtin_ids = set()
    return {
        str(model_id): config
        for model_id, config in data.items()
        if str(model_id) not in builtin_ids and isinstance(config, dict)
    }


def load_dynamic_providers() -> Dict[str, Any]:
    return _without_builtin_provider_overrides(_load_json(PROVIDER_FILE))


def save_dynamic_providers(data: Dict[str, Any]):
    _save_json(PROVIDER_FILE, _without_builtin_provider_overrides(dict(data or {})))


def load_dynamic_models() -> Dict[str, Any]:
    return _without_builtin_model_overrides(_load_json(MODEL_FILE))


def save_dynamic_models(data: Dict[str, Any]):
    _save_json(MODEL_FILE, _without_builtin_model_overrides(dict(data or {})))
