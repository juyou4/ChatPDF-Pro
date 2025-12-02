import json
import os
from typing import Dict, Any


BASE_DIR = os.path.join("data", "config")
PROVIDER_FILE = os.path.join(BASE_DIR, "providers.json")
MODEL_FILE = os.path.join(BASE_DIR, "models.json")


def _ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path: str, data: Dict[str, Any]):
    _ensure_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_dynamic_providers() -> Dict[str, Any]:
    return _load_json(PROVIDER_FILE)


def save_dynamic_providers(data: Dict[str, Any]):
    _save_json(PROVIDER_FILE, data)


def load_dynamic_models() -> Dict[str, Any]:
    return _load_json(MODEL_FILE)


def save_dynamic_models(data: Dict[str, Any]):
    _save_json(MODEL_FILE, data)
