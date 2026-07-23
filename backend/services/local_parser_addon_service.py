"""Managed on-demand installer for the optional local PDF parser.

MinerU is the default document route.  The local route remains available, but
its heavier OCR/layout dependencies are installed only after an explicit user
action.  This module deliberately owns a fixed requirements file instead of
accepting package names or paths from HTTP callers.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_mode import runtime
from services.layout_service import download_yolo_model, get_yolo_model_status

logger = logging.getLogger(__name__)

ADDON_ID = "local_parser"
ADDON_VERSION = "1"
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REQUIREMENTS_PATH = _BACKEND_ROOT / "requirements-local-parser.txt"
_STATE_FILENAME = "local_parser_addon.json"
_INSTALL_LOCK = threading.Lock()
_INSTALL_JOB: dict[str, Any] = {
    "status": "idle",
    "stage": "",
    "message": "",
    "error": "",
}

_PYTHON_COMPONENTS = (
    ("pdf2image", "pdf2image", "PDF 页面渲染"),
    ("pytesseract", "pytesseract", "本地 OCR"),
    ("doclayout_yolo", "doclayout_yolo", "版面识别"),
    ("torch", "torch", "版面模型运行时"),
    ("cv2", "opencv-python", "图像处理"),
    ("opendataloader_pdf", "opendataloader-pdf", "文本清理"),
    ("huggingface_hub", "huggingface_hub", "模型下载"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path() -> Path:
    return Path(runtime.data_dir) / "config" / _STATE_FILENAME


def _requirements_hash() -> str:
    try:
        return hashlib.sha256(_REQUIREMENTS_PATH.read_bytes()).hexdigest()
    except OSError:
        return ""


def _read_state() -> dict[str, Any]:
    path = _state_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_state(**updates: Any) -> None:
    path = _state_path()
    payload = _read_state()
    payload.update(updates)
    payload["addon"] = ADDON_ID
    payload["version"] = ADDON_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _python_component_status() -> list[dict[str, Any]]:
    return [
        {
            "id": module,
            "package": package,
            "label": label,
            "available": importlib.util.find_spec(module) is not None,
        }
        for module, package, label in _PYTHON_COMPONENTS
    ]


def _system_tool_status() -> list[dict[str, Any]]:
    return [
        {
            "id": "tesseract",
            "label": "Tesseract",
            "available": bool(shutil.which("tesseract")),
            "optional_for": "扫描版 PDF OCR",
        },
        {
            "id": "poppler",
            "label": "Poppler",
            "available": bool(shutil.which("pdftoppm")),
            "optional_for": "扫描版 PDF OCR",
        },
        {
            "id": "java",
            "label": "Java 11+",
            "available": bool(shutil.which("java")),
            "optional_for": "OpenDataLoader 文本清理",
        },
    ]


def _installer_supported(python_dependencies_ready: bool) -> tuple[bool, str]:
    if runtime.is_remote_host:
        return False, "远程服务模式不允许通过网页修改服务器运行环境"
    if getattr(sys, "frozen", False) and not python_dependencies_ready:
        return False, "当前桌面安装包未包含本地解析运行时，请安装包含本地解析组件的版本"
    if not _REQUIREMENTS_PATH.is_file():
        return False, "本地解析组件清单缺失"
    return True, ""


def _job_snapshot() -> dict[str, Any]:
    with _INSTALL_LOCK:
        return dict(_INSTALL_JOB)


def _set_job(**updates: Any) -> None:
    with _INSTALL_LOCK:
        _INSTALL_JOB.update(updates)


def get_local_parser_addon_status() -> dict[str, Any]:
    """Return installability and capability state without changing runtime state."""
    python_components = _python_component_status()
    python_ready = all(item["available"] for item in python_components)
    yolo_status = get_yolo_model_status()
    model_ready = bool(yolo_status.get("available"))
    system_tools = _system_tool_status()
    system_missing = [item["label"] for item in system_tools if not item["available"]]
    installer_supported, installer_message = _installer_supported(python_ready)
    job = _job_snapshot()
    state_record = _read_state()
    requirements_hash = _requirements_hash()

    if job.get("status") in {"queued", "installing"}:
        state = "installing"
    elif python_ready and model_ready:
        state = "ready"
    elif job.get("status") == "failed":
        state = "failed"
    elif python_ready:
        state = "model_required"
    else:
        state = "missing"

    warnings = []
    if system_missing:
        warnings.append("扫描 OCR 和 ODL 清理会按当前系统可用工具降级")

    return {
        "addon": ADDON_ID,
        "label": "本地解析组件",
        "ready": state == "ready",
        "state": state,
        "requirements_version": ADDON_VERSION,
        "requirements_hash": requirements_hash,
        "installed_requirements_hash": str(state_record.get("requirements_hash") or ""),
        "update_available": bool(
            requirements_hash
            and state_record.get("requirements_hash")
            and state_record.get("requirements_hash") != requirements_hash
        ),
        "installer": {
            "supported": installer_supported,
            "message": installer_message,
        },
        "installation": job,
        "python_components": python_components,
        "model": yolo_status,
        "system_tools": system_tools,
        "warnings": warnings,
    }


def _install_worker() -> None:
    try:
        components = _python_component_status()
        if not all(item["available"] for item in components):
            _set_job(status="installing", stage="installing_packages", message="正在安装本地解析运行时", error="", started_at=_utc_now())
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--upgrade-strategy",
                    "only-if-needed",
                    "-r",
                    str(_REQUIREMENTS_PATH),
                ],
                cwd=str(_BACKEND_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20 * 60,
                check=False,
            )
            if completed.returncode != 0:
                output = (completed.stderr or completed.stdout or "安装命令失败").strip()
                raise RuntimeError(output[-1800:])
            importlib.invalidate_caches()

        _set_job(status="installing", stage="downloading_model", message="正在下载 DocLayout-YOLO 权重", error="")
        if not get_yolo_model_status().get("available"):
            download_yolo_model()

        # Do not leave the job in "installing" while asking the public status
        # function to verify readiness: that state intentionally reports not-ready.
        _set_job(status="verifying", stage="verifying", message="正在校验本地解析组件", error="")
        final_status = get_local_parser_addon_status()
        if not final_status.get("ready"):
            raise RuntimeError("本地解析组件安装完成，但版面模型未通过可用性检查")

        _write_state(requirements_hash=_requirements_hash(), completed_at=_utc_now())
        _set_job(status="ready", stage="ready", message="本地解析组件已就绪", error="", completed_at=_utc_now())
    except subprocess.TimeoutExpired:
        message = "本地解析组件安装超时，请检查网络后重试"
        logger.warning("[LocalParserAddon] %s", message)
        _set_job(status="failed", stage="failed", message="", error=message, completed_at=_utc_now())
    except Exception as exc:
        logger.exception("[LocalParserAddon] Installation failed")
        _set_job(status="failed", stage="failed", message="", error=str(exc)[-1800:], completed_at=_utc_now())


def start_local_parser_addon_install() -> dict[str, Any]:
    """Start the managed install job once, then return a status snapshot."""
    current = get_local_parser_addon_status()
    if current.get("ready"):
        return current

    supported = bool((current.get("installer") or {}).get("supported"))
    if not supported:
        raise RuntimeError(str((current.get("installer") or {}).get("message") or "当前环境不支持安装本地解析组件"))

    with _INSTALL_LOCK:
        already_running = _INSTALL_JOB.get("status") in {"queued", "installing", "verifying"}
        if not already_running:
            _INSTALL_JOB.update({
                "status": "queued",
                "stage": "queued",
                "message": "正在准备本地解析组件",
                "error": "",
                "started_at": _utc_now(),
            })

    if already_running:
        return get_local_parser_addon_status()

    worker = threading.Thread(
        target=_install_worker,
        name="chatpdf-local-parser-addon",
        daemon=True,
    )
    worker.start()
    return get_local_parser_addon_status()
