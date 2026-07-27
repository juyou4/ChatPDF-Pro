"""
OCR Service for PDF text extraction
Supports both local Tesseract OCR and cloud OCR APIs
"""
import io
import ipaddress
import os
import re
import shutil
import socket
import threading
import time
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Dict
from pathlib import Path
from urllib.parse import urlparse

from runtime_mode import runtime
from services.document_parse_adapter import DocumentParseAdapter, MinerUDocumentParseAdapter
from services.mineru_progress import extract_remote_mineru_progress


# ============================================================
# 数据模型与抽象基类（适配器模式）
# ============================================================

@dataclass
class PageOCRResult:
    """单页 OCR 结果"""
    page_number: int              # 页码（从 1 开始）
    text: str                     # OCR 提取的文本
    success: bool                 # 是否成功
    error: Optional[str] = None   # 错误信息（失败时）


@dataclass
class OCRResult:
    """批量 OCR 结果"""
    pages: List[PageOCRResult]                                # 各页结果
    failed_pages: List[int] = field(default_factory=list)     # 失败页码列表
    errors: Dict[int, str] = field(default_factory=dict)      # 页码 -> 错误信息
    backend: str = ""                                         # 使用的后端名称
    layout_figures: List[Dict] = field(default_factory=list)  # MinerU 版面分析提取的 figure 数据


def select_ocr_target_pages(
    enable_ocr: Optional[str],
    total_pages: int,
    pages_needing_ocr: List[int],
) -> List[int]:
    """根据 OCR 模式决定需要执行 OCR 的 0-based 页码列表。

    保持历史兼容：未知模式按 auto 处理，仅对质量评估标记的页面执行 OCR。
    """
    mode = (enable_ocr or "auto").strip().lower()
    if mode == "never":
        return []
    if mode == "always":
        return list(range(max(total_pages, 0)))
    return list(pages_needing_ocr or [])


def apply_ocr_result_to_pages(
    result: dict,
    pages: List[dict],
    ocr_result: OCRResult,
    ocr_target_pages: List[int],
    rebuild_text: Optional[Callable[..., str]] = None,
    *,
    is_cjk: bool = False,
    min_replacement_ratio: float = 0.8,
) -> dict:
    """将 OCR 结果合并回 PDF 提取结果。

    page_numbers 在适配器接口中使用 0-based，PageOCRResult.page_number 使用
    1-based；这里统一转换，避免路由层重复页码映射和部分失败处理。
    """
    rebuild = rebuild_text or (lambda text, **_: text)
    ocr_page_map = {
        page_ocr.page_number - 1: page_ocr.text
        for page_ocr in ocr_result.pages
        if page_ocr.success
    }
    page_lookup = {}
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        try:
            display_page = int(page.get("page") or (index + 1))
        except (TypeError, ValueError):
            display_page = index + 1
        page_lookup[display_page] = page

    def _record_page_attempt(
        display_page: int,
        *,
        success: bool,
        applied: bool = False,
        error: Optional[str] = None,
    ) -> None:
        page = page_lookup.get(display_page)
        if page is None:
            return
        attempts = page.setdefault("ocr_attempts", [])
        attempt = {
            "backend": ocr_result.backend,
            "success": bool(success),
            "applied": bool(applied),
        }
        if success and not applied:
            attempt["not_applied_reason"] = "below_replacement_threshold"
        if error:
            attempt["error"] = str(error)
            page["ocr_last_error"] = str(error)
        attempts.append(attempt)

    merged_text_parts = []
    used_before = bool(result.get("ocr_used"))
    used_this_attempt = False
    applied_pages: List[int] = []
    for index, page in enumerate(pages):
        if index in ocr_page_map:
            ocr_content = ocr_page_map[index]
            original_content = page.get("content", "")

            if len(ocr_content) > len(original_content) * min_replacement_ratio:
                page["content"] = rebuild(ocr_content, is_cjk=is_cjk)
                page["source"] = "ocr"
                page["ocr_backend"] = ocr_result.backend
                result["ocr_used"] = True
                used_this_attempt = True
                applied_pages.append(index + 1)

        merged_text_parts.append(page.get("content", ""))

    if result.get("ocr_used"):
        result["full_text"] = "\n\n".join(merged_text_parts)
        if used_this_attempt and not result.get("ocr_backend"):
            result["ocr_backend"] = ocr_result.backend
        if used_this_attempt:
            backends = list(result.get("ocr_backends") or [])
            if ocr_result.backend and ocr_result.backend not in backends:
                backends.append(ocr_result.backend)
            result["ocr_backends"] = backends
            previous_pages = {
                int(page)
                for page in (result.get("ocr_pages") or [])
                if isinstance(page, int) or str(page).isdigit()
            }
            previous_pages.update(applied_pages)
            result["ocr_pages"] = sorted(previous_pages)

    failed_pages = sorted({
        int(page)
        for page in (ocr_result.failed_pages or [])
        if isinstance(page, int) or str(page).isdigit()
    })
    result["ocr_failed_pages"] = failed_pages
    previous_targets = {
        int(page)
        for page in (result.get("ocr_target_pages") or [])
        if isinstance(page, int) or str(page).isdigit()
    }
    previous_targets.update(
        int(page) + 1
        for page in (ocr_target_pages or [])
        if (isinstance(page, int) or str(page).isdigit()) and int(page) >= 0
    )
    result["ocr_target_pages"] = sorted(previous_targets)

    recorded_pages = set()
    execution_successful_pages = {
        int(page)
        for page in (result.get("ocr_execution_successful_pages") or [])
        if isinstance(page, int) or str(page).isdigit()
    }
    applied_page_set = {
        int(page)
        for page in (
            result.get("ocr_applied_pages")
            or result.get("ocr_pages")
            or result.get("ocr_successful_pages")
            or []
        )
        if isinstance(page, int) or str(page).isdigit()
    }
    applied_page_set.update(applied_pages)
    applied_this_attempt = set(applied_pages)
    for page_ocr in ocr_result.pages or []:
        try:
            display_page = int(page_ocr.page_number)
        except (TypeError, ValueError):
            continue
        recorded_pages.add(display_page)
        if page_ocr.success:
            execution_successful_pages.add(display_page)
        _record_page_attempt(
            display_page,
            success=bool(page_ocr.success),
            applied=display_page in applied_this_attempt,
            error=getattr(page_ocr, "error", None),
        )
    result["ocr_execution_successful_pages"] = sorted(execution_successful_pages)
    result["ocr_applied_pages"] = sorted(applied_page_set)
    # Compatibility field: "successful" means the OCR text became the
    # document's consumable text, not merely that the provider returned 200.
    result["ocr_successful_pages"] = sorted(applied_page_set)
    result["ocr_unapplied_pages"] = sorted(execution_successful_pages - applied_page_set)
    for display_page in failed_pages:
        if display_page in recorded_pages:
            continue
        _record_page_attempt(
            int(display_page),
            success=False,
            applied=False,
            error=(ocr_result.errors or {}).get(int(display_page)),
        )

    if failed_pages:
        failed_info = ", ".join(str(page) for page in failed_pages)
        result["ocr_warning"] = f"部分页面 OCR 失败（页码: {failed_info}）"

    if ocr_target_pages and len(failed_pages) == len(ocr_target_pages) and not used_before and not used_this_attempt:
        result["ocr_warning"] = "所有需要 OCR 的页面均处理失败，已保留原始提取文本"
        result["ocr_used"] = False

    if ocr_result.layout_figures:
        result["ocr_result"] = {
            "figures": ocr_result.layout_figures,
            "backend": ocr_result.backend,
        }

    return result


def _allow_private_ocr_urls() -> bool:
    value = os.environ.get("CHATPDF_ALLOW_PRIVATE_OCR_URLS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_external_ocr_host(host: str, port: int | None, *, service_name: str) -> None:
    """Resolve a hostname before connecting and reject non-public addresses."""
    try:
        records = socket.getaddrinfo(
            host,
            port or 443,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"{service_name} URL 主机名无法解析") from exc

    resolved_ips: set[Any] = set()
    for _family, _socktype, _proto, _canonname, sockaddr in records:
        try:
            resolved_ips.add(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    if not resolved_ips:
        raise ValueError(f"{service_name} URL 主机名未解析到有效 IP 地址")
    if any(not address.is_global for address in resolved_ips):
        raise ValueError(f"{service_name} URL 不允许解析到私网、保留或本机地址")


def validate_external_ocr_service_url(
    url: str,
    *,
    service_name: str = "在线 OCR 服务",
    allow_private: Optional[bool] = None,
) -> str:
    """校验在线 OCR 上游地址，降低误连内网/本机的风险。

    默认仅允许 HTTPS 公网地址。确需本地 Worker 的桌面或开发环境，可设置
    CHATPDF_ALLOW_PRIVATE_OCR_URLS=true 放行 HTTP 和私网地址。
    """
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        raise ValueError(f"{service_name} URL 不能为空")

    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{service_name} URL 格式无效")
    if parsed.username or parsed.password:
        raise ValueError(f"{service_name} URL 不允许包含用户名或密码")

    private_allowed = _allow_private_ocr_urls() if allow_private is None else allow_private
    if not private_allowed and parsed.scheme != "https":
        raise ValueError(f"{service_name} URL 必须使用 HTTPS")

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError(f"{service_name} URL 缺少主机名")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{service_name} URL 端口无效") from exc

    local_hosts = {"localhost", "localhost.localdomain"}
    if not private_allowed and (host in local_hosts or host.endswith(".local")):
        raise ValueError(f"{service_name} URL 不允许指向本机或局域网主机")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None and not private_allowed:
        if not ip.is_global:
            raise ValueError(f"{service_name} URL 不允许指向私网、保留或本机地址")
    elif not private_allowed:
        try:
            _resolve_external_ocr_host(host, port, service_name=service_name)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{service_name} URL 主机名校验失败") from exc

    return cleaned


def _read_positive_env_limit(name: str, default: int, *, maximum: int) -> int:
    try:
        return max(1, min(int(os.environ.get(name, str(default))), maximum))
    except (TypeError, ValueError):
        return default


def _same_ocr_service_origin(left: str, right: str) -> bool:
    try:
        left_parsed = urlparse(left)
        right_parsed = urlparse(right)
        left_port = left_parsed.port
        right_port = right_parsed.port
        if (left_parsed.scheme, left_port) in {("https", 443), ("http", 80)}:
            left_port = None
        if (right_parsed.scheme, right_port) in {("https", 443), ("http", 80)}:
            right_port = None
        return (
            left_parsed.scheme.lower(),
            (left_parsed.hostname or "").lower().rstrip("."),
            left_port,
        ) == (
            right_parsed.scheme.lower(),
            (right_parsed.hostname or "").lower().rstrip("."),
            right_port,
        )
    except ValueError:
        return False


_MAX_MINERU_ZIP_BYTES = _read_positive_env_limit(
    "CHATPDF_MAX_MINERU_ZIP_BYTES", 100 * 1024 * 1024, maximum=512 * 1024 * 1024
)
_MAX_MINERU_ZIP_ENTRIES = _read_positive_env_limit(
    "CHATPDF_MAX_MINERU_ZIP_ENTRIES", 1024, maximum=10_000
)
_MAX_MINERU_ZIP_ENTRY_BYTES = _read_positive_env_limit(
    "CHATPDF_MAX_MINERU_ZIP_ENTRY_BYTES", 64 * 1024 * 1024, maximum=256 * 1024 * 1024
)
_MAX_MINERU_ZIP_EXPANDED_BYTES = _read_positive_env_limit(
    "CHATPDF_MAX_MINERU_ZIP_EXPANDED_BYTES", 256 * 1024 * 1024, maximum=1024 * 1024 * 1024
)
_MINERU_DIRECT_ZIP_DOWNLOAD_ATTEMPTS = _read_positive_env_limit(
    "CHATPDF_MINERU_ZIP_DOWNLOAD_ATTEMPTS", 5, maximum=8
)


def create_mineru_direct_http_client(
    *,
    timeout_seconds: float = 300.0,
    connect_timeout_seconds: float = 30.0,
    disable_keepalive: bool = False,
):
    """Create an official MinerU client that never inherits system proxies."""
    import httpx

    options: dict[str, Any] = {
        "timeout": httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds),
        "trust_env": False,
    }
    if disable_keepalive:
        options["limits"] = httpx.Limits(
            max_connections=1,
            max_keepalive_connections=0,
        )
    return httpx.Client(**options)


def _download_limited_zip(client, zip_url: str, *, headers: Optional[dict] = None, service_name: str) -> bytes:
    """Download a verified OCR archive without buffering an unbounded response."""
    try:
        safe_url = validate_external_ocr_service_url(zip_url, service_name=service_name)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    with client.stream("GET", safe_url, headers=headers or {}, follow_redirects=False) as response:
        if not response.is_success:
            raise RuntimeError(f"{service_name} 下载失败 (HTTP {response.status_code})")
        content_length = response.headers.get("content-length")
        try:
            if content_length and int(content_length) > _MAX_MINERU_ZIP_BYTES:
                raise RuntimeError(f"{service_name} ZIP 超过大小上限")
        except ValueError:
            pass

        chunks: list[bytes] = []
        total_bytes = 0
        for chunk in response.iter_bytes():
            total_bytes += len(chunk)
            if total_bytes > _MAX_MINERU_ZIP_BYTES:
                raise RuntimeError(f"{service_name} ZIP 超过大小上限")
            chunks.append(chunk)
    return b"".join(chunks)


def _validate_mineru_zip(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Reject archive bombs before reading any member payload."""
    infos = zf.infolist()
    if len(infos) > _MAX_MINERU_ZIP_ENTRIES:
        raise RuntimeError("MinerU ZIP 条目数量超过上限")
    expanded_bytes = 0
    for info in infos:
        if info.file_size > _MAX_MINERU_ZIP_ENTRY_BYTES:
            raise RuntimeError(f"MinerU ZIP 条目过大: {info.filename}")
        expanded_bytes += info.file_size
        if expanded_bytes > _MAX_MINERU_ZIP_EXPANDED_BYTES:
            raise RuntimeError("MinerU ZIP 解压后总大小超过上限")
    return infos


class BaseOCRAdapter(ABC):
    """OCR 适配器抽象基类，定义所有 OCR 后端的统一接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器名称标识"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检测该 OCR 后端是否可用"""
        ...

    @abstractmethod
    def ocr_image(self, image) -> str:
        """对单张图片执行 OCR，返回文本"""
        ...

    @abstractmethod
    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        对指定页码执行 OCR

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        ...


# MinerU is a document parser and must never enter the page replacement OCR
# contract. Doc2X is no longer registered; its class remains only to read old
# local configuration during migration.
_DOCUMENT_PARSE_PROVIDER_NAMES = frozenset({"mineru"})

import json
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============================================================
# 在线 OCR 配置管理函数
# ============================================================

# Server mode retains the historical project ``data`` directory. Packaged
# desktop mode must keep credentials in the per-user application data folder
# instead of beside a project checkout or installation.
_LEGACY_ONLINE_OCR_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "online_ocr_config.json"
)
_ONLINE_OCR_CONFIG_PATH = Path(runtime.data_dir) / "online_ocr_config.json"
_OCR_PROVIDER_USAGE_PATH = _ONLINE_OCR_CONFIG_PATH.parent / "ocr_provider_usage.json"
_OCR_PROVIDER_USAGE_THREAD_LOCK = threading.RLock()
_ONLINE_OCR_CONFIG_THREAD_LOCK = threading.RLock()


def _restrict_online_ocr_config_permissions(path: Path) -> None:
    """Apply the strongest portable file-mode restriction available.

    Windows user-data directories inherit the owning user's ACL from Electron.
    On POSIX, explicitly remove group/world access because a freshly created
    file can otherwise inherit a permissive process umask.
    """
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.warning("无法收紧在线 OCR 配置文件权限")


def _migrate_legacy_online_ocr_config_if_needed() -> None:
    """Move a pre-user-data desktop OCR config once without leaving a copy."""
    target = _ONLINE_OCR_CONFIG_PATH
    legacy = _LEGACY_ONLINE_OCR_CONFIG_PATH
    if (
        not runtime.is_desktop
        or target == legacy
        or target.exists()
        or not legacy.exists()
    ):
        return

    temp_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temp_path.write_bytes(legacy.read_bytes())
        _restrict_online_ocr_config_permissions(temp_path)
        os.replace(temp_path, target)
        _restrict_online_ocr_config_permissions(target)
        legacy.unlink()
        logger.info("已将在线 OCR 配置迁移到桌面用户数据目录")
    except OSError as exc:
        logger.warning("在线 OCR 配置迁移失败，将继续使用现有配置: %s", type(exc).__name__)
        if temp_path:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _online_ocr_config_read_path() -> Path:
    """Return the active config, retaining a legacy fallback during migration."""
    _migrate_legacy_online_ocr_config_if_needed()
    if _ONLINE_OCR_CONFIG_PATH.exists():
        return _ONLINE_OCR_CONFIG_PATH
    if runtime.is_desktop and _LEGACY_ONLINE_OCR_CONFIG_PATH.exists():
        return _LEGACY_ONLINE_OCR_CONFIG_PATH
    return _ONLINE_OCR_CONFIG_PATH


@contextmanager
def _ocr_provider_usage_lock():
    """Serialize JSON read-modify-write across desktop processes and servers."""
    lock_path = Path(f"{_OCR_PROVIDER_USAGE_PATH}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _OCR_PROVIDER_USAGE_THREAD_LOCK:
        with open(lock_path, "a+b") as lock_file:
            locked = False
            try:
                lock_file.seek(0)
                lock_file.write(b"\0")
                lock_file.flush()
                if os.name == "nt":
                    import msvcrt

                    deadline = time.monotonic() + 10.0
                    while True:
                        try:
                            lock_file.seek(0)
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                            locked = True
                            break
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("OCR provider usage lock timed out")
                            time.sleep(0.02)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    locked = True
                yield
            finally:
                if locked:
                    try:
                        if os.name == "nt":
                            import msvcrt

                            lock_file.seek(0)
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass


def _usage_entry(entry: Any) -> dict:
    entry = dict(entry) if isinstance(entry, dict) else {}
    legacy_count = _usage_int(entry.get("count"))
    # Historic data records only an undifferentiated count.  Conservatively
    # treat it as prior success so an upgrade cannot falsely justify deletion.
    entry.setdefault("attempt_count", legacy_count)
    entry.setdefault("success_count", legacy_count)
    entry.setdefault("failure_count", 0)
    entry.setdefault("fallback_success_count", 0)
    entry.setdefault("first_seen_at", entry.get("last_used_at") or "")
    entry.setdefault("last_used_at", "")
    entry.setdefault("last_operation", "")
    entry.setdefault("operations", {})
    entry["operations"] = entry["operations"] if isinstance(entry["operations"], dict) else {}
    return entry


def _usage_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _record_usage_outcome(entry: dict, *, outcome: str, fallback: bool) -> None:
    entry["attempt_count"] = _usage_int(entry.get("attempt_count")) + 1
    if outcome == "success":
        entry["success_count"] = _usage_int(entry.get("success_count")) + 1
        if fallback:
            entry["fallback_success_count"] = _usage_int(entry.get("fallback_success_count")) + 1
    else:
        entry["failure_count"] = _usage_int(entry.get("failure_count")) + 1


def record_ocr_provider_use(
    provider: str,
    *,
    outcome: str = "success",
    operation: str = "page_ocr",
    fallback: bool = False,
) -> None:
    """Persist actual provider outcomes for rollout and sunset decisions.

    ``count`` remains a compatibility alias for successful requests.  New
    fields separate attempts, failures, fallback successes, and operation type.
    """
    name = "".join(char for char in str(provider or "") if char.isalnum() or char in {"-", "_"})
    normalized_outcome = str(outcome or "").strip().lower()
    normalized_operation = "".join(char for char in str(operation or "") if char.isalnum() or char in {"-", "_"})
    if not name or normalized_outcome not in {"success", "failure"} or not normalized_operation:
        return
    try:
        with _ocr_provider_usage_lock():
            data: dict = {}
            if _OCR_PROVIDER_USAGE_PATH.exists():
                loaded = json.loads(_OCR_PROVIDER_USAGE_PATH.read_text(encoding="utf-8"))
                data = loaded if isinstance(loaded, dict) else {}
            now = datetime.now(timezone.utc).isoformat()
            entry = _usage_entry(data.get(name))
            _record_usage_outcome(entry, outcome=normalized_outcome, fallback=bool(fallback))
            operation_entry = _usage_entry(entry["operations"].get(normalized_operation))
            _record_usage_outcome(operation_entry, outcome=normalized_outcome, fallback=bool(fallback))
            operation_entry["last_used_at"] = now
            operation_entry["first_seen_at"] = operation_entry.get("first_seen_at") or now
            entry["operations"][normalized_operation] = operation_entry
            entry["count"] = _usage_int(entry.get("success_count"))
            entry["last_used_at"] = now
            entry["first_seen_at"] = entry.get("first_seen_at") or now
            entry["last_operation"] = normalized_operation
            data[name] = entry
            temp_path = _OCR_PROVIDER_USAGE_PATH.with_name(
                f".{_OCR_PROVIDER_USAGE_PATH.name}.{uuid.uuid4().hex}.tmp"
            )
            temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp_path, _OCR_PROVIDER_USAGE_PATH)
    except Exception as exc:
        logger.warning("记录 OCR provider 使用情况失败: %s", exc)


def get_ocr_provider_usage(provider: str) -> dict:
    try:
        with _ocr_provider_usage_lock():
            data = json.loads(_OCR_PROVIDER_USAGE_PATH.read_text(encoding="utf-8")) if _OCR_PROVIDER_USAGE_PATH.exists() else {}
            entry = data.get(provider) if isinstance(data, dict) else None
            normalized = _usage_entry(entry)
            normalized["count"] = _usage_int(normalized.get("success_count"))
            return normalized
    except Exception:
        return {
            "count": 0,
            "attempt_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "fallback_success_count": 0,
            "first_seen_at": "",
            "last_used_at": "",
            "last_operation": "",
            "operations": {},
        }

# 各在线 OCR 提供商的环境变量名映射
# 注意：token_mode、enable_ocr、enable_formula、enable_table 等字段
# 不通过环境变量配置，而是从配置文件加载
_ENV_VAR_MAP = {
    "mistral": {
        "api_key": "CHATPDF_MISTRAL_OCR_API_KEY",
        "base_url": "CHATPDF_MISTRAL_OCR_BASE_URL",
    },
    "mineru": {
        "worker_url": "CHATPDF_MINERU_WORKER_URL",
        "auth_key": "CHATPDF_MINERU_AUTH_KEY",
        "token": "CHATPDF_MINERU_TOKEN",
        "base_url": "CHATPDF_MINERU_BASE_URL",
    },
    "doc2x": {
        "worker_url": "CHATPDF_DOC2X_WORKER_URL",
        "auth_key": "CHATPDF_DOC2X_AUTH_KEY",
        "token": "CHATPDF_DOC2X_TOKEN",
    },
}

# 各在线 OCR 提供商的默认配置
_DEFAULT_CONFIG = {
    "mistral": {
        "api_key": "",
        "base_url": "https://api.mistral.ai",
    },
    "mineru": {
        "access_mode": "worker",
        "base_url": "https://mineru.net/api/v4",
        "worker_url": "",
        "auth_key": "",
        "token_mode": "frontend",
        "token": "",
        "enable_ocr": False,
        "enable_formula": True,
        "enable_table": True,
        "model_version": "vlm",
    },
    "doc2x": {
        "worker_url": "",
        "auth_key": "",
        "token_mode": "frontend",
        "token": "",
    },
}


_MINERU_MODEL_VERSIONS = {"vlm", "pipeline"}


def normalize_mineru_model_version(value: Any) -> str:
    """Return one supported MinerU model version for config and job records."""
    model_version = str(value or "vlm").strip().lower()
    return model_version if model_version in _MINERU_MODEL_VERSIONS else "vlm"


def _load_online_ocr_config(provider: str) -> dict:
    """
    加载在线 OCR 配置，优先级：环境变量 > 配置文件 > 默认值

    对于在 _ENV_VAR_MAP 中定义的字段，按上述优先级加载。
    对于不在 _ENV_VAR_MAP 中的字段（如 token_mode、enable_ocr、enable_formula、
    enable_table），仅从配置文件和默认值加载（不通过环境变量配置）。

    参数:
        provider: 服务提供商名称（如 "mistral"、"mineru"、"doc2x"）
    返回:
        配置字典，包含该提供商的所有配置字段
    """
    # 获取默认配置
    defaults = _DEFAULT_CONFIG.get(provider, {"api_key": "", "base_url": ""})
    result = dict(defaults)

    # 获取该提供商的环境变量映射（用于区分哪些字段可通过环境变量配置）
    env_map = _ENV_VAR_MAP.get(provider, {})

    # 第二优先级：从配置文件加载
    with _ONLINE_OCR_CONFIG_THREAD_LOCK:
        config_path = _online_ocr_config_read_path()
        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    all_config = json.load(f)
                if provider in all_config and isinstance(all_config[provider], dict):
                    provider_config = all_config[provider]
                    for key in result:
                        if key in provider_config:
                            value = provider_config[key]
                            # 对于字符串类型字段，仅覆盖非空值
                            # 对于布尔类型字段，直接覆盖
                            if isinstance(value, bool):
                                result[key] = value
                            elif value:
                                result[key] = value
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.error(f"读取在线 OCR 配置文件失败: {e}")

    # 第一优先级：从环境变量加载（仅覆盖 _ENV_VAR_MAP 中定义的字段）
    for field_name, env_var_name in env_map.items():
        env_value = os.environ.get(env_var_name, "")
        if env_value:
            result[field_name] = env_value

    if provider == "mineru":
        result["model_version"] = normalize_mineru_model_version(
            result.get("model_version")
        )

    return result


def _save_online_ocr_config(provider: str, config: dict) -> None:
    """
    保存在线 OCR 配置到本地文件

    参数:
        provider: 服务提供商名称
        config: 配置字典，包含 api_key 和/或 base_url
    """
    try:
        config = dict(config or {})
        if provider == "mineru":
            config["model_version"] = normalize_mineru_model_version(
                config.get("model_version")
            )
        with _ONLINE_OCR_CONFIG_THREAD_LOCK:
            config_read_path = _online_ocr_config_read_path()
            # 确保目录存在
            _ONLINE_OCR_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

            # 读取现有配置（如果存在）
            all_config: dict = {}
            if config_read_path.exists():
                try:
                    with open(config_read_path, "r", encoding="utf-8") as f:
                        all_config = json.load(f)
                except (json.JSONDecodeError, IOError):
                    # 配置文件损坏，使用空字典重新开始
                    logger.warning("在线 OCR 配置文件损坏，将重新创建")
                    all_config = {}

            # 更新指定提供商的配置
            if provider not in all_config:
                all_config[provider] = {}
            all_config[provider].update(config)

            # Persist credentials atomically. A crash during a direct write can
            # otherwise leave a partial JSON file that loses an unrelated
            # provider's key on the next save.
            temp_path = _ONLINE_OCR_CONFIG_PATH.with_name(
                f".{_ONLINE_OCR_CONFIG_PATH.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(all_config, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                _restrict_online_ocr_config_permissions(temp_path)
                os.replace(temp_path, _ONLINE_OCR_CONFIG_PATH)
                _restrict_online_ocr_config_permissions(_ONLINE_OCR_CONFIG_PATH)
            except Exception:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

        logger.info(f"在线 OCR 配置已保存: provider={provider}")
    except (IOError, OSError) as e:
        logger.error(f"保存在线 OCR 配置文件失败: {e}")
        raise


def _mask_api_key(api_key: str) -> str:
    """
    脱敏 API Key，仅显示前 4 位和后 4 位

    参数:
        api_key: 原始 API Key
    返回:
        脱敏后的字符串，如 "sk-x...xxxx"
    """
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:4] + "..." + api_key[-4:]

# OCR 可用性标志
TESSERACT_AVAILABLE = False
PDF2IMAGE_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

# 自动检测本地 OCR 工具路径
def _find_local_ocr_tools():
    """查找本地安装的 OCR 工具"""
    base_dir = Path(__file__).resolve().parents[2]  # Chatpdf 根目录
    ocr_dir = base_dir / "ocr_tools"
    
    # Tesseract 路径
    tesseract_paths = [
        ocr_dir / "tesseract" / "tesseract.exe",  # Windows 本地安装
        ocr_dir / "tesseract" / "tesseract",  # Linux/Mac 本地安装
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),  # Windows 默认
        Path("/usr/bin/tesseract"),  # Linux 默认
        Path("/usr/local/bin/tesseract"),  # macOS Homebrew
        Path("/opt/homebrew/bin/tesseract"),  # macOS M1 Homebrew
    ]
    
    for path in tesseract_paths:
        if path.exists():
            return str(path.parent)
    
    return None

# 设置 Tesseract 路径
_tesseract_path = _find_local_ocr_tools()
if _tesseract_path:
    os.environ["PATH"] = _tesseract_path + os.pathsep + os.environ.get("PATH", "")

# Poppler 路径
def _find_poppler():
    """
    查找 Poppler 路径

    增强逻辑：
    - 搜索失败时记录所有已搜索路径到日志
    - 检测 ocr_tools/poppler/ 目录存在但无可执行文件的情况并记录警告
    """
    base_dir = Path(__file__).resolve().parents[2]
    ocr_dir = base_dir / "ocr_tools"

    poppler_paths = [
        ocr_dir / "poppler" / "Library" / "bin",  # Windows 本地安装
        Path("/usr/bin"),  # Linux
        Path("/usr/local/bin"),  # macOS Homebrew
        Path("/opt/homebrew/bin"),  # macOS M1 Homebrew
    ]

    # 检测 ocr_tools/poppler/ 目录是否存在但内部无可执行文件
    poppler_dir = ocr_dir / "poppler"
    if poppler_dir.exists() and poppler_dir.is_dir():
        # 检查该目录及其子目录中是否有 pdftoppm 可执行文件
        has_executable = False
        for sub_path in poppler_paths:
            # 仅检查属于 poppler_dir 子路径的搜索路径
            try:
                sub_path.relative_to(poppler_dir)
            except ValueError:
                continue
            if (sub_path / "pdftoppm.exe").exists() or (sub_path / "pdftoppm").exists():
                has_executable = True
                break

        if not has_executable:
            logger.warning(
                f"Poppler 目录 '{poppler_dir}' 存在但未找到可执行文件（pdftoppm），"
                "该目录可能为空或未正确安装。"
                "请确保 Poppler 已正确解压到该目录中。"
            )

    for path in poppler_paths:
        if (path / "pdftoppm.exe").exists() or (path / "pdftoppm").exists():
            return str(path)

    # 搜索失败，记录所有已搜索路径到日志
    searched_paths_str = "\n  ".join(str(p) for p in poppler_paths)
    logger.warning(
        f"未找到 Poppler 可执行文件（pdftoppm），已搜索以下路径:\n  {searched_paths_str}\n"
        "请安装 Poppler 以启用 PDF 转图像功能。\n"
        "安装指引:\n"
        "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
        "  - macOS: brew install poppler\n"
        "  - Linux: sudo apt-get install poppler-utils"
    )
    return None

_poppler_path = _find_poppler()
if _poppler_path:
    os.environ["PATH"] = _poppler_path + os.pathsep + os.environ.get("PATH", "")

try:
    import pytesseract
    from PIL import Image
    # 设置 Tesseract 命令路径
    if _tesseract_path:
        tesseract_cmd = os.path.join(_tesseract_path, "tesseract.exe" if os.name == "nt" else "tesseract")
        if os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    from pdf2image import convert_from_bytes
    # 设置 Poppler 路径
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# 具体适配器实现
# ============================================================

class TesseractAdapter(BaseOCRAdapter):
    """Tesseract OCR 适配器"""

    def __init__(self, lang: str = "chi_sim+eng"):
        """
        初始化 Tesseract 适配器

        参数:
            lang: Tesseract 语言代码（默认中文+英文）
        """
        self._lang = lang

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "tesseract"

    def is_available(self) -> bool:
        """Verify the Python packages, executable, and requested language data."""
        if not (TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE):
            return False
        command = str(getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract") or "tesseract")
        if not Path(command).is_file() and shutil.which(command) is None:
            return False
        try:
            languages = set(pytesseract.get_languages(config=""))
        except Exception:
            return False
        requested = {language.strip() for language in self._lang.split("+") if language.strip()}
        return requested.issubset(languages)

    def ocr_image(self, image) -> str:
        """
        使用 Tesseract 对单张图片执行 OCR

        参数:
            image: PIL Image 对象
        返回:
            OCR 提取的文本
        """
        if not TESSERACT_AVAILABLE:
            return ""
        return pytesseract.image_to_string(image, lang=self._lang)

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        对指定页码执行 Tesseract OCR

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image 未安装，请运行: pip install pdf2image")

        poppler_path = _find_poppler()
        if poppler_path is None:
            raise RuntimeError(
                "Poppler 未找到，无法将 PDF 转换为图像。\n"
                "请按照以下指引安装 Poppler:\n"
                "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
                "  - macOS: brew install poppler\n"
                "  - Linux: sudo apt-get install poppler-utils\n"
                "详情请参考: https://poppler.freedesktop.org/"
            )

        pages_result: List[PageOCRResult] = []
        failed_pages: List[int] = []
        errors: Dict[int, str] = {}

        for page_num in page_numbers:
            # pdf2image 的 first_page/last_page 从 1 开始，page_numbers 从 0 开始
            pdf2image_page = page_num + 1
            try:
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    first_page=pdf2image_page,
                    last_page=pdf2image_page,
                    poppler_path=poppler_path
                )
                if images:
                    raw_text = self.ocr_image(images[0])
                    text = clean_ocr_text(raw_text)
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text=text,
                        success=True
                    ))
                else:
                    # 未能转换出图像
                    failed_pages.append(pdf2image_page)
                    errors[pdf2image_page] = "PDF 页面转换为图像失败：未生成图像"
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text="",
                        success=False,
                        error="PDF 页面转换为图像失败：未生成图像"
                    ))
            except Exception as e:
                # 单页错误隔离：捕获异常，记录错误，继续处理下一页
                error_msg = f"页面 {pdf2image_page} OCR 失败: {str(e)}"
                logger.error(error_msg)
                failed_pages.append(pdf2image_page)
                errors[pdf2image_page] = str(e)
                pages_result.append(PageOCRResult(
                    page_number=pdf2image_page,
                    text="",
                    success=False,
                    error=str(e)
                ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name
        )


class PaddleOCRAdapter(BaseOCRAdapter):
    """PaddleOCR 适配器"""

    def __init__(self):
        """初始化 PaddleOCR 适配器"""
        self._paddle_ocr = None

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "paddleocr"

    def is_available(self) -> bool:
        """检测 PaddleOCR 和 pdf2image 是否可用"""
        return PADDLEOCR_AVAILABLE and PDF2IMAGE_AVAILABLE

    def _get_paddle_ocr(self):
        """延迟加载 PaddleOCR 实例"""
        if self._paddle_ocr is None and PADDLEOCR_AVAILABLE:
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                show_log=False,
                use_gpu=False
            )
        return self._paddle_ocr

    def ocr_image(self, image) -> str:
        """
        使用 PaddleOCR 对单张图片执行 OCR

        参数:
            image: PIL Image 对象
        返回:
            OCR 提取的文本
        """
        if not PADDLEOCR_AVAILABLE:
            return ""
        ocr = self._get_paddle_ocr()
        if ocr is None:
            return ""
        import numpy as np
        img_array = np.array(image)
        result = ocr.ocr(img_array, cls=True)

        if not result or not result[0]:
            return ""

        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                lines.append(text)
        return '\n'.join(lines)

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        对指定页码执行 PaddleOCR

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image 未安装，请运行: pip install pdf2image")

        poppler_path = _find_poppler()
        if poppler_path is None:
            raise RuntimeError(
                "Poppler 未找到，无法将 PDF 转换为图像。\n"
                "请按照以下指引安装 Poppler:\n"
                "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
                "  - macOS: brew install poppler\n"
                "  - Linux: sudo apt-get install poppler-utils\n"
                "详情请参考: https://poppler.freedesktop.org/"
            )

        pages_result: List[PageOCRResult] = []
        failed_pages: List[int] = []
        errors: Dict[int, str] = {}

        for page_num in page_numbers:
            # pdf2image 的 first_page/last_page 从 1 开始，page_numbers 从 0 开始
            pdf2image_page = page_num + 1
            try:
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    first_page=pdf2image_page,
                    last_page=pdf2image_page,
                    poppler_path=poppler_path
                )
                if images:
                    raw_text = self.ocr_image(images[0])
                    text = clean_ocr_text(raw_text)
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text=text,
                        success=True
                    ))
                else:
                    # 未能转换出图像
                    failed_pages.append(pdf2image_page)
                    errors[pdf2image_page] = "PDF 页面转换为图像失败：未生成图像"
                    pages_result.append(PageOCRResult(
                        page_number=pdf2image_page,
                        text="",
                        success=False,
                        error="PDF 页面转换为图像失败：未生成图像"
                    ))
            except Exception as e:
                # 单页错误隔离：捕获异常，记录错误，继续处理下一页
                error_msg = f"页面 {pdf2image_page} OCR 失败: {str(e)}"
                logger.error(error_msg)
                failed_pages.append(pdf2image_page)
                errors[pdf2image_page] = str(e)
                pages_result.append(PageOCRResult(
                    page_number=pdf2image_page,
                    text="",
                    success=False,
                    error=str(e)
                ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name
        )


class WorkerOCRAdapter(BaseOCRAdapter):
    """Worker 代理模式 OCR 适配器基类，封装 Worker URL、Auth Key、Token Mode 等公共逻辑"""

    def __init__(self, worker_url: str, auth_key: str = "",
                 token: str = "", token_mode: str = "frontend"):
        """
        初始化 Worker 代理适配器

        参数:
            worker_url: Cloudflare Worker 代理服务地址
            auth_key: Worker 端的认证密钥（对应 Worker 的 AUTH_SECRET 环境变量）
            token: 各 OCR 服务的 API Token
            token_mode: Token 传递模式，"frontend"（前端透传）或 "worker"（Worker 配置）
        """
        self._worker_url = ""
        if worker_url:
            try:
                self._worker_url = validate_external_ocr_service_url(
                    worker_url,
                    service_name="OCR Worker",
                )
            except ValueError as exc:
                logger.warning("OCR Worker URL 已被拒绝: %s", exc)
        self._auth_key = auth_key
        self._token = token
        self._token_mode = token_mode  # "frontend" 或 "worker"

    def is_available(self) -> bool:
        """Worker URL 已配置且 Token 可用（worker 模式或 frontend 模式有 token）"""
        if not self._worker_url:
            return False
        if self._token_mode == "frontend":
            return bool(self._token)
        return True  # worker 模式不需要前端提供 token

    def ocr_image(self, image) -> str:
        """在线 OCR 不支持单图模式，返回空字符串"""
        return ""

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        子类需覆盖此方法实现具体的 OCR 处理逻辑

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率
        返回:
            OCRResult 包含各页结果和错误信息
        """
        raise NotImplementedError("子类必须实现 ocr_pages 方法")

    def _build_headers(self) -> dict:
        """
        构建请求头：Auth Key + Token（根据 token_mode）

        返回:
            请求头字典，包含 X-Auth-Key（如有）。
            子类覆盖以添加特定的 Token 头。
        """
        headers = {}
        if self._auth_key:
            headers["X-Auth-Key"] = self._auth_key
        # 子类覆盖以添加特定的 Token 头
        return headers

    def _check_worker_response(self, response, step: str) -> None:
        """
        检查 Worker 响应状态码

        参数:
            response: httpx.Response 对象
            step: 当前步骤描述（用于错误消息）
        异常:
            RuntimeError: 当响应状态码非成功时抛出，401/403 特殊提示认证失败
        """
        if response.is_success:
            return
        status_code = response.status_code
        try:
            error_detail = response.text
        except Exception:
            error_detail = "未知错误"
        if status_code in (401, 403):
            raise RuntimeError(f"{self.name} {step}失败: 认证失败 (HTTP {status_code})")
        if status_code == 404:
            raise RuntimeError(
                f"{self.name} {step}失败: Worker 路由不存在 (HTTP 404)，"
                "请检查 Worker URL 是否填到了 pb-ocr-proxy 部署地址，且已部署 /health、/mineru/upload 等路由"
            )
        raise RuntimeError(f"{self.name} {step}失败 (HTTP {status_code}): {error_detail}")


# ============================================================
# 模块级工具函数
# ============================================================

def _markdown_to_text(markdown: str) -> str:
    """
    将 Markdown 内容转换为纯文本，清理 Markdown 标记

    从 MistralAdapter 中提取为模块级函数，供所有适配器复用。

    参数:
        markdown: Markdown 格式的文本
    返回:
        纯文本字符串
    """
    if not markdown:
        return ""

    text = markdown

    # 移除图片标记 ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # 移除链接标记 [text](url)，保留链接文本
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # 移除标题标记 # ## ### 等
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 移除粗体标记 **text** 或 __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # 移除斜体标记 *text* 或 _text_（注意不要误匹配粗体）
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'\1', text)
    # 移除行内代码标记 `code`
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # 移除代码块标记 ```...```
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除水平分割线 --- 或 *** 或 ___
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 移除无序列表标记 - 或 * 或 +（行首）
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    # 移除有序列表标记 1. 2. 等（行首）
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    # 移除引用标记 >
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    # 移除删除线标记 ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ============================================================
# MinerU OCR 适配器
# ============================================================

class MinerUAdapter(WorkerOCRAdapter):
    """MinerU OCR 适配器，通过 Worker 代理处理 PDF"""

    def __init__(self, worker_url: str, auth_key: str = "",
                 token: str = "", token_mode: str = "frontend",
                 enable_ocr: bool = False, enable_formula: bool = True,
                 enable_table: bool = True, model_version: str = "vlm"):
        """
        初始化 MinerU OCR 适配器

        参数:
            worker_url: Cloudflare Worker 代理服务地址
            auth_key: Worker 端的认证密钥
            token: MinerU API Token
            token_mode: Token 传递模式，"frontend" 或 "worker"
            enable_ocr: 是否启用 OCR 识别
            enable_formula: 是否启用公式识别
            enable_table: 是否启用表格识别
            model_version: MinerU 解析模型版本，默认使用 VLM 后端
        """
        super().__init__(worker_url, auth_key, token, token_mode)
        self._enable_ocr = enable_ocr
        self._enable_formula = enable_formula
        self._enable_table = enable_table
        self._model_version = normalize_mineru_model_version(model_version)

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "mineru"

    def _build_headers(self) -> dict:
        """
        构建请求头：在基类基础上添加 X-MinerU-Key（frontend 模式）

        返回:
            请求头字典
        """
        headers = super()._build_headers()
        if self._token_mode == "frontend" and self._token:
            headers["X-MinerU-Key"] = self._token
        return headers

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """Reject page-level use; MinerU only exposes document parsing."""
        raise RuntimeError("MinerU 不支持逐页 OCR；请使用 MinerU 深度解析获取文档级结构化结果")

    def _legacy_document_ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200,
    ) -> OCRResult:
        """Historical whole-document conversion kept out of the page OCR contract."""
        import httpx

        try:
            payload = self.analyze_pdf(pdf_bytes)
            markdown_content = payload.get("full_md", "")
            layout_figures = payload.get("layout_figures", [])
            logger.info(f"MinerU OCR: 结果下载并解压成功，提取到 {len(layout_figures)} 个 figure 区域")

            # 步骤 4：将 Markdown 转换为纯文本
            text = _markdown_to_text(markdown_content)
            text = clean_ocr_text(text)

            # 步骤 5：构建 OCRResult（MinerU 返回整个文档的文本，按页分配）
            pages_result: List[PageOCRResult] = []
            for page_num in page_numbers:
                display_page = page_num + 1  # 用于显示的页码（从 1 开始）
                pages_result.append(PageOCRResult(
                    page_number=display_page,
                    text=text,
                    success=True,
                ))

            return OCRResult(
                pages=pages_result,
                failed_pages=[],
                errors={},
                backend=self.name,
                layout_figures=layout_figures,
            )

        except httpx.TimeoutException as e:
            logger.error(f"MinerU OCR: 网络连接超时: {e}")
            error_msg = f"MinerU OCR 网络连接超时: {e}"
        except RuntimeError as e:
            logger.error(f"MinerU OCR: {e}")
            error_msg = str(e)
        except Exception as e:
            logger.error(f"MinerU OCR: 未知错误: {e}")
            error_msg = f"MinerU OCR 处理失败: {e}"

        # 错误时返回包含错误信息的 OCRResult
        pages_result = []
        failed_pages = []
        errors = {}
        for page_num in page_numbers:
            display_page = page_num + 1
            failed_pages.append(display_page)
            errors[display_page] = error_msg
            pages_result.append(PageOCRResult(
                page_number=display_page,
                text="",
                success=False,
                error=error_msg,
            ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name,
        )

    def analyze_pdf(self, pdf_bytes: bytes, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None, cancel_event: Any = None) -> Dict[str, Any]:
        """运行 MinerU Worker 并返回 full.md / middle.json / content_list.json 等完整解析载荷。"""
        submission = self.submit_document(pdf_bytes, progress_callback=progress_callback)
        return self.poll_document(
            submission,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def submit_document(
        self,
        pdf_bytes: bytes,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Upload one PDF and return a durable Worker batch identity."""
        import httpx

        with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            if progress_callback:
                progress_callback({"stage": "uploading", "message": "上传 PDF 到 MinerU Worker"})
            logger.info("MinerU OCR: 开始上传 PDF 文件...")
            batch_id = self._upload_pdf(client, pdf_bytes)
            logger.info(f"MinerU OCR: 上传成功，batch_id={batch_id}")
        return {"batch_id": batch_id, "access_mode": "worker"}

    def poll_document(
        self,
        submission: Dict[str, Any],
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
    ) -> Dict[str, Any]:
        """Poll and download one previously submitted Worker batch."""
        batch_id = str((submission or {}).get("batch_id") or "")
        return self.resume_batch(
            batch_id,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            data_id=str((submission or {}).get("data_id") or ""),
        )

    def resume_batch(
        self,
        batch_id: str,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
        data_id: str = "",
    ) -> Dict[str, Any]:
        """Resume a Worker-backed remote job without uploading the PDF again."""
        import httpx

        if not batch_id:
            raise RuntimeError("恢复 MinerU 任务缺少 batch_id")
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            if progress_callback:
                progress_callback({"stage": "resuming", "message": "恢复 MinerU 远端任务", "batch_id": batch_id})
            full_zip_url = self._poll_result(client, batch_id, progress_callback=progress_callback, cancel_event=cancel_event)
            payload = self._download_and_extract_payload(client, full_zip_url)
            payload.update({"batch_id": batch_id, "full_zip_url": full_zip_url, "access_mode": "worker"})
            return payload

    def cancel_batch(self, batch_id: str, *, data_id: str = "") -> dict:
        """Ask the Worker proxy to cancel a remote job.

        The proxy endpoint is deliberately explicit so an old proxy cannot be
        mistaken for a successful cancellation.  Callers retain the returned
        state and still stop local polling when the endpoint is unsupported.
        """
        import httpx

        if not batch_id:
            return {"attempted": False, "state": "not_requested", "reason": "missing_batch_id"}
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = client.post(f"{self._worker_url}/mineru/cancel/{batch_id}", headers=self._build_headers())
            if response.is_success:
                return {"attempted": True, "state": "sent", "status_code": response.status_code}
            return {"attempted": True, "state": "rejected", "status_code": response.status_code, "detail": response.text[:300]}
        except Exception as exc:
            return {"attempted": True, "state": "error", "detail": str(exc)}

    def _upload_pdf(self, client, pdf_bytes: bytes) -> str:
        """
        上传 PDF 到 MinerU Worker 代理

        参数:
            client: httpx.Client 实例
            pdf_bytes: PDF 原始字节
        返回:
            batch_id 批次标识
        异常:
            RuntimeError: 上传失败时抛出
        """
        headers = self._build_headers()
        response = client.post(
            f"{self._worker_url}/mineru/upload",
            headers=headers,
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
            data={
                "is_ocr": str(self._enable_ocr).lower(),
                "enable_formula": str(self._enable_formula).lower(),
                "enable_table": str(self._enable_table).lower(),
                "model_version": self._model_version,
            },
        )
        self._check_worker_response(response, "上传 PDF")
        data = response.json()
        batch_id = data.get("batch_id")
        if not batch_id:
            raise RuntimeError("MinerU 上传成功但未返回 batch_id")
        return batch_id

    def _poll_result(self, client, batch_id: str, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None, cancel_event: Any = None) -> str:
        """
        轮询 MinerU 处理结果

        参数:
            client: httpx.Client 实例
            batch_id: 批次标识
        返回:
            full_zip_url ZIP 文件下载地址
        异常:
            RuntimeError: 处理失败或超时时抛出
        """
        headers = self._build_headers()
        max_attempts = 300
        poll_interval = 3  # 秒

        for attempt in range(max_attempts):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("MinerU 深度解析已取消")
            response = client.get(
                f"{self._worker_url}/mineru/result/{batch_id}",
                headers=headers,
            )
            self._check_worker_response(response, "轮询结果")
            data = response.json()
            state = data.get("state", "")

            if state == "done":
                full_zip_url = data.get("full_zip_url")
                if not full_zip_url:
                    raise RuntimeError("MinerU 处理完成但未返回 full_zip_url")
                return full_zip_url
            elif state == "failed":
                error_msg = data.get("error", "未知错误")
                raise RuntimeError(f"MinerU 处理失败: {error_msg}")

            # 继续等待
            if progress_callback:
                progress = {
                    "stage": "polling",
                    "message": f"等待 MinerU 处理（{attempt + 1}/{max_attempts}）",
                    "batch_id": batch_id,
                    "poll_attempt": attempt + 1,
                    "poll_total": max_attempts,
                    "remote_state": state,
                }
                progress.update(extract_remote_mineru_progress(data))
                progress_callback(progress)
            logger.debug(
                f"MinerU OCR: 轮询中 ({attempt + 1}/{max_attempts})，"
                f"当前状态: {state}"
            )
            time.sleep(poll_interval)

        raise RuntimeError(
            f"MinerU 处理超时: 已轮询 {max_attempts} 次（共 {max_attempts * poll_interval} 秒）"
        )

    def _download_and_extract(self, client, zip_url: str) -> Tuple[str, List[Dict]]:
        """
        下载 ZIP 文件并解压提取 full.md 内容和 middle.json 版面分析数据

        参数:
            client: httpx.Client 实例
            zip_url: ZIP 文件下载地址
        返回:
            (full.md 文本内容, 版面分析提取的 figure 列表)
        异常:
            RuntimeError: 下载失败、解压失败或未找到 full.md 时抛出
        """
        payload = self._download_and_extract_payload(client, zip_url)
        return payload.get("full_md", ""), payload.get("layout_figures", [])

    def _download_and_extract_payload(self, client, zip_url: str) -> Dict[str, Any]:
        """下载 ZIP 并保留 MinerU 的完整 JSON 结果，供深度解析适配器使用。"""
        # Worker may return a pre-signed storage URL. Credentials are only sent
        # when it remains on the configured Worker origin, never to an URL
        # supplied by the remote response on another origin.
        headers = self._build_headers() if _same_ocr_service_origin(zip_url, self._worker_url) else {}
        zip_bytes = _download_limited_zip(
            client,
            zip_url,
            headers=headers,
            service_name="MinerU ZIP",
        )

        try:
            zip_data = io.BytesIO(zip_bytes)
            with zipfile.ZipFile(zip_data, "r") as zf:
                infos = _validate_mineru_zip(zf)
                full_md_path = None
                middle_json_path = None
                content_list_path = None
                for name in (info.filename for info in infos):
                    if name.endswith("full.md"):
                        full_md_path = name
                    elif name.endswith("middle.json"):
                        middle_json_path = name
                    elif name.endswith("content_list.json"):
                        content_list_path = name

                # 必需项以前恰好反了：缺 full.md 直接 raise，而缺 content_list.json /
                # middle.json 只记一条 info 就继续。真正喂数据的是后两者——正文、块
                # 结构、表格与坐标全部来自它们；full.md 的唯一活消费点是文本覆盖率
                # 见证。一份没有结构化数据的产物本来是不该通过的。
                if content_list_path is None and middle_json_path is None:
                    raise RuntimeError(
                        "MinerU ZIP 中未找到 content_list.json 或 middle.json，"
                        f"ZIP 包含: {[info.filename for info in infos]}"
                    )

                if full_md_path is None:
                    logger.warning(
                        "MinerU ZIP 中未找到 full.md，文本覆盖率见证将不可用。"
                        f"ZIP 包含: {[info.filename for info in infos]}"
                    )
                full_md = zf.read(full_md_path).decode("utf-8") if full_md_path else ""
                middle_json = None
                content_list_json = None
                if middle_json_path:
                    middle_json = json.loads(zf.read(middle_json_path).decode("utf-8"))
                if content_list_path:
                    content_list_json = json.loads(zf.read(content_list_path).decode("utf-8"))

                layout_figures = []
                layout_source = middle_json if middle_json is not None else content_list_json
                if layout_source is not None:
                    try:
                        layout_figures = self._parse_layout_figures(json.dumps(layout_source, ensure_ascii=False))
                        logger.info(
                            f"MinerU: 从 {middle_json_path or content_list_path} 提取到 "
                            f"{len(layout_figures)} 个 figure 区域"
                        )
                    except Exception as e:
                        logger.warning(f"MinerU: 解析版面 figure 数据失败: {e}")
                else:
                    logger.info(
                        f"MinerU ZIP 中未找到 middle.json 或 content_list.json，"
                        f"跳过版面分析。ZIP 包含: {[info.filename for info in infos]}"
                    )

                return {
                    "full_md": full_md,
                    "middle_json": middle_json,
                    "content_list_json": content_list_json,
                    "layout_figures": layout_figures,
                    "zip_entries": [info.filename for info in infos],
                    "paths": {
                        "full_md": full_md_path,
                        "middle_json": middle_json_path,
                        "content_list_json": content_list_path,
                    },
                }
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"MinerU ZIP 文件解压失败: {e}")

    @staticmethod
    def _parse_layout_figures(layout_json_str: str) -> List[Dict]:
        """
        从 MinerU middle.json / content_list.json 中提取 figure 区域

        MinerU middle.json 格式示例:
        [
            {
                "type": "image",
                "bbox": [x0, y0, x1, y1],
                "page_idx": 0,
                "img_caption": "Figure 1. ...",
                ...
            },
            ...
        ]

        也兼容 pdf_info_dict 格式:
        {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "preproc_blocks": [
                        {"type": "image", "bbox": [...], ...}
                    ]
                }
            ]
        }

        返回标准化的 figure 列表，每项包含:
        - page_idx (0-indexed)
        - bbox [x0, y0, x1, y1] (PDF points)
        - caption
        - figure_index (如 "Figure 1")
        - confidence
        """
        import re as _re

        data = json.loads(layout_json_str)
        figures = []

        # 模式 1: content_list.json — 顶层列表
        if isinstance(data, list):
            for item in data:
                item_type = item.get("type", "")
                if item_type in ("image", "figure"):
                    fig = MinerUAdapter._normalize_layout_item(item)
                    if fig:
                        figures.append(fig)
            return figures

        # 模式 2: middle.json — pdf_info_dict 格式
        pdf_info = data.get("pdf_info", [])
        if isinstance(pdf_info, list):
            for page_info in pdf_info:
                page_idx = page_info.get("page_idx", 0)
                for block in page_info.get("preproc_blocks", []):
                    block_type = block.get("type", "")
                    if block_type in ("image", "figure"):
                        block["page_idx"] = page_idx
                        fig = MinerUAdapter._normalize_layout_item(block)
                        if fig:
                            figures.append(fig)

        return figures

    @staticmethod
    def _normalize_layout_item(item: dict) -> Optional[Dict]:
        """将 MinerU 版面分析的单个 item 标准化为 figure dict"""
        import re as _re

        bbox = item.get("bbox") or item.get("img_body_bbox")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None

        page_idx = item.get("page_idx", 0)
        if isinstance(page_idx, str):
            try:
                page_idx = int(page_idx)
            except (ValueError, TypeError):
                page_idx = 0

        # 提取 caption
        caption = (
            item.get("img_caption", "")
            or item.get("caption", "")
            or item.get("text", "")
            or ""
        )

        # 从 caption 提取 figure index
        figure_index = None
        for pattern in [
            r'(Figure\s+\d+[a-zA-Z]?)',
            r'(Fig\.?\s+\d+[a-zA-Z]?)',
            r'(图\s*\d+[a-zA-Z]?)',
        ]:
            m = _re.search(pattern, caption, _re.IGNORECASE)
            if m:
                figure_index = m.group(1)
                break

        return {
            "page": page_idx,
            "bbox": [float(v) for v in bbox],
            "caption": caption.strip(),
            "label": figure_index or "",
            "figure_id": f"mineru_fig_p{page_idx}_{id(item) % 10000}",
            "confidence": item.get("score", 0.8),
        }


class MinerUDirectAdapter(MinerUAdapter):
    """MinerU 官方 API 直连适配器，不依赖 Cloudflare Worker 代理。"""

    def __init__(
        self,
        token: str,
        base_url: str = "https://mineru.net/api/v4",
        enable_ocr: bool = False,
        enable_formula: bool = True,
        enable_table: bool = True,
        model_version: str = "vlm",
    ):
        self._base_url = ""
        if base_url:
            try:
                self._base_url = validate_external_ocr_service_url(
                    base_url,
                    service_name="MinerU API Base URL",
                ).rstrip("/")
            except ValueError as exc:
                logger.warning("MinerU API Base URL 已被拒绝: %s", exc)
        super().__init__(
            worker_url=self._base_url or "https://mineru.net/api/v4",
            auth_key="",
            token=token,
            token_mode="frontend",
            enable_ocr=enable_ocr,
            enable_formula=enable_formula,
            enable_table=enable_table,
            model_version=model_version,
        )

    @property
    def name(self) -> str:
        return "mineru"

    def is_available(self) -> bool:
        return bool(self._base_url and self._token)

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def analyze_pdf(self, pdf_bytes: bytes, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None, cancel_event: Any = None) -> Dict[str, Any]:
        """通过 MinerU 官方 API 运行解析并返回完整结果载荷。"""
        submission = self.submit_document(pdf_bytes, progress_callback=progress_callback)
        return self.poll_document(
            submission,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def submit_document(
        self,
        pdf_bytes: bytes,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Request an official upload URL, upload the PDF, and return its batch."""
        if not self.is_available():
            raise RuntimeError("MinerU 直连模式未配置 Token 或 Base URL")

        with create_mineru_direct_http_client() as client:
            if progress_callback:
                progress_callback({"stage": "requesting_upload", "message": "申请 MinerU 上传链接"})
            logger.info("MinerU Direct: 开始申请上传链接...")
            data_id = f"chatpdf_{uuid.uuid4().hex}"
            batch_id, upload_url = self._create_upload_url(client, data_id=data_id)

            if progress_callback:
                progress_callback({"stage": "uploading", "message": "上传 PDF 到 MinerU", "batch_id": batch_id, "data_id": data_id})
            logger.info("MinerU Direct: 开始上传 PDF 到 MinerU OSS...")
            upload_response = client.put(upload_url, content=pdf_bytes)
            if not upload_response.is_success:
                raise RuntimeError(
                    f"MinerU OSS 上传失败 (HTTP {upload_response.status_code}): "
                    f"{upload_response.text[:300]}"
                )
        return {"batch_id": batch_id, "data_id": data_id, "access_mode": "direct"}

    def poll_document(
        self,
        submission: Dict[str, Any],
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
    ) -> Dict[str, Any]:
        """Poll and download a previously submitted official MinerU batch."""
        return self.resume_batch(
            str((submission or {}).get("batch_id") or ""),
            data_id=str((submission or {}).get("data_id") or ""),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def resume_batch(
        self,
        batch_id: str,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
        data_id: str = "",
    ) -> Dict[str, Any]:
        """Resume polling a submitted official MinerU batch after restart."""
        if not self.is_available():
            raise RuntimeError("MinerU 直连模式未配置 Token 或 Base URL")
        if not batch_id:
            raise RuntimeError("恢复 MinerU 任务缺少 batch_id")
        with create_mineru_direct_http_client() as client:
            if progress_callback:
                progress_callback({"stage": "resuming", "message": "恢复 MinerU 远端任务", "batch_id": batch_id, "data_id": data_id})
            full_zip_url = self._poll_direct_result(
                client,
                batch_id,
                data_id=data_id,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            payload = self._download_direct_zip(
                client,
                full_zip_url,
                refresh_zip_url=lambda: self._poll_direct_result(
                    client,
                    batch_id,
                    data_id=data_id,
                    cancel_event=cancel_event,
                    max_attempts=1,
                    poll_interval=0,
                ),
                progress_callback=progress_callback,
            )
            payload.update({
                "batch_id": batch_id,
                "data_id": data_id,
                "full_zip_url": full_zip_url,
                "access_mode": "direct",
            })
            return payload

    def cancel_batch(self, batch_id: str, *, data_id: str = "") -> dict:
        """Best-effort cancellation against the official batch endpoint."""
        if not batch_id:
            return {"attempted": False, "state": "not_requested", "reason": "missing_batch_id"}
        try:
            with create_mineru_direct_http_client(
                timeout_seconds=30.0,
                connect_timeout_seconds=10.0,
            ) as client:
                response = client.delete(
                    f"{self._base_url}/extract-results/batch/{batch_id}",
                    headers=self._auth_headers(),
                )
                if response.status_code in (404, 405):
                    response = client.post(
                        f"{self._base_url}/extract-results/batch/{batch_id}/cancel",
                        headers=self._auth_headers(),
                    )
            if response.is_success:
                return {"attempted": True, "state": "sent", "status_code": response.status_code}
            return {"attempted": True, "state": "rejected", "status_code": response.status_code, "detail": response.text[:300]}
        except Exception as exc:
            return {"attempted": True, "state": "error", "detail": str(exc)}

    def _create_upload_url(self, client, *, data_id: str) -> tuple[str, str]:
        response = client.post(
            f"{self._base_url}/file-urls/batch",
            headers={**self._auth_headers(), "Content-Type": "application/json"},
            json={
                "enable_formula": self._enable_formula,
                "enable_table": self._enable_table,
                "language": "ch",
                "model_version": self._model_version,
                "files": [{
                    "name": "document.pdf",
                    "data_id": data_id,
                    "is_ocr": self._enable_ocr,
                }],
            },
        )
        if not response.is_success:
            raise RuntimeError(f"MinerU 申请上传链接失败 (HTTP {response.status_code}): {response.text[:300]}")
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(f"MinerU 申请上传链接失败: {data.get('msg') or data}")
        batch_id = data.get("data", {}).get("batch_id")
        file_urls = data.get("data", {}).get("file_urls") or []
        upload_url = file_urls[0] if file_urls else ""
        if not batch_id or not upload_url:
            raise RuntimeError("MinerU 申请上传链接成功但未返回 batch_id 或 file_urls")
        return batch_id, upload_url

    def _poll_direct_result(
        self,
        client,
        batch_id: str,
        *,
        data_id: str = "",
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
        max_attempts: int = 300,
        poll_interval: float = 3,
    ) -> str:
        max_attempts = max(1, int(max_attempts or 1))
        poll_interval = max(0.0, float(poll_interval or 0.0))
        for attempt in range(max_attempts):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("MinerU 深度解析已取消")
            response = client.get(
                f"{self._base_url}/extract-results/batch/{batch_id}",
                headers=self._auth_headers(),
            )
            if not response.is_success:
                raise RuntimeError(f"MinerU 查询结果失败 (HTTP {response.status_code}): {response.text[:300]}")
            data = response.json()
            if data.get("code") != 0:
                raise RuntimeError(f"MinerU 查询结果失败: {data.get('msg') or data}")
            result = self._select_direct_extract_result(data, data_id=data_id)
            state = result.get("state", "")
            if state == "done":
                full_zip_url = result.get("full_zip_url")
                if not full_zip_url:
                    raise RuntimeError("MinerU 处理完成但未返回 full_zip_url")
                return full_zip_url
            if state == "failed":
                raise RuntimeError(f"MinerU 处理失败: {result.get('err_msg') or result.get('error') or '未知错误'}")
            if progress_callback:
                progress = {
                    "stage": "polling",
                    "message": f"等待 MinerU 处理（{attempt + 1}/{max_attempts}）",
                    "batch_id": batch_id,
                    "data_id": data_id,
                    "poll_attempt": attempt + 1,
                    "poll_total": max_attempts,
                    "remote_state": state,
                }
                progress.update(extract_remote_mineru_progress(result))
                progress_callback(progress)
            logger.debug("MinerU Direct: 轮询中 (%s/%s)，当前状态: %s", attempt + 1, max_attempts, state)
            time.sleep(poll_interval)
        raise RuntimeError(f"MinerU 处理超时: 已轮询 {max_attempts} 次（共 {max_attempts * poll_interval} 秒）")

    @staticmethod
    def _select_direct_extract_result(data: Dict[str, Any], *, data_id: str = "") -> Dict[str, Any]:
        """Return the per-file result from MinerU batch status response."""
        payload = data.get("data") if isinstance(data, dict) else None
        if not isinstance(payload, dict):
            return {}

        extract_result = payload.get("extract_result")
        if isinstance(extract_result, list):
            candidates = [item for item in extract_result if isinstance(item, dict)]
            if data_id:
                for item in candidates:
                    if item.get("data_id") == data_id:
                        return item
            return candidates[0] if candidates else {}
        if isinstance(extract_result, dict):
            return extract_result

        return payload

    def _download_direct_zip(
        self,
        client,
        zip_url: str,
        *,
        refresh_zip_url: Optional[Callable[[], str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Download a direct-result archive with independent TLS attempts.

        MinerU returns a pre-signed object-storage URL.  A transient EOF on
        that host is unrelated to the completed parsing batch, so retries must
        not reuse a potentially bad pooled connection or force a PDF reupload.
        Refreshing the result URL between attempts also covers short-lived
        storage signatures.
        """
        del client  # API polling is performed by the refresh callback below.
        last_error: Exception | None = None
        current_zip_url = str(zip_url or "")
        total_attempts = _MINERU_DIRECT_ZIP_DOWNLOAD_ATTEMPTS

        for attempt in range(total_attempts):
            attempt_number = attempt + 1
            if progress_callback:
                progress_callback({
                    "stage": "downloading",
                    "message": f"下载 MinerU 解析结果（{attempt_number}/{total_attempts}）",
                    "download_attempt": attempt_number,
                    "download_total": total_attempts,
                })
            try:
                # Result archives are commonly hosted on a different CDN than
                # the MinerU API. A fresh client and Connection: close make a
                # transient TLS EOF isolated to this one attempt.
                with create_mineru_direct_http_client(
                    disable_keepalive=True,
                ) as download_client:
                    zip_bytes = _download_limited_zip(
                        download_client,
                        current_zip_url,
                        headers={"Connection": "close"},
                        service_name="MinerU ZIP",
                    )
                try:
                    return self._extract_direct_zip_payload(zip_bytes)
                except zipfile.BadZipFile as exc:
                    # A connection can terminate after HTTP framing succeeds,
                    # leaving an incomplete archive. Treat that as transport
                    # retryable rather than a terminal parse failure.
                    last_error = exc
            except Exception as exc:
                last_error = exc

            if attempt >= total_attempts - 1:
                break

            if progress_callback:
                progress_callback({
                    "stage": "retrying_download",
                    "message": "结果下载连接中断，正在刷新 MinerU 下载地址后重试",
                    "download_attempt": attempt_number,
                    "download_total": total_attempts,
                })
            if refresh_zip_url:
                try:
                    refreshed_url = str(refresh_zip_url() or "").strip()
                    if refreshed_url:
                        current_zip_url = refreshed_url
                except Exception as refresh_error:
                    logger.warning(
                        "MinerU Direct: refresh result URL failed after download attempt %s: %s",
                        attempt_number,
                        refresh_error,
                    )
            time.sleep(min(2 ** attempt_number, 8))

        raise RuntimeError(
            f"MinerU ZIP 下载失败（已使用新连接重试 {total_attempts} 次）: {last_error}"
        )

    def _extract_direct_zip_payload(self, zip_bytes: bytes) -> Dict[str, Any]:
        """Validate and normalize a downloaded official MinerU archive."""
        try:
            zip_data = io.BytesIO(zip_bytes)
            with zipfile.ZipFile(zip_data, "r") as zf:
                infos = _validate_mineru_zip(zf)
                full_md_path = None
                middle_json_path = None
                content_list_path = None
                for name in (info.filename for info in infos):
                    if name.endswith("full.md"):
                        full_md_path = name
                    elif name.endswith("middle.json"):
                        middle_json_path = name
                    elif name.endswith("content_list.json"):
                        content_list_path = name

                # 必需项以前恰好反了：缺 full.md 直接 raise，而缺 content_list.json /
                # middle.json 只记一条 info 就继续。真正喂数据的是后两者——正文、块
                # 结构、表格与坐标全部来自它们；full.md 的唯一活消费点是文本覆盖率
                # 见证。一份没有结构化数据的产物本来是不该通过的。
                if content_list_path is None and middle_json_path is None:
                    raise RuntimeError(
                        "MinerU ZIP 中未找到 content_list.json 或 middle.json，"
                        f"ZIP 包含: {[info.filename for info in infos]}"
                    )

                if full_md_path is None:
                    logger.warning(
                        "MinerU ZIP 中未找到 full.md，文本覆盖率见证将不可用。"
                        f"ZIP 包含: {[info.filename for info in infos]}"
                    )
                full_md = zf.read(full_md_path).decode("utf-8") if full_md_path else ""
                middle_json = json.loads(zf.read(middle_json_path).decode("utf-8")) if middle_json_path else None
                content_list_json = json.loads(zf.read(content_list_path).decode("utf-8")) if content_list_path else None

                layout_figures = []
                layout_source = middle_json if middle_json is not None else content_list_json
                if layout_source is not None:
                    try:
                        layout_figures = self._parse_layout_figures(json.dumps(layout_source, ensure_ascii=False))
                    except Exception as exc:
                        logger.warning("MinerU Direct: 解析版面 figure 数据失败: %s", exc)

                return {
                    "full_md": full_md,
                    "middle_json": middle_json,
                    "content_list_json": content_list_json,
                    "layout_figures": layout_figures,
                    "zip_entries": [info.filename for info in infos],
                    "paths": {
                        "full_md": full_md_path,
                        "middle_json": middle_json_path,
                        "content_list_json": content_list_path,
                    },
                }
        except zipfile.BadZipFile:
            raise


# ============================================================
# Doc2X OCR 适配器
# ============================================================

class Doc2XAdapter(WorkerOCRAdapter):
    """Doc2X OCR 适配器，通过 Worker 代理处理 PDF"""

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "doc2x"

    def _build_headers(self) -> dict:
        """
        构建请求头：在基类基础上添加 X-Doc2X-Key（frontend 模式）

        返回:
            请求头字典
        """
        headers = super()._build_headers()
        if self._token_mode == "frontend" and self._token:
            headers["X-Doc2X-Key"] = self._token
        return headers

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """Reject page-level use; Doc2X is retained only as a legacy migration provider."""
        raise RuntimeError("Doc2X 不支持逐页 OCR；请迁移到本地 OCR 或文档级解析服务")

    def _legacy_document_ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200,
    ) -> OCRResult:
        """Historical whole-document conversion kept out of the page OCR contract."""
        import httpx

        try:
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                # 步骤 1：上传 PDF
                logger.info("Doc2X OCR: 开始上传 PDF 文件...")
                uid = self._upload_pdf(client, pdf_bytes)
                logger.info(f"Doc2X OCR: 上传成功，uid={uid}")

                # 步骤 2：轮询状态，完成时直接获取 Markdown
                logger.info("Doc2X OCR: 开始轮询处理状态...")
                markdown_content = self._poll_status(client, uid)
                logger.info("Doc2X OCR: 处理完成，已获取 Markdown 内容")

                # 步骤 3：将 Markdown 转换为纯文本
                text = _markdown_to_text(markdown_content)
                text = clean_ocr_text(text)

                # 步骤 4：构建 OCRResult（Doc2X 返回整个文档的文本，按页分配）
                pages_result: List[PageOCRResult] = []
                for page_num in page_numbers:
                    display_page = page_num + 1  # 用于显示的页码（从 1 开始）
                    pages_result.append(PageOCRResult(
                        page_number=display_page,
                        text=text,
                        success=True,
                    ))

                return OCRResult(
                    pages=pages_result,
                    failed_pages=[],
                    errors={},
                    backend=self.name,
                )

        except httpx.TimeoutException as e:
            logger.error(f"Doc2X OCR: 网络连接超时: {e}")
            error_msg = f"Doc2X OCR 网络连接超时: {e}"
        except RuntimeError as e:
            logger.error(f"Doc2X OCR: {e}")
            error_msg = str(e)
        except Exception as e:
            logger.error(f"Doc2X OCR: 未知错误: {e}")
            error_msg = f"Doc2X OCR 处理失败: {e}"

        # 错误时返回包含错误信息的 OCRResult
        pages_result = []
        failed_pages = []
        errors = {}
        for page_num in page_numbers:
            display_page = page_num + 1
            failed_pages.append(display_page)
            errors[display_page] = error_msg
            pages_result.append(PageOCRResult(
                page_number=display_page,
                text="",
                success=False,
                error=error_msg,
            ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name,
        )

    def _upload_pdf(self, client, pdf_bytes: bytes) -> str:
        """
        上传 PDF 到 Doc2X Worker 代理

        参数:
            client: httpx.Client 实例
            pdf_bytes: PDF 原始字节
        返回:
            uid 任务标识
        异常:
            RuntimeError: 上传失败时抛出
        """
        headers = self._build_headers()
        response = client.post(
            f"{self._worker_url}/doc2x/upload",
            headers=headers,
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
            data={
                "ocr": "true",
                "formula_mode": "dollar",
            },
        )
        self._check_worker_response(response, "上传 PDF")
        data = response.json()
        uid = data.get("uid")
        if not uid:
            raise RuntimeError("Doc2X 上传成功但未返回 uid")
        return uid

    def _poll_status(self, client, uid: str) -> str:
        """
        轮询 Doc2X 处理状态，完成时返回 Markdown 内容

        参数:
            client: httpx.Client 实例
            uid: 任务标识
        返回:
            Markdown 内容字符串
        异常:
            RuntimeError: 处理失败或超时时抛出
        """
        headers = self._build_headers()
        max_attempts = 100
        poll_interval = 3  # 秒

        for attempt in range(max_attempts):
            response = client.get(
                f"{self._worker_url}/doc2x/status/{uid}",
                headers=headers,
            )
            self._check_worker_response(response, "轮询状态")
            data = response.json()
            state = data.get("state", "")

            if state == "done":
                markdown = data.get("markdown", "")
                return markdown
            elif state == "failed":
                error_msg = data.get("error", "未知错误")
                raise RuntimeError(f"Doc2X 处理失败: {error_msg}")

            # 继续等待
            logger.debug(
                f"Doc2X OCR: 轮询中 ({attempt + 1}/{max_attempts})，"
                f"当前状态: {state}"
            )
            if cancel_event is not None:
                if cancel_event.wait(poll_interval):
                    raise RuntimeError("MinerU 深度解析已取消")
            else:
                time.sleep(poll_interval)

        raise RuntimeError(
            f"Doc2X 处理超时: 已轮询 {max_attempts} 次（共 {max_attempts * poll_interval} 秒）"
        )


class MistralAdapter(BaseOCRAdapter):
    """Mistral OCR 在线适配器，通过 Mistral API 执行 PDF OCR"""

    def __init__(self, api_key: str, base_url: str = "https://api.mistral.ai"):
        """
        初始化 Mistral OCR 适配器

        参数:
            api_key: Mistral API Key
            base_url: Mistral API 基础 URL（默认 https://api.mistral.ai）
        """
        self._api_key = api_key
        self._base_url = ""
        if base_url:
            try:
                self._base_url = validate_external_ocr_service_url(
                    base_url,
                    service_name="Mistral OCR Base URL",
                )
            except ValueError as exc:
                logger.warning("Mistral OCR Base URL 已被拒绝: %s", exc)

    @property
    def name(self) -> str:
        """适配器名称标识"""
        return "mistral"

    def is_available(self) -> bool:
        """API Key 已配置则视为可用"""
        return bool(self._api_key and self._base_url)

    def ocr_image(self, image) -> str:
        """在线 OCR 不支持单图模式，返回空字符串"""
        return ""

    def ocr_pages(
        self,
        pdf_bytes: bytes,
        page_numbers: List[int],
        dpi: int = 200
    ) -> OCRResult:
        """
        调用 Mistral OCR API 处理整个 PDF，
        然后从结果中提取指定页码的文本。

        流程：
        1. 上传 PDF → 获取 file_id
        2. 获取签名 URL
        3. 调用 OCR 接口
        4. 解析结果，提取指定页码文本
        5. 清理远程文件（失败不影响结果）

        参数:
            pdf_bytes: PDF 原始字节
            page_numbers: 需要 OCR 的页码列表（从 0 开始）
            dpi: 图像转换分辨率（在线 OCR 忽略此参数）
        返回:
            OCRResult 包含各页结果和错误信息
        """
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        file_id = None

        try:
            # 使用 httpx 同步客户端，设置合理的超时时间
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                # 步骤 1：上传 PDF 到 Mistral
                logger.info("Mistral OCR: 开始上传 PDF 文件...")
                upload_resp = client.post(
                    f"{self._base_url}/v1/files",
                    headers=headers,
                    files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
                    data={"purpose": "ocr"},
                )
                self._check_http_error(upload_resp, "上传 PDF")
                upload_data = upload_resp.json()
                file_id = upload_data.get("id")
                if not file_id:
                    raise RuntimeError("Mistral OCR: 上传成功但未返回文件 ID")
                logger.info(f"Mistral OCR: 文件上传成功，file_id={file_id}")

                # 步骤 2：获取签名 URL
                logger.info("Mistral OCR: 获取签名 URL...")
                url_resp = client.get(
                    f"{self._base_url}/v1/files/{file_id}/url",
                    headers=headers,
                )
                self._check_http_error(url_resp, "获取签名 URL")
                url_data = url_resp.json()
                signed_url = url_data.get("url")
                if not signed_url:
                    raise RuntimeError("Mistral OCR: 获取的签名 URL 格式不正确")
                logger.info("Mistral OCR: 签名 URL 获取成功")

                # 步骤 3：调用 OCR 接口
                logger.info("Mistral OCR: 开始 OCR 处理...")
                ocr_resp = client.post(
                    f"{self._base_url}/v1/ocr",
                    headers={**headers, "Content-Type": "application/json", "Accept": "application/json"},
                    json={
                        "model": "mistral-ocr-latest",
                        "document": {"type": "document_url", "document_url": signed_url},
                        "include_image_base64": False,
                    },
                )
                self._check_http_error(ocr_resp, "OCR 处理")
                ocr_data = ocr_resp.json()
                if not ocr_data or "pages" not in ocr_data:
                    raise RuntimeError("Mistral OCR: OCR 处理成功但返回的数据格式不正确")
                logger.info(f"Mistral OCR: OCR 处理完成，共 {len(ocr_data['pages'])} 页")

                # 步骤 4：解析结果，提取指定页码文本
                result = self._parse_ocr_response(ocr_data, page_numbers)

                return result

        except httpx.TimeoutException as e:
            logger.error(f"Mistral OCR: 网络连接超时: {e}")
            raise RuntimeError(f"Mistral OCR 网络连接超时: {e}") from e
        except httpx.HTTPError as e:
            # httpx 的其他网络错误（非 HTTP 状态码错误）
            logger.error(f"Mistral OCR: 网络错误: {e}")
            raise RuntimeError(f"Mistral OCR 网络错误: {e}") from e
        finally:
            # 上传成功后的任意后续阶段都可能失败；必须尝试清理远端临时文件，
            # 不能只在 OCR 成功路径删除。
            if file_id:
                try:
                    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as cleanup_client:
                        self._delete_file(cleanup_client, headers, file_id)
                except Exception as exc:
                    logger.warning("Mistral OCR: 初始化远程文件清理客户端失败: %s", exc)

    def _check_http_error(self, response, step: str) -> None:
        """
        检查 HTTP 响应状态码，对 401/403 做特殊处理

        参数:
            response: httpx.Response 对象
            step: 当前步骤描述（用于错误消息）
        """
        if response.is_success:
            return

        status_code = response.status_code
        try:
            error_detail = response.text
        except Exception:
            error_detail = response.reason_phrase or "未知错误"

        if status_code in (401, 403):
            raise RuntimeError(
                f"Mistral OCR {step}失败: API Key 无效或已过期 "
                f"(HTTP {status_code})"
            )

        raise RuntimeError(
            f"Mistral OCR {step}失败 (HTTP {status_code}): {error_detail}"
        )

    def _parse_ocr_response(
        self, ocr_data: dict, page_numbers: List[int]
    ) -> OCRResult:
        """
        解析 Mistral OCR 响应，提取指定页码的文本

        参数:
            ocr_data: Mistral OCR API 返回的 JSON 数据
            page_numbers: 需要提取的页码列表（从 0 开始）
        返回:
            OCRResult 包含各页结果
        """
        api_pages = ocr_data.get("pages", [])
        pages_result: List[PageOCRResult] = []
        failed_pages: List[int] = []
        errors: Dict[int, str] = {}

        for page_num in page_numbers:
            # page_numbers 从 0 开始，Mistral API 返回的 pages 列表也是从 0 开始索引
            display_page = page_num + 1  # 用于显示的页码（从 1 开始）

            if page_num < len(api_pages):
                page_data = api_pages[page_num]
                markdown_content = page_data.get("markdown", "")
                # 将 Markdown 转换为纯文本
                text = self._markdown_to_text(markdown_content)
                text = clean_ocr_text(text)
                pages_result.append(PageOCRResult(
                    page_number=display_page,
                    text=text,
                    success=True,
                ))
            else:
                # 请求的页码超出 OCR 结果范围
                error_msg = f"页码 {display_page} 超出 OCR 结果范围（共 {len(api_pages)} 页）"
                logger.warning(f"Mistral OCR: {error_msg}")
                failed_pages.append(display_page)
                errors[display_page] = error_msg
                pages_result.append(PageOCRResult(
                    page_number=display_page,
                    text="",
                    success=False,
                    error=error_msg,
                ))

        return OCRResult(
            pages=pages_result,
            failed_pages=failed_pages,
            errors=errors,
            backend=self.name,
        )

    @staticmethod
    def _markdown_to_text(markdown: str) -> str:
        """
        将 Markdown 内容转换为纯文本，清理 Markdown 标记

        委托给模块级函数 _markdown_to_text()，保持向后兼容。

        参数:
            markdown: Markdown 格式的文本
        返回:
            纯文本字符串
        """
        return _markdown_to_text(markdown)

    def _delete_file(self, client, headers: dict, file_id: str) -> None:
        """
        删除 Mistral 服务器上的临时文件，失败不影响 OCR 结果

        参数:
            client: httpx.Client 实例
            headers: 请求头（包含 Authorization）
            file_id: 要删除的文件 ID
        """
        try:
            resp = client.delete(
                f"{self._base_url}/v1/files/{file_id}",
                headers=headers,
            )
            if resp.is_success:
                logger.info(f"Mistral OCR: 远程文件已清理，file_id={file_id}")
            else:
                logger.warning(
                    f"Mistral OCR: 远程文件清理失败 (HTTP {resp.status_code})，"
                    f"file_id={file_id}"
                )
        except Exception as e:
            logger.warning(f"Mistral OCR: 远程文件清理异常: {e}，file_id={file_id}")


# ============================================================
# OCR 适配器注册表
# ============================================================

class OCRRegistry:
    """OCR 适配器注册表，管理可用适配器的注册与查找"""

    def __init__(self):
        self._adapters: Dict[str, BaseOCRAdapter] = {}

    def register(self, adapter: BaseOCRAdapter) -> None:
        """
        注册一个适配器（仅当 is_available() 为 True 时）

        参数:
            adapter: 要注册的 OCR 适配器实例
        """
        if adapter.name in _DOCUMENT_PARSE_PROVIDER_NAMES:
            logger.warning("拒绝将文档级解析器注册为逐页 OCR: %s", adapter.name)
            return
        if adapter.is_available():
            self._adapters[adapter.name] = adapter
            logger.info(f"OCR 适配器已注册: {adapter.name}")
        else:
            logger.debug(f"OCR 适配器 {adapter.name} 不可用，跳过注册")

    def get_adapter(self, name: str = "auto") -> Optional[BaseOCRAdapter]:
        """
        获取适配器

        参数:
            name: 适配器名称，"auto" 时只选择本地逐页 OCR 适配器
        返回:
            适配器实例，无可用适配器时返回 None
        """
        name = str(name or "auto").strip().lower()
        if name in _DOCUMENT_PARSE_PROVIDER_NAMES:
            logger.warning("%s 是文档级解析器，不能作为逐页 OCR 获取", name)
            return None
        if name == "auto":
            # 自动页级 OCR 不得静默上传整篇 PDF。云端服务必须由用户显式选择。
            for key in ["paddleocr", "tesseract"]:
                if key in self._adapters:
                    logger.debug(f"自动选择 OCR 适配器: {key}")
                    return self._adapters[key]
            # 无任何可用适配器，记录可用性检测结果
            logger.warning(
                "无可用的 OCR 适配器。已检测的后端: "
                f"paddleocr={'已注册' if 'paddleocr' in self._adapters else '不可用'}, "
                f"tesseract={'已注册' if 'tesseract' in self._adapters else '不可用'}；"
                "在线 OCR 需要显式选择"
            )
            return None
        return self._adapters.get(name)

    # 在线 OCR 适配器名称集合，用于回退时排除
    _ONLINE_ADAPTERS = {"mistral"}

    def get_local_adapter(self, exclude: Optional[List[str]] = None) -> Optional[BaseOCRAdapter]:
        """
        获取本地 OCR 适配器（排除在线适配器和指定的适配器）

        用于在线 OCR 失败时回退到本地引擎。
        按优先级 paddleocr > tesseract 返回第一个可用的本地适配器。

        参数:
            exclude: 需要额外排除的适配器名称列表
        返回:
            本地适配器实例，无可用本地适配器时返回 None
        """
        exclude_set = set(self._ONLINE_ADAPTERS)
        if exclude:
            exclude_set.update(exclude)

        # 本地适配器优先级：paddleocr > tesseract
        for key in ["paddleocr", "tesseract"]:
            if key in self._adapters and key not in exclude_set:
                logger.debug(f"选择本地 OCR 适配器: {key}")
                return self._adapters[key]

        logger.warning("无可用的本地 OCR 适配器用于回退")
        return None

    def list_available(self) -> Dict[str, bool]:
        """
        列出所有已注册适配器的可用状态

        返回:
            字典，键为适配器名称，值为 True（已注册即可用）
        """
        return {name: True for name in self._adapters}


class DocumentParserRegistry:
    """Registry for document-level parsers, separate from PageOCR providers."""

    def __init__(self):
        self._adapters: Dict[str, DocumentParseAdapter] = {}

    def register(self, adapter: DocumentParseAdapter) -> None:
        if not isinstance(adapter, DocumentParseAdapter):
            raise TypeError("文档解析适配器必须实现 DocumentParseAdapter 合同")
        if adapter.is_available():
            self._adapters[adapter.name] = adapter
            logger.info("文档解析适配器已注册: %s", adapter.name)
        else:
            logger.debug("文档解析适配器 %s 不可用，跳过注册", adapter.name)

    def unregister(self, name: str) -> None:
        self._adapters.pop(str(name or "").strip().lower(), None)

    def get_adapter(self, name: str) -> DocumentParseAdapter | None:
        return self._adapters.get(str(name or "").strip().lower())

    def list_available(self) -> Dict[str, bool]:
        return {name: True for name in self._adapters}


def is_ocr_available() -> dict:
    """
    检查可用的 OCR 后端

    使用全局 OCRRegistry 注册表查询已注册的适配器，
    同时保持向后兼容的返回格式。
    """
    available = _ocr_registry.list_available()
    document_parsers = _document_parser_registry.list_available()
    return {
        "tesseract": available.get("tesseract", False),
        "paddleocr": available.get("paddleocr", False),
        "mistral": available.get("mistral", False),
        "mineru": document_parsers.get("mineru", False),
        "doc2x": False,
        "any": bool(available or document_parsers)
    }


def detect_pdf_quality(text: str, page_count: int) -> Tuple[bool, str]:
    """
    Detect if PDF text extraction quality is poor and needs OCR
    Returns: (needs_ocr, reason)
    """
    if not text or len(text.strip()) < 100:
        return True, "文本内容过少，可能是扫描版PDF"
    
    # Check for garbled text (high ratio of replacement characters or control chars)
    garbled_chars = len(re.findall(r'[�□■◆●○◇△▽▲▼\x00-\x08\x0b\x0c\x0e-\x1f]', text))
    total_chars = len(text)
    
    if total_chars > 0 and garbled_chars / total_chars > 0.05:
        return True, "检测到大量乱码字符，可能是编码问题"
    
    # Check average chars per page (academic papers typically have 2000-4000 chars/page)
    avg_chars_per_page = total_chars / max(page_count, 1)
    if avg_chars_per_page < 200:
        return True, f"每页平均字符数过少({avg_chars_per_page:.0f})，可能是图片型PDF"
    
    # Check for meaningful content (not just whitespace and numbers)
    meaningful_text = re.sub(r'[\s\d\W]+', '', text)
    if len(meaningful_text) < total_chars * 0.3:
        return True, "有效文本内容比例过低"
    
    return False, "文本提取质量正常"


def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR output text"""
    if not text:
        return ""
    
    # Fix common OCR errors
    text = re.sub(r'[|l](?=[a-z])', 'I', text)  # |ower -> lower
    text = re.sub(r'(?<=[a-z])[0O](?=[a-z])', 'o', text)  # g0od -> good
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove isolated single characters (OCR noise)
    text = re.sub(r'\n[a-zA-Z]\n', '\n', text)
    
    return text.strip()


class OCRService:
    """OCR service with multiple backend support"""
    
    def __init__(self, backend: str = "auto", lang: str = "chi_sim+eng"):
        """
        Initialize OCR service
        
        Args:
            backend: "tesseract", "paddleocr", or "auto"
            lang: Language code for Tesseract (chi_sim+eng for Chinese+English)
        """
        self.backend = backend
        self.lang = lang
        self._paddle_ocr = None
        
        if backend == "auto":
            if PADDLEOCR_AVAILABLE and PDF2IMAGE_AVAILABLE:
                self.backend = "paddleocr"
            elif TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE:
                self.backend = "tesseract"
            else:
                self.backend = "none"
    
    def _get_paddle_ocr(self):
        """Lazy load PaddleOCR instance"""
        if self._paddle_ocr is None and PADDLEOCR_AVAILABLE:
            self._paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                show_log=False,
                use_gpu=False
            )
        return self._paddle_ocr
    
    def ocr_image(self, image: "Image.Image") -> str:
        """OCR a single image"""
        if self.backend == "tesseract" and TESSERACT_AVAILABLE:
            return pytesseract.image_to_string(image, lang=self.lang)
        
        elif self.backend == "paddleocr" and PADDLEOCR_AVAILABLE:
            ocr = self._get_paddle_ocr()
            import numpy as np
            img_array = np.array(image)
            result = ocr.ocr(img_array, cls=True)
            
            if not result or not result[0]:
                return ""
            
            lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    lines.append(text)
            return '\n'.join(lines)
        
        return ""
    
    def ocr_pdf_bytes(self, pdf_bytes: bytes, dpi: int = 200) -> dict:
        """
        OCR a PDF from bytes
        
        Args:
            pdf_bytes: PDF file content as bytes
            dpi: Resolution for PDF to image conversion
            
        Returns:
            dict with full_text, pages, total_pages
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image not installed. Run: pip install pdf2image")
        
        if self.backend == "none":
            raise RuntimeError("No OCR backend available. Install pytesseract or paddleocr")
        
        logger.info("Starting OCR with backend: %s", self.backend)
        
        # Convert PDF to images (with poppler path if available)
        try:
            if _poppler_path:
                images = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=_poppler_path)
            else:
                images = convert_from_bytes(pdf_bytes, dpi=dpi)
        except Exception as e:
            raise RuntimeError(
                f"PDF转图片失败: {e}\n"
                "请按照以下指引安装 Poppler:\n"
                "  - Windows: 下载 https://github.com/oschwartz10612/poppler-windows/releases 并解压到 ocr_tools/poppler/\n"
                "  - macOS: brew install poppler\n"
                "  - Linux: sudo apt-get install poppler-utils\n"
                "详情请参考: https://poppler.freedesktop.org/"
            )
        
        total_pages = len(images)
        
        pages = []
        full_text_parts = []
        
        for i, image in enumerate(images):
            logger.debug("OCR processing page %s/%s...", i + 1, total_pages)
            
            page_text = self.ocr_image(image)
            page_text = clean_ocr_text(page_text)
            
            pages.append({
                "page": i + 1,
                "content": page_text
            })
            full_text_parts.append(page_text)
        
        full_text = "\n\n".join(full_text_parts)
        
        return {
            "full_text": full_text,
            "total_pages": total_pages,
            "pages": pages,
            "ocr_used": True,
            "ocr_backend": self.backend
        }


# ============================================================
# 全局 OCR 注册表实例
# ============================================================

# 创建全局注册表并注册可用的适配器
_ocr_registry = OCRRegistry()
_document_parser_registry = DocumentParserRegistry()
_ocr_registry.register(TesseractAdapter())
_ocr_registry.register(PaddleOCRAdapter())

# 注册在线 OCR 适配器：加载 Mistral OCR 配置并注册
_mistral_config = _load_online_ocr_config("mistral")
_ocr_registry.register(MistralAdapter(
    api_key=_mistral_config.get("api_key", ""),
    base_url=_mistral_config.get("base_url", "https://api.mistral.ai"),
))

# 注册在线 OCR 适配器：加载 MinerU OCR 配置并注册
_mineru_config = _load_online_ocr_config("mineru")
if _mineru_config.get("access_mode") == "direct":
    _document_parser_registry.register(MinerUDocumentParseAdapter(MinerUDirectAdapter(
        token=_mineru_config.get("token", ""),
        base_url=_mineru_config.get("base_url", "https://mineru.net/api/v4"),
        enable_ocr=_mineru_config.get("enable_ocr", False),
        enable_formula=_mineru_config.get("enable_formula", True),
        enable_table=_mineru_config.get("enable_table", True),
        model_version=_mineru_config.get("model_version", "vlm"),
    )))
else:
    _document_parser_registry.register(MinerUDocumentParseAdapter(MinerUAdapter(
        worker_url=_mineru_config.get("worker_url", ""),
        auth_key=_mineru_config.get("auth_key", ""),
        token=_mineru_config.get("token", ""),
        token_mode=_mineru_config.get("token_mode", "frontend"),
        enable_ocr=_mineru_config.get("enable_ocr", False),
        enable_formula=_mineru_config.get("enable_formula", True),
        enable_table=_mineru_config.get("enable_table", True),
        model_version=_mineru_config.get("model_version", "vlm"),
    )))


# 保留旧的全局 OCRService 实例（向后兼容）
_ocr_service: Optional[OCRService] = None


def get_ocr_service(backend: str = "auto") -> OCRService:
    """
    获取 OCR 服务实例

    优先通过 OCRRegistry 注册表获取适配器，
    如果注册表中有可用适配器则使用注册表的结果确定后端，
    否则回退到旧的 OCRService 实例化逻辑。
    保持向后兼容的函数签名。

    参数:
        backend: OCR 后端名称，"auto" 时自动选择
    返回:
        OCRService 实例
    """
    global _ocr_service

    requested_backend = str(backend or "auto").strip().lower()
    if requested_backend in _DOCUMENT_PARSE_PROVIDER_NAMES:
        raise ValueError(f"{requested_backend} 是文档级解析器，不能用于逐页 OCR")

    # 通过注册表确定实际可用的后端
    adapter = _ocr_registry.get_adapter(requested_backend)
    if adapter is not None:
        resolved_backend = adapter.name
    else:
        resolved_backend = requested_backend

    if _ocr_service is None or _ocr_service.backend != resolved_backend:
        _ocr_service = OCRService(backend=resolved_backend)
    return _ocr_service


def ocr_pdf(pdf_bytes: bytes, backend: str = "auto", dpi: int = 200) -> dict:
    """
    便捷函数：对 PDF 执行 OCR

    使用全局 OCRRegistry 注册表获取适配器，
    保持向后兼容的函数签名和返回格式。

    参数:
        pdf_bytes: PDF 文件字节内容
        backend: OCR 后端名称，"auto" 时自动选择
        dpi: 图像转换分辨率
    返回:
        包含提取文本数据的字典
    """
    service = get_ocr_service(backend)
    return service.ocr_pdf_bytes(pdf_bytes, dpi=dpi)
