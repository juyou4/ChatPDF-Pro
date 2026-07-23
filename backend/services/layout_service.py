"""
Layout Service - 基于 DocLayout-YOLO 的文档布局检测

借鉴 MinerU 的 ImageBody/Text 分离思想，
使用 DocLayout-YOLO 模型精确检测页面中的图像主体区域。

核心功能：
- detect_layout: 检测页面所有布局元素
- get_image_body_bboxes: 只返回 ImageBody 类别的 bbox
"""
import json
import hashlib
import logging
import os
import shutil
import sys
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# DocLayout-YOLO 类别定义（与 MinerU 一致）
CATEGORY_MAP = {
    0: "Title",
    1: "Text",
    2: "Abandon",
    3: "ImageBody",
    4: "ImageCaption",
    5: "TableBody",
    6: "TableCaption",
    7: "TableFootnote",
    8: "InterlineEquation",
    9: "EquationNumber",
}

_MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1280.pt"
_MODEL_REPO_ID = "opendatalab/PDF-Extract-Kit-1.0"
_MODEL_REPO_FILENAME = "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
_MODEL_CONFIG_FILENAME = "layout_model.json"
_MODEL_HASH_CONFIG_KEY = "model_sha256"

# 全局单例
_model_instance = None
_model_device = None
_model_path: Optional[Path] = None
_model_lock = threading.Lock()
_last_error = ""


def _runtime_data_dir() -> Path:
    """返回运行时数据目录，桌面端对应用户 AppData。"""
    try:
        from runtime_mode import runtime
        return Path(runtime.data_dir)
    except Exception:
        return Path(__file__).resolve().parents[2] / "data"


def _model_config_path() -> Path:
    return _runtime_data_dir() / "config" / _MODEL_CONFIG_FILENAME


def _default_model_dir() -> Path:
    return _runtime_data_dir() / "models" / "layout"


def _default_model_path() -> Path:
    return _default_model_dir() / _MODEL_FILENAME


def _load_model_config() -> Dict[str, Any]:
    path = _model_config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[LayoutService] Failed to read model config %s: %s", path, exc)
        return {}


def _save_model_config(config: Dict[str, Any]) -> None:
    path = _model_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_user_path(value: str) -> Path:
    cleaned = (value or "").strip().strip('"')
    if not cleaned:
        raise ValueError("路径不能为空")
    if "\x00" in cleaned:
        raise ValueError("路径包含非法字符")
    if cleaned.startswith("\\\\") or cleaned.startswith("//"):
        raise ValueError("不允许使用网络共享路径")
    return Path(cleaned).expanduser().resolve()


def _is_within_directory(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_verified_managed_model(path: Path) -> bool:
    """Only load weights produced by the managed official download flow."""
    if not _is_within_directory(path, _default_model_dir()) or not path.is_file():
        return False
    config = _load_model_config()
    configured_path = str(config.get("model_path") or "").strip()
    expected_hash = str(config.get(_MODEL_HASH_CONFIG_KEY) or "").strip().lower()
    if not configured_path or not expected_hash:
        return False
    try:
        return Path(configured_path).expanduser().resolve() == path.resolve() and _sha256_file(path) == expected_hash
    except OSError:
        return False


def _candidate_model_paths() -> List[tuple[Path, str]]:
    """按优先级返回可用权重候选路径。"""
    candidates: List[tuple[Path, str]] = []
    configured = (_load_model_config().get("model_path") or "").strip()
    if configured:
        try:
            configured_path = Path(configured).expanduser().resolve()
            candidates.append((configured_path, "custom"))
        except Exception:
            pass

    candidates.append((_default_model_path(), "default"))

    # 源码开发模式保留兼容：本地忽略的 backend/models/*.pt 可直接用于调试。
    if not getattr(sys, "frozen", False):
        candidates.append((Path(__file__).parent.parent / "models" / _MODEL_FILENAME, "source_tree"))

    return candidates


def _resolve_model_path() -> tuple[Optional[Path], str]:
    for path, source in _candidate_model_paths():
        if path.exists() and path.is_file():
            if source in {"custom", "default"} and not _is_verified_managed_model(path):
                logger.warning("[LayoutService] 忽略未校验的受管 YOLO 权重: %s", path)
                continue
            return path, source
    return None, "missing"


def _check_dependencies() -> tuple[bool, str]:
    missing = [
        name
        for name in ("doclayout_yolo", "torch", "cv2")
        if find_spec(name) is None
    ]
    if missing:
        return False, f"缺少依赖: {', '.join(missing)}"
    return True, ""


def reset_model_cache() -> None:
    """清空已加载的 YOLO 模型，配置变更后调用。"""
    global _model_instance, _model_device, _model_path
    with _model_lock:
        _model_instance = None
        _model_device = None
        _model_path = None


def get_yolo_model_status() -> Dict[str, Any]:
    """返回 DocLayout-YOLO 运行依赖和权重配置状态。"""
    deps_ok, deps_error = _check_dependencies()
    path, source = _resolve_model_path()
    config = _load_model_config()
    model_installed = path is not None and path.exists()
    configured_path = (config.get("model_path") or "").strip()

    return {
        "available": bool(deps_ok and model_installed),
        "dependencies_available": deps_ok,
        "dependencies_error": deps_error,
        "model_installed": bool(model_installed),
        "model_path": str(path) if path else "",
        "configured_model_path": configured_path,
        "source": source,
        "default_install_dir": str(_default_model_dir()),
        "expected_filename": _MODEL_FILENAME,
        "download": {
            "repo_id": _MODEL_REPO_ID,
            "filename": _MODEL_REPO_FILENAME,
        },
        "last_error": _last_error,
    }


def configure_yolo_model_path(model_path: str) -> Dict[str, Any]:
    """选择已验证的受管 YOLO 权重，禁止任意 .pt 反序列化。"""
    global _last_error

    path = _normalize_user_path(model_path)
    if path.is_dir():
        path = path / _MODEL_FILENAME
    if not path.exists() or not path.is_file():
        _last_error = f"权重文件不存在: {path}"
        raise FileNotFoundError(_last_error)
    if path.suffix.lower() != ".pt":
        _last_error = "权重文件必须是 .pt 文件"
        raise ValueError(_last_error)

    if not _is_within_directory(path, _default_model_dir()):
        _last_error = "仅允许选择应用受管模型目录中的官方权重"
        raise ValueError(_last_error)
    if not _is_verified_managed_model(path):
        _last_error = "权重未通过校验，请使用内置下载功能重新下载"
        raise ValueError(_last_error)

    _save_model_config({
        "model_path": str(path),
        _MODEL_HASH_CONFIG_KEY: _sha256_file(path),
    })
    reset_model_cache()
    _last_error = ""
    return get_yolo_model_status()


def reset_yolo_model_config() -> Dict[str, Any]:
    """清除用户自定义 YOLO 权重路径，回到默认安装目录。"""
    global _last_error
    config_path = _model_config_path()
    config = _load_model_config()
    default_path = _default_model_path()
    if default_path.exists() and _is_verified_managed_model(default_path):
        _save_model_config({
            "model_path": str(default_path),
            _MODEL_HASH_CONFIG_KEY: _sha256_file(default_path),
        })
    elif config_path.exists():
        config_path.unlink()
    reset_model_cache()
    _last_error = ""
    return get_yolo_model_status()


def download_yolo_model(install_dir: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """下载 DocLayout-YOLO 权重到用户指定目录或默认用户数据目录。"""
    global _last_error

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        _last_error = f"缺少 huggingface_hub，无法自动下载: {exc}"
        raise RuntimeError(_last_error) from exc

    if install_dir and install_dir.strip():
        target_dir = _normalize_user_path(install_dir)
        if target_dir.suffix.lower() == ".pt":
            target_dir = target_dir.parent
        if not _is_within_directory(target_dir, _default_model_dir()):
            raise ValueError("YOLO 权重只能安装到应用受管模型目录")
    else:
        target_dir = _default_model_dir()
    target_path = target_dir / _MODEL_FILENAME

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not force:
            if not _is_verified_managed_model(target_path):
                raise ValueError("现有 YOLO 权重未通过校验，请勾选强制下载以替换它")
            _save_model_config({
                "model_path": str(target_path),
                _MODEL_HASH_CONFIG_KEY: _sha256_file(target_path),
            })
            reset_model_cache()
            _last_error = ""
            status = get_yolo_model_status()
            status["downloaded"] = False
            return status

        cache_dir = target_dir / ".hf_cache"
        downloaded = hf_hub_download(
            repo_id=_MODEL_REPO_ID,
            filename=_MODEL_REPO_FILENAME,
            cache_dir=str(cache_dir),
        )

        part_path = target_path.with_suffix(target_path.suffix + ".part")
        if part_path.exists():
            part_path.unlink()
        shutil.copy2(downloaded, part_path)
        part_path.replace(target_path)

        # 清理 HuggingFace 临时缓存，避免在用户目录留下一份冗余权重副本
        shutil.rmtree(cache_dir, ignore_errors=True)

        _save_model_config({
            "model_path": str(target_path),
            _MODEL_HASH_CONFIG_KEY: _sha256_file(target_path),
        })
        reset_model_cache()
        _last_error = ""
        status = get_yolo_model_status()
        status["downloaded"] = True
        return status
    except Exception as exc:
        _last_error = f"YOLO 权重下载失败: {exc}"
        logger.error("[LayoutService] %s", _last_error)
        raise

def _get_device() -> str:
    """检测可用设备
    
    在 Web 服务器环境中强制使用 CPU，避免与 sentence-transformers 等
    其他 GPU 模型共享显存导致 CUDA 硬崩溃（exit code 1, 无 traceback）。
    设置环境变量 LAYOUT_USE_GPU=1 可强制启用 GPU。
    """
    force_gpu = os.environ.get("LAYOUT_USE_GPU", "").strip() == "1"
    if force_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"[LayoutService] Using GPU (forced): {torch.cuda.get_device_name(0)}")
                return "cuda"
        except ImportError:
            pass
    logger.info("[LayoutService] Using CPU (safe mode)")
    return "cpu"


def _download_model(model_path: Path) -> Path:
    """兼容旧调用：不在推理路径自动下载。"""
    if model_path.exists():
        return model_path
    raise FileNotFoundError(
        "DocLayout-YOLO 权重未安装。请在 OCR 设置中一键下载，或手动指定 .pt 权重路径。"
    )


def _get_model():
    """获取或初始化模型单例（lazy init）"""
    global _model_instance, _model_device, _model_path, _last_error

    resolved_path, _source = _resolve_model_path()
    if not resolved_path:
        _last_error = "DocLayout-YOLO 权重未安装"
        raise FileNotFoundError(
            "DocLayout-YOLO 权重未安装。请在 OCR 设置中一键下载，或手动指定 .pt 权重路径。"
        )

    with _model_lock:
        if _model_instance is not None and _model_path == resolved_path:
            return _model_instance

        device = _get_device()
        _model_device = device

        try:
            from doclayout_yolo import YOLOv10
            _model_instance = YOLOv10(str(resolved_path)).to(device)
            _model_path = resolved_path
            _last_error = ""
            logger.info(f"[LayoutService] Model loaded on {device}: {resolved_path}")
            return _model_instance
        except Exception as e:
            _last_error = f"DocLayout-YOLO 加载失败: {e}"
            logger.error(f"[LayoutService] Failed to load model: {e}")
            raise


def detect_layout(
    page_image: Image.Image,
    conf: float = 0.15,
    iou: float = 0.45,
    imgsz: int = 1280,
) -> List[Dict[str, Any]]:
    """检测页面布局

    Args:
        page_image: PIL Image 格式的页面图片
        conf: 置信度阈值
        iou: NMS IoU 阈值
        imgsz: 推理图片尺寸

    Returns:
        List[dict]: 每个元素含 category_id, category_name, bbox, score
    """
    model = _get_model()

    prediction = model.predict(
        page_image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    results = []
    if not hasattr(prediction, "boxes") or prediction.boxes is None:
        return results

    for xyxy, conf_val, cls in zip(
        prediction.boxes.xyxy.cpu(),
        prediction.boxes.conf.cpu(),
        prediction.boxes.cls.cpu(),
    ):
        cat_id = int(cls.item())
        coords = xyxy.tolist()
        results.append({
            "category_id": cat_id,
            "category_name": CATEGORY_MAP.get(cat_id, f"Unknown_{cat_id}"),
            "bbox": coords,  # [x0, y0, x1, y1] 像素坐标
            "score": round(float(conf_val.item()), 3),
        })

    return results


def get_image_body_bboxes(
    page_image: Image.Image,
    conf: float = 0.15,
) -> List[List[float]]:
    """只返回 ImageBody 类别的 bbox 列表

    Args:
        page_image: PIL Image 格式的页面图片
        conf: 置信度阈值

    Returns:
        List[[x0, y0, x1, y1]]: ImageBody 的像素坐标 bbox 列表
    """
    detections = detect_layout(page_image, conf=conf)
    return [
        det["bbox"]
        for det in detections
        if det["category_id"] == 3  # ImageBody
    ]


def get_table_bboxes(
    page_image: Image.Image,
    conf: float = 0.15,
    include_caption: bool = False,
    include_footnote: bool = False,
) -> List[Dict[str, Any]]:
    """返回 DocLayout-YOLO 检测到的表格相关 bbox。

    默认只返回 TableBody；caption/footnote 只用于需要更宽裁剪区域的 fallback
    场景，避免把正文段落误当作表格内容进入索引。
    """
    allowed = {5}  # TableBody
    if include_caption:
        allowed.add(6)
    if include_footnote:
        allowed.add(7)

    detections = detect_layout(page_image, conf=conf)
    return [
        det
        for det in detections
        if det.get("category_id") in allowed
    ]


def pixel_bbox_to_page_pts(
    pixel_bbox: List[float],
    page_width_px: int,
    page_height_px: int,
    page_width_pts: float,
    page_height_pts: float,
) -> List[float]:
    """将像素坐标 bbox 转换为 PDF page points 坐标

    Args:
        pixel_bbox: [x0, y0, x1, y1] 像素坐标
        page_width_px: 页面图片宽度（像素）
        page_height_px: 页面图片高度（像素）
        page_width_pts: PDF 页面宽度（points）
        page_height_pts: PDF 页面高度（points）

    Returns:
        [x0, y0, x1, y1] page points 坐标
    """
    if page_width_px <= 0 or page_height_px <= 0:
        return pixel_bbox

    scale_x = page_width_pts / page_width_px
    scale_y = page_height_pts / page_height_px

    return [
        pixel_bbox[0] * scale_x,
        pixel_bbox[1] * scale_y,
        pixel_bbox[2] * scale_x,
        pixel_bbox[3] * scale_y,
    ]


def is_available() -> bool:
    """检查 layout service 是否可用"""
    return bool(get_yolo_model_status().get("available"))
