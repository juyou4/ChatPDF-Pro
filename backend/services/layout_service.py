"""
Layout Service - 基于 DocLayout-YOLO 的文档布局检测

借鉴 MinerU 的 ImageBody/Text 分离思想，
使用 DocLayout-YOLO 模型精确检测页面中的图像主体区域。

核心功能：
- detect_layout: 检测页面所有布局元素
- get_image_body_bboxes: 只返回 ImageBody 类别的 bbox
"""
import logging
import os
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

# 默认模型路径
_MODEL_DIR = Path(__file__).parent.parent / "models"
_MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1280.pt"

# 全局单例
_model_instance = None
_model_device = None



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
    """从 HuggingFace 下载 DocLayout-YOLO 模型权重"""
    if model_path.exists():
        logger.info(f"[LayoutService] Model already exists: {model_path}")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        logger.info("[LayoutService] Downloading DocLayout-YOLO model from HuggingFace...")
        downloaded = hf_hub_download(
            repo_id="opendatalab/PDF-Extract-Kit-1.0",
            filename="models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt",
            local_dir=str(model_path.parent / "_hf_cache"),
        )
        # 移动到目标位置
        import shutil
        shutil.move(str(downloaded), str(model_path))
        logger.info(f"[LayoutService] Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"[LayoutService] Failed to download model: {e}")
        raise


def _get_model():
    """获取或初始化模型单例（lazy init）"""
    global _model_instance, _model_device

    if _model_instance is not None:
        return _model_instance

    model_path = _MODEL_DIR / _MODEL_FILENAME

    # 自动下载
    model_path = _download_model(model_path)

    device = _get_device()
    _model_device = device

    try:
        from doclayout_yolo import YOLOv10
        _model_instance = YOLOv10(str(model_path)).to(device)
        logger.info(f"[LayoutService] Model loaded on {device}")
        return _model_instance
    except Exception as e:
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
    try:
        from doclayout_yolo import YOLOv10
        return True
    except ImportError:
        return False
