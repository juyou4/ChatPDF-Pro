"""
Figure Validation - 质量门校验服务

负责：
1. 基础校验：亮度、尺寸、宽高比
2. 文本污染检测（可选高级）
3. Fallback 策略链
"""
import logging
from typing import List, Dict, Optional, Tuple
import io

from PIL import Image
import numpy as np

from schemas.figure_schema import LogicalFigureSchema, RenderResult, ValidationResult

logger = logging.getLogger(__name__)


# 校验规则配置
VALIDATION_RULES = {
    # 亮度检测
    "min_brightness": 10,      # 全黑检测
    "max_brightness": 245,     # 全白检测
    
    # 尺寸检测
    "min_width": 50,
    "min_height": 50,
    
    # 宽高比检测
    "min_aspect_ratio": 0.1,
    "max_aspect_ratio": 10.0,
    
    # 文本污染检测（可选）
    "text_pollution_threshold": 0.3,  # 超过 30% 文本认为有污染
}


# Fallback 策略配置
FALLBACK_STRATEGIES = [
    {"type": "padding", "value": 10, "description": "扩展 padding 到 10pt"},
    {"type": "padding", "value": 20, "description": "扩展 padding 到 20pt"},
    {"type": "switch_bbox", "target": "full", "description": "切换到 full_bbox"},
    {"type": "switch_bbox", "target": "body", "description": "切换到 body_bbox"},
    {"type": "retry_original", "description": "重试原始 bbox"},
]


def validate_image_quality(image_base64: str) -> ValidationResult:
    """校验图像质量
    
    Args:
        image_base64: Base64 编码的图像
        
    Returns:
        ValidationResult: 校验结果
    """
    if not image_base64:
        return ValidationResult(
            passed=False,
            error_message="Empty image data"
        )
    
    try:
        import base64 as _b64
        # 去掉 data URL 前缀（如果有）
        raw_b64 = image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        # base64 解码为图像字节
        image_bytes = _b64.b64decode(raw_b64)
        
        img = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        
        # 1. 尺寸检测
        checks = {}
        details = {}
        
        if width < VALIDATION_RULES["min_width"]:
            checks["min_width"] = False
            details["width"] = width
        else:
            checks["min_width"] = True
        
        if height < VALIDATION_RULES["min_height"]:
            checks["min_height"] = False
            details["height"] = height
        else:
            checks["min_height"] = True
        
        # 2. 宽高比检测
        aspect_ratio = width / height if height > 0 else 0
        details["aspect_ratio"] = aspect_ratio
        
        if aspect_ratio < VALIDATION_RULES["min_aspect_ratio"]:
            checks["aspect_ratio"] = False
        elif aspect_ratio > VALIDATION_RULES["max_aspect_ratio"]:
            checks["aspect_ratio"] = False
        else:
            checks["aspect_ratio"] = True
        
        # 3. 亮度检测
        img_array = np.array(img)
        avg_brightness = img_array.mean()
        details["avg_brightness"] = float(avg_brightness)
        
        if avg_brightness < VALIDATION_RULES["min_brightness"]:
            checks["too_dark"] = True  # 标记为异常
        else:
            checks["too_dark"] = False
        
        if avg_brightness > VALIDATION_RULES["max_brightness"]:
            checks["too_white"] = True  # 标记为异常
        else:
            checks["too_white"] = False
        
        # 判断是否通过
        passed = all([
            checks.get("min_width", False),
            checks.get("min_height", False),
            checks.get("aspect_ratio", False),
            not checks.get("too_dark", True),
            not checks.get("too_white", True),
        ])
        
        # 收集错误信息
        errors = []
        if not checks.get("min_width"):
            errors.append(f"Width too small: {width}")
        if not checks.get("min_height"):
            errors.append(f"Height too small: {height}")
        if not checks.get("aspect_ratio"):
            errors.append(f"Abnormal aspect ratio: {aspect_ratio}")
        if checks.get("too_dark"):
            errors.append(f"Image too dark: {avg_brightness}")
        if checks.get("too_white"):
            errors.append(f"Image too white: {avg_brightness}")
        
        return ValidationResult(
            passed=passed,
            checks=checks,
            details=details,
            error_message="; ".join(errors) if errors else None
        )
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return ValidationResult(
            passed=False,
            error_message=f"Validation error: {str(e)}"
        )


def apply_fallback(
    figure: LogicalFigureSchema,
    strategy: Dict,
    pdf_doc,
    render_func
) -> Tuple[Optional[RenderResult], str]:
    """应用 fallback 策略
    
    Args:
        figure: LogicalFigureSchema
        strategy: Fallback 策略配置
        pdf_doc: PyMuPDF 文档
        render_func: 渲染函数
        
    Returns:
        (RenderResult 或 None, 应用的策略描述)
    """
    strategy_type = strategy.get("type")
    
    try:
        if strategy_type == "padding":
            padding = strategy.get("value", 10)
            result = render_func(pdf_doc, figure, padding=padding)
            return result, f"padding={padding}"
        
        elif strategy_type == "switch_bbox":
            target = strategy.get("target")
            if target == "full" and figure.full_bbox_page_pts:
                # 临时交换 bbox
                original = figure.body_bbox_page_pts
                figure.body_bbox_page_pts = figure.full_bbox_page_pts
                figure.full_bbox_page_pts = original
                result = render_func(pdf_doc, figure)
                # 恢复
                figure.full_bbox_page_pts = figure.body_bbox_page_pts
                figure.body_bbox_page_pts = original
                return result, "switched to full_bbox"
            
            elif target == "body" and figure.body_bbox_page_pts:
                # 临时交换
                original = figure.full_bbox_page_pts
                figure.full_bbox_page_pts = figure.body_bbox_page_pts
                result = render_func(pdf_doc, figure)
                # 恢复
                figure.full_bbox_page_pts = original
                return result, "switched to body_bbox"
        
        elif strategy_type == "retry_original":
            result = render_func(pdf_doc, figure)
            return result, "retry_original"
    
    except Exception as e:
        logger.warning(f"Fallback failed: {strategy_type}, error: {e}")
    
    return None, ""


def validate_and_fallback(
    figure: LogicalFigureSchema,
    pdf_doc,
    render_func
) -> RenderResult:
    """验证图像质量，如果失败则应用 fallback
    
    Args:
        figure: LogicalFigureSchema
        pdf_doc: PyMuPDF 文档
        render_func: 渲染函数 (pdf_doc, figure, padding) -> RenderResult
        
    Returns:
        RenderResult: 包含验证状态
    """
    # 首次渲染
    result = render_func(pdf_doc, figure)
    
    if not result.success:
        # 渲染失败，尝试 fallback
        for strategy in FALLBACK_STRATEGIES:
            fallback_result, strategy_desc = apply_fallback(
                figure, strategy, pdf_doc, render_func
            )
            if fallback_result and fallback_result.success:
                # 验证 fallback 结果
                if fallback_result.display_image_base64:
                    validation = validate_image_quality(fallback_result.display_image_base64)
                    if validation.passed:
                        fallback_result.fallback_applied = strategy_desc
                        fallback_result.validation_status = {
                            "passed": True,
                            "fallback": strategy_desc,
                            "checks": validation.checks
                        }
                        logger.info(f"Figure {figure.figure_id}: fallback succeeded with {strategy_desc}")
                        return fallback_result
    
    # 验证首次渲染结果
    if result.success and result.display_image_base64:
        validation = validate_image_quality(result.display_image_base64)
        result.validation_status = {
            "passed": validation.passed,
            "checks": validation.checks,
            "details": validation.details,
            "error": validation.error_message
        }
        
        if not validation.passed:
            # 验证失败，尝试 fallback
            for strategy in FALLBACK_STRATEGIES:
                fallback_result, strategy_desc = apply_fallback(
                    figure, strategy, pdf_doc, render_func
                )
                if fallback_result and fallback_result.success:
                    # 验证 fallback 结果
                    if fallback_result.display_image_base64:
                        fallback_validation = validate_image_quality(fallback_result.display_image_base64)
                        if fallback_validation.passed:
                            fallback_result.fallback_applied = strategy_desc
                            fallback_result.validation_status = {
                                "passed": True,
                                "fallback": strategy_desc,
                                "checks": fallback_validation.checks
                            }
                            logger.info(f"Figure {figure.figure_id}: fallback succeeded with {strategy_desc}")
                            return fallback_result
    
    return result


__all__ = [
    "validate_image_quality",
    "validate_and_fallback",
    "apply_fallback",
    "VALIDATION_RULES",
    "FALLBACK_STRATEGIES",
]
