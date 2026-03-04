"""用户反馈 API

提供结构化反馈收集接口，支持正面/负面反馈及问题分类。
反馈存储为 JSON 文件，按日期归档。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# 反馈存储目录，由 app.py 注入
_feedback_dir: Path = Path("data/feedback")


class FeedbackRequest(BaseModel):
    """反馈请求体"""
    doc_id: str
    message_idx: int
    feedback_type: str  # "like" | "dislike" | "report"
    # 以下为 dislike/report 时可选
    issue_types: Optional[List[str]] = None  # ["wrong_answer", "wrong_citation", "irrelevant", "offensive"]
    detail: Optional[str] = None
    # 上下文快照
    question: Optional[str] = None
    answer: Optional[str] = None
    model: Optional[str] = None


def init_feedback_dir(data_dir: Path):
    """初始化反馈存储目录"""
    global _feedback_dir
    _feedback_dir = data_dir / "feedback"
    _feedback_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"反馈存储目录: {_feedback_dir}")


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """保存用户反馈"""
    try:
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": request.doc_id,
            "message_idx": request.message_idx,
            "feedback_type": request.feedback_type,
            "issue_types": request.issue_types or [],
            "detail": request.detail or "",
            "question": request.question or "",
            "answer": (request.answer or "")[:500],
            "model": request.model or "",
        }

        # 按日期归档
        date_str = datetime.now().strftime("%Y-%m-%d")
        feedback_file = _feedback_dir / f"feedback_{date_str}.jsonl"

        _feedback_dir.mkdir(parents=True, exist_ok=True)
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")

        logger.info(
            f"[Feedback] {request.feedback_type} for doc={request.doc_id} "
            f"msg={request.message_idx} issues={request.issue_types}"
        )
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"[Feedback] 保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"反馈保存失败: {str(e)}")


@router.get("/feedback/stats")
async def get_feedback_stats():
    """获取反馈统计（可选）"""
    try:
        total = 0
        likes = 0
        dislikes = 0
        for f in _feedback_dir.glob("feedback_*.jsonl"):
            for line in open(f, encoding="utf-8"):
                try:
                    fb = json.loads(line.strip())
                    total += 1
                    if fb.get("feedback_type") == "like":
                        likes += 1
                    elif fb.get("feedback_type") in ("dislike", "report"):
                        dislikes += 1
                except json.JSONDecodeError:
                    pass
        return {"total": total, "likes": likes, "dislikes": dislikes}
    except Exception:
        return {"total": 0, "likes": 0, "dislikes": 0}
