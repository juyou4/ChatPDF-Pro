"""
提示词池服务 - 参考 paper-burner-x 的 prompt-pool.js 实现
支持多提示词轮换、健康监控、自动降级
"""
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum


class PromptStatus(str, Enum):
    """提示词健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    DEACTIVATED = "deactivated"


@dataclass
class HealthStatus:
    """健康状态信息"""
    status: PromptStatus = PromptStatus.HEALTHY
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_used: Optional[str] = None
    last_success: Optional[str] = None
    last_failure: Optional[str] = None
    deactivated_at: Optional[str] = None
    deactivation_reason: Optional[str] = None
    average_response_time: float = 0.0
    request_history: List[Dict] = field(default_factory=list)


@dataclass
class PromptVariant:
    """提示词变体"""
    id: str
    name: str
    system_prompt: str
    user_prompt_template: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    is_active: bool = False
    user_selected: bool = False
    ai_generated: bool = False
    health_status: HealthStatus = field(default_factory=HealthStatus)


@dataclass
class HealthConfig:
    """健康管理配置"""
    max_consecutive_failures: int = 2
    deactivation_enabled: bool = True
    resurrection_time_minutes: int = 15
    resurrection_enabled: bool = True
    switch_on_failure: bool = True


class PromptPoolService:
    """提示词池管理服务"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(__file__).parent.parent / "data" / "prompt_pool"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.prompts: Dict[str, PromptVariant] = {}
        self.health_config = HealthConfig()
        self._load_prompts()
        self._load_config()
    
    def _load_prompts(self):
        """加载提示词"""
        prompts_file = self.storage_path / "prompts.json"
        if not prompts_file.exists():
            return
        
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for prompt_data in data.get("prompts", []):
                health_data = prompt_data.get("health_status", {})
                health_status = HealthStatus(
                    status=PromptStatus(health_data.get("status", "healthy")),
                    total_requests=health_data.get("total_requests", 0),
                    success_count=health_data.get("success_count", 0),
                    failure_count=health_data.get("failure_count", 0),
                    consecutive_failures=health_data.get("consecutive_failures", 0),
                    last_used=health_data.get("last_used"),
                    last_success=health_data.get("last_success"),
                    last_failure=health_data.get("last_failure"),
                    deactivated_at=health_data.get("deactivated_at"),
                    deactivation_reason=health_data.get("deactivation_reason"),
                    average_response_time=health_data.get("average_response_time", 0.0),
                    request_history=health_data.get("request_history", [])
                )
                
                prompt = PromptVariant(
                    id=prompt_data["id"],
                    name=prompt_data["name"],
                    system_prompt=prompt_data["system_prompt"],
                    user_prompt_template=prompt_data["user_prompt_template"],
                    description=prompt_data.get("description", ""),
                    category=prompt_data.get("category", "general"),
                    tags=prompt_data.get("tags", []),
                    created_at=prompt_data.get("created_at", datetime.now().isoformat()),
                    usage_count=prompt_data.get("usage_count", 0),
                    is_active=prompt_data.get("is_active", False),
                    user_selected=prompt_data.get("user_selected", False),
                    ai_generated=prompt_data.get("ai_generated", False),
                    health_status=health_status
                )
                self.prompts[prompt.id] = prompt
                
        except Exception as e:
            print(f"[PromptPool] Failed to load prompts: {e}")
    
    def _load_config(self):
        """加载配置"""
        config_file = self.storage_path / "config.json"
        if not config_file.exists():
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.health_config = HealthConfig(
                max_consecutive_failures=data.get("max_consecutive_failures", 2),
                deactivation_enabled=data.get("deactivation_enabled", True),
                resurrection_time_minutes=data.get("resurrection_time_minutes", 15),
                resurrection_enabled=data.get("resurrection_enabled", True),
                switch_on_failure=data.get("switch_on_failure", True)
            )
        except Exception as e:
            print(f"[PromptPool] Failed to load config: {e}")
    
    def _save_prompts(self):
        """保存提示词"""
        try:
            data = {
                "prompts": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "system_prompt": p.system_prompt,
                        "user_prompt_template": p.user_prompt_template,
                        "description": p.description,
                        "category": p.category,
                        "tags": p.tags,
                        "created_at": p.created_at,
                        "usage_count": p.usage_count,
                        "is_active": p.is_active,
                        "user_selected": p.user_selected,
                        "ai_generated": p.ai_generated,
                        "health_status": {
                            "status": p.health_status.status.value,
                            "total_requests": p.health_status.total_requests,
                            "success_count": p.health_status.success_count,
                            "failure_count": p.health_status.failure_count,
                            "consecutive_failures": p.health_status.consecutive_failures,
                            "last_used": p.health_status.last_used,
                            "last_success": p.health_status.last_success,
                            "last_failure": p.health_status.last_failure,
                            "deactivated_at": p.health_status.deactivated_at,
                            "deactivation_reason": p.health_status.deactivation_reason,
                            "average_response_time": p.health_status.average_response_time,
                            "request_history": p.health_status.request_history[-20:]
                        }
                    }
                    for p in self.prompts.values()
                ]
            }
            
            with open(self.storage_path / "prompts.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"[PromptPool] Failed to save prompts: {e}")
    
    def _save_config(self):
        """保存配置"""
        try:
            data = asdict(self.health_config)
            with open(self.storage_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[PromptPool] Failed to save config: {e}")
    
    def add_prompt(self, name: str, system_prompt: str, user_prompt_template: str,
                   description: str = "", category: str = "general",
                   tags: List[str] = None, ai_generated: bool = False) -> PromptVariant:
        """添加提示词"""
        import uuid
        prompt_id = str(uuid.uuid4())[:8]
        
        prompt = PromptVariant(
            id=prompt_id,
            name=name,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            description=description,
            category=category,
            tags=tags or [],
            ai_generated=ai_generated
        )
        
        self.prompts[prompt_id] = prompt
        self._save_prompts()
        return prompt
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptVariant]:
        """获取提示词"""
        return self.prompts.get(prompt_id)
    
    def get_all_prompts(self) -> List[Dict]:
        """获取所有提示词"""
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "category": p.category,
                "tags": p.tags,
                "usage_count": p.usage_count,
                "is_active": p.is_active,
                "user_selected": p.user_selected,
                "health_status": p.health_status.status.value,
                "success_rate": (p.health_status.success_count / p.health_status.total_requests * 100
                                if p.health_status.total_requests > 0 else 100)
            }
            for p in self.prompts.values()
        ]
    
    def select_healthy_prompt(self, exclude_id: str = None) -> Optional[PromptVariant]:
        """
        选择一个健康的提示词
        优先选择用户选中的、健康状态好的
        """
        # 筛选可用的提示词
        available = [
            p for p in self.prompts.values()
            if p.id != exclude_id
            and p.user_selected
            and p.health_status.status in [PromptStatus.HEALTHY, PromptStatus.DEGRADED]
        ]
        
        if not available:
            # 如果没有用户选中的，选择任意健康的
            available = [
                p for p in self.prompts.values()
                if p.id != exclude_id
                and p.health_status.status in [PromptStatus.HEALTHY, PromptStatus.DEGRADED]
            ]
        
        if not available:
            return None
        
        # 优先选择完全健康的
        healthy = [p for p in available if p.health_status.status == PromptStatus.HEALTHY]
        if healthy:
            # 按成功率排序
            healthy.sort(
                key=lambda p: (
                    p.health_status.success_count / p.health_status.total_requests
                    if p.health_status.total_requests > 0 else 1
                ),
                reverse=True
            )
            return healthy[0]
        
        # 选择降级状态中最好的
        available.sort(key=lambda p: p.health_status.success_count, reverse=True)
        return available[0]
    
    def record_usage(self, prompt_id: str, success: bool, 
                     response_time: float = 0, error: str = None):
        """
        记录提示词使用结果
        
        Args:
            prompt_id: 提示词ID
            success: 是否成功
            response_time: 响应时间(秒)
            error: 错误信息
        """
        if prompt_id not in self.prompts:
            return
        
        prompt = self.prompts[prompt_id]
        health = prompt.health_status
        now = datetime.now().isoformat()
        
        # 更新统计
        health.total_requests += 1
        health.last_used = now
        prompt.usage_count += 1
        
        # 记录请求历史
        record = {
            "timestamp": now,
            "success": success,
            "response_time": response_time,
            "error": error
        }
        health.request_history.append(record)
        if len(health.request_history) > 20:
            health.request_history = health.request_history[-20:]
        
        if success:
            health.success_count += 1
            health.consecutive_failures = 0
            health.last_success = now
            
            # 更新平均响应时间
            if response_time > 0:
                recent_times = [
                    r["response_time"] for r in health.request_history
                    if r.get("success") and r.get("response_time", 0) > 0
                ]
                if recent_times:
                    health.average_response_time = sum(recent_times) / len(recent_times)
            
            # 恢复健康状态
            if health.status == PromptStatus.DEGRADED:
                health.status = PromptStatus.HEALTHY
                print(f"[PromptPool] Prompt {prompt.name} recovered to healthy")
        else:
            health.failure_count += 1
            health.consecutive_failures += 1
            health.last_failure = now
            
            print(f"[PromptPool] Prompt {prompt.name} failed, consecutive: {health.consecutive_failures}")
            
            # 更新健康状态
            self._update_health_status(prompt_id)
        
        self._save_prompts()
    
    def _update_health_status(self, prompt_id: str):
        """更新提示词健康状态"""
        prompt = self.prompts.get(prompt_id)
        if not prompt:
            return
        
        health = prompt.health_status
        config = self.health_config
        
        # 检查是否需要失活
        if (config.deactivation_enabled and 
            health.consecutive_failures >= config.max_consecutive_failures):
            
            health.status = PromptStatus.DEACTIVATED
            health.deactivated_at = datetime.now().isoformat()
            health.deactivation_reason = f"连续失败{health.consecutive_failures}次"
            prompt.is_active = False
            
            print(f"[PromptPool] Prompt {prompt.name} deactivated: {health.deactivation_reason}")
            
        elif health.consecutive_failures >= config.max_consecutive_failures // 2:
            health.status = PromptStatus.DEGRADED
            print(f"[PromptPool] Prompt {prompt.name} degraded")
    
    def check_resurrection(self):
        """检查并复活失活的提示词"""
        if not self.health_config.resurrection_enabled:
            return
        
        now = datetime.now()
        resurrection_delta = timedelta(minutes=self.health_config.resurrection_time_minutes)
        
        for prompt in self.prompts.values():
            if prompt.health_status.status != PromptStatus.DEACTIVATED:
                continue
            
            if not prompt.health_status.deactivated_at:
                continue
            
            deactivated_time = datetime.fromisoformat(prompt.health_status.deactivated_at)
            if now - deactivated_time >= resurrection_delta:
                # 复活
                prompt.health_status.status = PromptStatus.DEGRADED
                prompt.health_status.consecutive_failures = 0
                prompt.health_status.deactivated_at = None
                prompt.health_status.deactivation_reason = None
                
                print(f"[PromptPool] Prompt {prompt.name} resurrected")
        
        self._save_prompts()
    
    def toggle_prompt(self, prompt_id: str, selected: bool) -> bool:
        """切换提示词选中状态"""
        if prompt_id not in self.prompts:
            return False
        
        self.prompts[prompt_id].user_selected = selected
        self._save_prompts()
        return True
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """删除提示词"""
        if prompt_id not in self.prompts:
            return False
        
        del self.prompts[prompt_id]
        self._save_prompts()
        return True
    
    def get_default_prompts(self) -> List[Dict]:
        """获取默认提示词模板"""
        return [
            {
                "name": "通用翻译",
                "system_prompt": "你是专业的文档翻译助手。请准确翻译用户提供的内容，保持原文的格式和结构。",
                "user_prompt_template": "请将以下内容翻译成${targetLang}：\n\n${content}",
                "category": "general",
                "description": "通用翻译提示词，适用于大多数场景"
            },
            {
                "name": "学术论文翻译",
                "system_prompt": "你是专业的学术论文翻译助手。请准确翻译学术内容，保持专业术语的准确性，保留公式、引用格式。",
                "user_prompt_template": "请将以下学术内容翻译成${targetLang}，注意保持专业术语准确：\n\n${content}",
                "category": "academic",
                "description": "适用于学术论文、研究报告的翻译"
            },
            {
                "name": "技术文档翻译",
                "system_prompt": "你是专业的技术文档翻译助手。请准确翻译技术内容，保持代码块、命令、API名称不变。",
                "user_prompt_template": "请将以下技术文档翻译成${targetLang}，保持代码和技术术语不变：\n\n${content}",
                "category": "technical",
                "description": "适用于技术文档、API文档的翻译"
            }
        ]


# 全局实例
prompt_pool_service = PromptPoolService()


def get_healthy_prompt(exclude_id: str = None) -> Optional[Dict]:
    """获取一个健康的提示词"""
    prompt = prompt_pool_service.select_healthy_prompt(exclude_id)
    if prompt:
        return {
            "id": prompt.id,
            "name": prompt.name,
            "system_prompt": prompt.system_prompt,
            "user_prompt_template": prompt.user_prompt_template
        }
    return None


def record_prompt_usage(prompt_id: str, success: bool, response_time: float = 0, error: str = None):
    """记录提示词使用"""
    prompt_pool_service.record_usage(prompt_id, success, response_time, error)
