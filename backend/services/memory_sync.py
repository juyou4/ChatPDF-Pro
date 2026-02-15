"""
记忆文件监听和增量同步服务

监听 Markdown 记忆文件的变化，自动触发索引更新。
使用 watchdog 库监听文件系统事件，debounce 机制避免频繁更新。
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog 库未安装，文件监听功能将不可用。安装: pip install watchdog")


class MemoryFileHandler(FileSystemEventHandler):
    """记忆文件变化处理器"""
    
    def __init__(self, memory_service, debounce_seconds: float = 1.5):
        """
        初始化文件处理器
        
        Args:
            memory_service: MemoryService 实例
            debounce_seconds: debounce 延迟（秒），默认 1.5 秒
        """
        self.memory_service = memory_service
        self.debounce_seconds = debounce_seconds
        self._pending_files = set()
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        # 只处理 Markdown 文件
        if not event.src_path.endswith('.md'):
            return
        
        # 添加到待处理集合
        with self._lock:
            self._pending_files.add(event.src_path)
            
            # 取消之前的定时器
            if self._timer:
                self._timer.cancel()
            
            # 创建新的定时器（debounce）
            self._timer = threading.Timer(
                self.debounce_seconds,
                self._process_pending_files
            )
            self._timer.start()
    
    def _process_pending_files(self) -> None:
        """处理待处理的文件（debounce 后执行）"""
        with self._lock:
            if not self._pending_files:
                return
            
            files_to_process = list(self._pending_files)
            self._pending_files.clear()
            self._timer = None
        
        try:
            logger.info(f"检测到 {len(files_to_process)} 个 Markdown 文件变化，触发索引更新")
            # 触发索引重建（异步，不阻塞）
            threading.Thread(
                target=self._rebuild_index_async,
                daemon=True
            ).start()
        except Exception as e:
            logger.error(f"处理文件变化失败: {e}")
    
    def _rebuild_index_async(self) -> None:
        """异步重建索引"""
        try:
            # 获取所有记忆条目
            all_entries = self.memory_service.store.get_all_entries()
            
            # 重建向量索引
            self.memory_service.index.rebuild(all_entries)
            
            logger.info(f"索引重建完成，共 {len(all_entries)} 条记忆")
        except Exception as e:
            logger.error(f"异步重建索引失败: {e}")


class MemoryFileWatcher:
    """记忆文件监听器"""
    
    def __init__(self, memory_service, memory_dir: str):
        """
        初始化文件监听器
        
        Args:
            memory_service: MemoryService 实例
            memory_dir: Markdown 记忆文件目录
        """
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog 库未安装，无法启用文件监听")
        
        self.memory_service = memory_service
        self.memory_dir = Path(memory_dir)
        self.observer: Optional[Observer] = None
        self.handler: Optional[MemoryFileHandler] = None
    
    def start(self) -> None:
        """启动文件监听"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog 不可用，跳过文件监听启动")
            return
        
        if self.observer and self.observer.is_alive():
            logger.warning("文件监听器已在运行")
            return
        
        try:
            # 确保目录存在
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建处理器和观察者
            self.handler = MemoryFileHandler(self.memory_service)
            self.observer = Observer()
            self.observer.schedule(
                self.handler,
                str(self.memory_dir),
                recursive=False  # 不递归监听子目录
            )
            
            # 启动监听
            self.observer.start()
            logger.info(f"记忆文件监听已启动: {self.memory_dir}")
        except Exception as e:
            logger.error(f"启动文件监听失败: {e}")
            raise
    
    def stop(self) -> None:
        """停止文件监听"""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5.0)
                logger.info("记忆文件监听已停止")
            except Exception as e:
                logger.error(f"停止文件监听失败: {e}")
            finally:
                self.observer = None
                self.handler = None
    
    def is_running(self) -> bool:
        """检查监听器是否在运行"""
        return self.observer is not None and self.observer.is_alive()
