"""
术语库服务 - 参考 paper-burner-x 的 glossary-matcher.js 实现
支持专业术语的一致性翻译
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class GlossaryEntry:
    """术语条目"""
    term: str                    # 原文术语
    translation: str             # 翻译
    case_sensitive: bool = False # 是否区分大小写
    whole_word: bool = True      # 是否全词匹配
    enabled: bool = True         # 是否启用
    category: str = "general"    # 分类
    notes: str = ""              # 备注


@dataclass
class GlossarySet:
    """术语集"""
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    entries: List[GlossaryEntry] = field(default_factory=list)
    max_terms_in_prompt: int = 50  # 单次提示词中最多包含的术语数


class TrieNode:
    """Trie 树节点"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.entries: List[GlossaryEntry] = []


class GlossaryMatcher:
    """
    高效多模式术语匹配器
    使用 Trie 树实现，时间复杂度 O(L + m)
    L = 文本长度，m = 匹配数
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.case_insensitive_root = TrieNode()
        self._version = 0
    
    def build(self, entries: List[GlossaryEntry]):
        """构建 Trie 树"""
        self.root = TrieNode()
        self.case_insensitive_root = TrieNode()
        
        for entry in entries:
            if not entry.term or not entry.translation or not entry.enabled:
                continue
            
            root = self.root if entry.case_sensitive else self.case_insensitive_root
            term = entry.term if entry.case_sensitive else entry.term.lower()
            
            node = root
            for char in term:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            
            node.entries.append(entry)
        
        self._version += 1
    
    def _is_word_boundary(self, text: str, index: int) -> bool:
        """检查是否为单词边界"""
        if index < 0 or index >= len(text):
            return True
        char = text[index]
        return not (char.isalnum() or char == '_')
    
    def _has_ascii_word(self, s: str) -> bool:
        """检查是否包含 ASCII 单词字符"""
        return bool(re.search(r'[A-Za-z0-9_]', s))
    
    def find_matches(self, text: str) -> List[Dict[str, str]]:
        """
        在文本中查找所有匹配的术语
        
        Args:
            text: 要搜索的文本
            
        Returns:
            匹配的术语列表 [{"term": ..., "translation": ...}, ...]
        """
        if not text:
            return []
        
        matches = []
        seen = set()
        text_lower = text.lower()
        
        for i in range(len(text)):
            # 大小写敏感匹配
            self._search_from_position(text, i, self.root, True, matches, seen)
            # 大小写不敏感匹配
            self._search_from_position(text_lower, i, self.case_insensitive_root, False, matches, seen)
        
        return matches
    
    def _search_from_position(self, text: str, start_pos: int, root: TrieNode, 
                               case_sensitive: bool, matches: List, seen: set):
        """从指定位置开始搜索"""
        node = root
        pos = start_pos
        
        while pos < len(text):
            char = text[pos]
            
            if char not in node.children:
                break
            
            node = node.children[char]
            pos += 1
            
            # 检查当前节点是否有完整术语
            for entry in node.entries:
                if entry.case_sensitive != case_sensitive:
                    continue
                
                # 全词匹配检查
                if entry.whole_word and self._has_ascii_word(entry.term):
                    if not self._is_word_boundary(text, start_pos - 1):
                        continue
                    if not self._is_word_boundary(text, pos):
                        continue
                
                key = f"{entry.term}=>{entry.translation}"
                if key not in seen:
                    matches.append({
                        "term": entry.term,
                        "translation": entry.translation
                    })
                    seen.add(key)
    
    def has_any_match(self, text: str) -> bool:
        """快速检查文本是否包含任何术语"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        for i in range(len(text)):
            if self._has_match_at_position(text, i, self.root, True):
                return True
            if self._has_match_at_position(text_lower, i, self.case_insensitive_root, False):
                return True
        
        return False
    
    def _has_match_at_position(self, text: str, start_pos: int, root: TrieNode, 
                                case_sensitive: bool) -> bool:
        """检查指定位置是否有匹配"""
        node = root
        pos = start_pos
        
        while pos < len(text):
            char = text[pos]
            if char not in node.children:
                break
            
            node = node.children[char]
            pos += 1
            
            for entry in node.entries:
                if entry.case_sensitive == case_sensitive:
                    if not entry.whole_word:
                        return True
                    if self._has_ascii_word(entry.term):
                        if (self._is_word_boundary(text, start_pos - 1) and 
                            self._is_word_boundary(text, pos)):
                            return True
                    else:
                        return True
        
        return False


class GlossaryService:
    """术语库服务"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(__file__).parent.parent / "data" / "glossary"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.glossary_sets: Dict[str, GlossarySet] = {}
        self.matcher = GlossaryMatcher()
        self._load_glossary_sets()
    
    def _load_glossary_sets(self):
        """从磁盘加载术语集"""
        index_file = self.storage_path / "index.json"
        if not index_file.exists():
            return
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            for set_id in index.get("sets", []):
                set_file = self.storage_path / f"{set_id}.json"
                if set_file.exists():
                    with open(set_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    entries = [GlossaryEntry(**e) for e in data.get("entries", [])]
                    self.glossary_sets[set_id] = GlossarySet(
                        id=data["id"],
                        name=data["name"],
                        description=data.get("description", ""),
                        enabled=data.get("enabled", True),
                        entries=entries,
                        max_terms_in_prompt=data.get("max_terms_in_prompt", 50)
                    )
            
            self._rebuild_matcher()
            
        except Exception as e:
            print(f"[Glossary] Failed to load glossary sets: {e}")
    
    def _save_glossary_sets(self):
        """保存术语集到磁盘"""
        try:
            # 保存索引
            index = {"sets": list(self.glossary_sets.keys())}
            with open(self.storage_path / "index.json", 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            
            # 保存每个术语集
            for set_id, glossary_set in self.glossary_sets.items():
                data = {
                    "id": glossary_set.id,
                    "name": glossary_set.name,
                    "description": glossary_set.description,
                    "enabled": glossary_set.enabled,
                    "max_terms_in_prompt": glossary_set.max_terms_in_prompt,
                    "entries": [asdict(e) for e in glossary_set.entries]
                }
                with open(self.storage_path / f"{set_id}.json", 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            print(f"[Glossary] Failed to save glossary sets: {e}")
    
    def _rebuild_matcher(self):
        """重建匹配器"""
        all_entries = []
        for glossary_set in self.glossary_sets.values():
            if glossary_set.enabled:
                all_entries.extend(glossary_set.entries)
        self.matcher.build(all_entries)
    
    def create_glossary_set(self, name: str, description: str = "") -> GlossarySet:
        """创建新的术语集"""
        import uuid
        set_id = str(uuid.uuid4())[:8]
        
        glossary_set = GlossarySet(
            id=set_id,
            name=name,
            description=description
        )
        self.glossary_sets[set_id] = glossary_set
        self._save_glossary_sets()
        return glossary_set
    
    def add_entry(self, set_id: str, term: str, translation: str, 
                  case_sensitive: bool = False, whole_word: bool = True,
                  category: str = "general", notes: str = "") -> bool:
        """添加术语条目"""
        if set_id not in self.glossary_sets:
            return False
        
        entry = GlossaryEntry(
            term=term,
            translation=translation,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            category=category,
            notes=notes
        )
        self.glossary_sets[set_id].entries.append(entry)
        self._save_glossary_sets()
        self._rebuild_matcher()
        return True
    
    def import_entries(self, set_id: str, entries: List[Dict]) -> int:
        """批量导入术语"""
        if set_id not in self.glossary_sets:
            return 0
        
        count = 0
        for entry_data in entries:
            if "term" in entry_data and "translation" in entry_data:
                entry = GlossaryEntry(
                    term=entry_data["term"],
                    translation=entry_data["translation"],
                    case_sensitive=entry_data.get("case_sensitive", False),
                    whole_word=entry_data.get("whole_word", True),
                    category=entry_data.get("category", "general"),
                    notes=entry_data.get("notes", "")
                )
                self.glossary_sets[set_id].entries.append(entry)
                count += 1
        
        if count > 0:
            self._save_glossary_sets()
            self._rebuild_matcher()
        
        return count
    
    def find_matches(self, text: str) -> List[Dict[str, str]]:
        """在文本中查找匹配的术语"""
        return self.matcher.find_matches(text)
    
    def build_glossary_instruction(self, matches: List[Dict[str, str]], 
                                    target_lang: str = "中文",
                                    max_terms: int = 50) -> str:
        """
        构建术语库提示词指令
        
        Args:
            matches: 匹配的术语列表
            target_lang: 目标语言
            max_terms: 最大术语数
            
        Returns:
            提示词指令字符串
        """
        if not matches:
            return ""
        
        # 限制术语数量
        limited_matches = matches[:max_terms]
        
        # 构建术语表
        terms_list = "\n".join([
            f"- {m['term']} → {m['translation']}"
            for m in limited_matches
        ])
        
        instruction = f"""【术语表】
以下术语在翻译时请使用指定的译法，保持一致性：

{terms_list}

请在翻译过程中严格遵循上述术语表。"""
        
        return instruction
    
    def get_all_sets(self) -> List[Dict]:
        """获取所有术语集信息"""
        return [
            {
                "id": gs.id,
                "name": gs.name,
                "description": gs.description,
                "enabled": gs.enabled,
                "entry_count": len(gs.entries)
            }
            for gs in self.glossary_sets.values()
        ]
    
    def get_set_entries(self, set_id: str) -> List[Dict]:
        """获取术语集的所有条目"""
        if set_id not in self.glossary_sets:
            return []
        return [asdict(e) for e in self.glossary_sets[set_id].entries]
    
    def toggle_set(self, set_id: str, enabled: bool) -> bool:
        """启用/禁用术语集"""
        if set_id not in self.glossary_sets:
            return False
        self.glossary_sets[set_id].enabled = enabled
        self._save_glossary_sets()
        self._rebuild_matcher()
        return True
    
    def delete_set(self, set_id: str) -> bool:
        """删除术语集"""
        if set_id not in self.glossary_sets:
            return False
        
        del self.glossary_sets[set_id]
        
        # 删除文件
        set_file = self.storage_path / f"{set_id}.json"
        if set_file.exists():
            set_file.unlink()
        
        self._save_glossary_sets()
        self._rebuild_matcher()
        return True


# 全局实例
glossary_service = GlossaryService()


def get_glossary_matches(text: str) -> List[Dict[str, str]]:
    """获取文本中的术语匹配"""
    return glossary_service.find_matches(text)


def build_glossary_prompt(text: str, target_lang: str = "中文") -> str:
    """为文本构建术语库提示词"""
    matches = glossary_service.find_matches(text)
    if not matches:
        return ""
    return glossary_service.build_glossary_instruction(matches, target_lang)
