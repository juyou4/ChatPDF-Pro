"""chat_service 关键兼容逻辑测试"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.chat_service import extract_reasoning_content


def test_extract_reasoning_content_supports_alternative_provider_fields():
    """兼容不同 provider 的思考字段命名。"""
    assert extract_reasoning_content({"reasoning": "步骤一"}) == "步骤一"
    assert extract_reasoning_content({"thinking": "步骤二"}) == "步骤二"
    assert extract_reasoning_content({"reasoning_text": "步骤三"}) == "步骤三"
    assert extract_reasoning_content({"thinking_text": "步骤四"}) == "步骤四"
