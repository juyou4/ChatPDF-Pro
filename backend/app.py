"""
ChatPDF Pro - 支持截图功能的后端API
添加视觉模型支持：GPT-4V, Claude Vision, Gemini Vision
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import PyPDF2
import pdfplumber
import io
import os
import hashlib
from datetime import datetime
import httpx
import json
import base64

app = FastAPI(title="ChatPDF Pro with Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Mount static files for serving PDFs
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

documents_store = {}

class ChatRequest(BaseModel):
    doc_id: str
    question: str
    api_key: str
    model: str
    api_provider: str
    selected_text: Optional[str] = None

class ChatVisionRequest(BaseModel):
    """支持截图的对话请求"""
    doc_id: str
    question: str
    api_key: str
    model: str
    api_provider: str
    image_base64: Optional[str] = None  # 截图的base64编码
    selected_text: Optional[str] = None

class SummaryRequest(BaseModel):
    doc_id: str
    api_key: str
    model: str
    api_provider: str

# 支持视觉的模型配置
VISION_MODELS = {
    "openai": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
    "anthropic": [
        "claude-3-opus-20240229", 
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307",
        "claude-sonnet-4-5-20250929"
    ],
    "gemini": [
        "gemini-pro-vision", 
        "gemini-2.5-pro", 
        "gemini-2.5-flash-preview-09-2025"
    ],
    "grok": ["grok-4.1", "grok-vision-beta"]
}

# AI模型配置
AI_MODELS = {
    "openai": {
        "name": "OpenAI GPT",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "models": {
            "gpt-4o": "GPT-4o (视觉)",
            "gpt-4-turbo": "GPT-4 Turbo (视觉)",
            "gpt-4o-mini": "GPT-4o Mini (视觉)",
            "gpt-3.5-turbo": "GPT-3.5 Turbo"
        }
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "models": {
            "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 (视觉)",
            "claude-3-opus-20240229": "Claude 3 Opus (视觉)",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet (视觉)",
            "claude-3-haiku-20240307": "Claude 3 Haiku (视觉)"
        }
    },
    "gemini": {
        "name": "Google Gemini",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "models": {
            "gemini-2.5-pro": "Gemini 2.5 Pro (视觉)",
            "gemini-2.5-flash-preview-09-2025": "Gemini 2.5 Flash (视觉)",
            "gemini-pro-vision": "Gemini Pro Vision"
        }
    },
    "grok": {
        "name": "xAI Grok",
        "endpoint": "https://api.x.ai/v1/chat/completions",
        "models": {
            "grok-4.1": "Grok 4.1 (视觉)",
            "grok-vision-beta": "Grok Vision Beta"
        }
    },
    "deepseek": {
        "name": "DeepSeek",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "models": {
            "deepseek-chat": "DeepSeek Chat",
            "deepseek-v3.2-exp": "DeepSeek V3.2"
        }
    },
    "ollama": {
        "name": "Local (Ollama)",
        "endpoint": "http://localhost:11434/api/chat",
        "models": {
            "llama3": "Llama 3 (Local)",
            "mistral": "Mistral (Local)",
            "qwen2": "Qwen 2 (Local)",
            "llava": "LLaVA (Local Vision)"
        }
    }
}

# ==================== PDF处理 ====================

def extract_text_from_pdf(pdf_file) -> dict:
    """从PDF提取文本 - 使用pdfplumber提供更好的文本提取质量"""
    pages_content = []

    # Reset file pointer to beginning
    pdf_file.seek(0)

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract text with better layout handling
            text = page.extract_text(layout=True) or ""

            # Try to extract tables if they exist
            tables = page.extract_tables()
            table_text = ""
            if tables:
                table_text = "\n\n[表格内容]\n"
                for table in tables:
                    # Format table as markdown-like text
                    for row in table:
                        if row:
                            table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                    table_text += "\n"

            combined_text = text + table_text

            pages_content.append({
                "page": page_num + 1,
                "content": combined_text.strip(),
                "char_count": len(combined_text),
                "has_tables": len(tables) > 0 if tables else False
            })

    return {
        "total_pages": len(pages_content),
        "pages": pages_content,
        "full_text": "\n\n".join([p["content"] for p in pages_content])
    }

def generate_doc_id(content: str) -> str:
    """生成文档唯一ID"""
    return hashlib.md5(content.encode()).hexdigest()

# ==================== AI API调用 ====================

async def call_openai_api(messages: List[dict], api_key: str, model: str = "gpt-4o"):
    """调用OpenAI API (支持视觉)"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4000
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"OpenAI API错误: {response.text}"
            )
        
        return response.json()

async def call_anthropic_api(messages: List[dict], api_key: str, model: str):
    """调用Anthropic Claude API (支持视觉)"""
    system_message = ""
    user_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            user_messages.append(msg)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "max_tokens": 4000,
                "system": system_message,
                "messages": user_messages
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Anthropic API错误: {response.text}"
            )
        
        result = response.json()
        return {
            "choices": [{
                "message": {
                    "content": result["content"][0]["text"]
                }
            }]
        }

async def call_gemini_api(messages: List[dict], api_key: str, model: str):
    """调用Google Gemini API (支持视觉)"""
    # Gemini使用不同的API格式
    contents = []
    
    for msg in messages:
        if msg["role"] == "system":
            # 系统消息合并到第一个用户消息
            continue
            
        parts = []
        
        # 文本内容
        if isinstance(msg["content"], str):
            parts.append({"text": msg["content"]})
        elif isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "text":
                    parts.append({"text": item["text"]})
                elif item["type"] == "image_url":
                    # Gemini需要base64图片数据
                    image_data = item["image_url"]["url"].split(",")[1]
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data
                        }
                    })
        
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": parts
        })
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json={
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 4000
                }
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Gemini API错误: {response.text}"
            )
        
        result = response.json()
        return {
            "choices": [{
                "message": {
                    "content": result["candidates"][0]["content"]["parts"][0]["text"]
                }
            }]
        }

async def call_grok_api(messages: List[dict], api_key: str, model: str):
    """调用xAI Grok API (OpenAI兼容格式)"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4000
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Grok API错误: {response.text}"
            )
        
        return response.json()

async def call_ollama_api(messages: List[dict], model: str):
    """调用本地Ollama API"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Ollama API format
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json=payload
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama API错误: {response.text}"
                )
                
            result = response.json()
            return {
                "choices": [{
                    "message": {
                        "content": result["message"]["content"]
                    }
                }]
            }
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="无法连接到本地Ollama服务，请确保Ollama已启动 (localhost:11434)"
            )

async def call_ai_api(messages: List[dict], api_key: str, model: str, provider: str):
    """统一的AI API调用接口"""
    if provider == "openai":
        return await call_openai_api(messages, api_key, model)
    elif provider == "anthropic":
        return await call_anthropic_api(messages, api_key, model)
    elif provider == "gemini":
        return await call_gemini_api(messages, api_key, model)
    elif provider == "grok":
        return await call_grok_api(messages, api_key, model)
    elif provider == "ollama":
        return await call_ollama_api(messages, model)
    else:
        # 其他提供商使用OpenAI兼容格式
        endpoint = AI_MODELS.get(provider, {}).get("endpoint", "")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 4000
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"{provider} API错误: {response.text}"
                )
            
            return response.json()

# ==================== API端点 ====================

@app.get("/models")
async def get_models():
    """获取可用模型列表"""
    return AI_MODELS

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """与PDF文档对话（不带截图）"""
    if request.doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    doc = documents_store[request.doc_id]
    
    # 构建上下文
    context = ""
    if request.selected_text:
        context = f"用户选中的文本：\n{request.selected_text}\n\n"
    else:
        context = doc["data"]["full_text"][:8000]
    
    system_prompt = f"""你是专业的文档分析助手。用户上传了一份PDF文档。

文档名称：{doc["filename"]}
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

请根据文档内容准确回答用户的问题。如果文档中没有相关信息，请明确告知。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.question}
    ]
    
    try:
        response = await call_ai_api(messages, request.api_key, request.model, request.api_provider)
        answer = response["choices"][0]["message"]["content"]
        
        return {
            "answer": answer,
            "doc_id": request.doc_id,
            "question": request.question,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI调用失败: {str(e)}")

@app.post("/chat/stream")
async def chat_with_pdf_stream(request: ChatRequest):
    """与PDF文档对话（SSE流式响应）"""
    if request.doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    doc = documents_store[request.doc_id]
    
    # 构建上下文
    context = ""
    if request.selected_text:
        context = f"用户选中的文本：\n{request.selected_text}\n\n"
    else:
        context = doc["data"]["full_text"][:8000]
    
    system_prompt = f"""你是专业的文档分析助手。用户上传了一份PDF文档。

文档名称：{doc["filename"]}
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

请根据文档内容准确回答用户的问题。如果文档中没有相关信息，请明确告知。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.question}
    ]
    
    async def event_generator():
        try:
            response = await call_ai_api(messages, request.api_key, request.model, request.api_provider)
            answer = response["choices"][0]["message"]["content"]
            
            # 按单词流式发送
            words = answer.split(' ')
            for i, word in enumerate(words):
                chunk = word if i == 0 else f" {word}"
                yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
            
            # 发送完成标记
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/vision")
async def chat_with_vision(request: ChatVisionRequest):
    """支持截图的对话API"""
    if request.doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    doc = documents_store[request.doc_id]
    
    # 检查模型是否支持视觉
    if request.image_base64:
        supports_vision = request.model in VISION_MODELS.get(request.api_provider, [])
        if not supports_vision:
            raise HTTPException(
                status_code=400, 
                detail=f"模型 {request.model} 不支持图片输入"
            )
    
    # 构建上下文
    context = ""
    if request.selected_text:
        context = f"用户选中的文本：\n{request.selected_text}\n\n"
    else:
        context = doc["data"]["full_text"][:5000]
    
    # 构建提示词
    if request.image_base64:
        system_prompt = f"""你是专业的文档分析助手，具备视觉理解能力。

文档名称：{doc["filename"]}
文档总页数：{doc["data"]["total_pages"]}

文档部分内容：
{context}

用户提供了一张文档截图。请：
1. 仔细分析截图中的内容（文字、表格、图表、公式等）
2. 结合文档上下文回答用户的问题
3. 如果截图中有表格，准确提取数据
4. 如果截图中有图表，详细解释其含义
5. 如果截图中有公式，解释公式的含义和推导

回答要详细、准确、专业。"""
    else:
        system_prompt = f"""你是专业的文档分析助手。

文档名称：{doc["filename"]}
文档总页数：{doc["data"]["total_pages"]}

文档内容：
{context}

请根据文档内容回答用户的问题。"""
    
    # 构建消息
    if request.image_base64:
        # 有截图的情况
        if request.api_provider in ["openai", "grok"]:
            # OpenAI格式
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": request.question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{request.image_base64}"
                            }
                        }
                    ]
                }
            ]
        elif request.api_provider == "anthropic":
            # Claude格式
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": request.image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": request.question
                        }
                    ]
                }
            ]
        elif request.api_provider == "gemini":
            # Gemini格式（在call_gemini_api中处理）
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": request.question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{request.image_base64}"
                            }
                        }
                    ]
                }
            ]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"提供商 {request.api_provider} 暂不支持视觉功能"
            )
    else:
        # 无截图的普通对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ]
    
    try:
        response = await call_ai_api(
            messages, 
            request.api_key, 
            request.model, 
            request.api_provider
        )
        answer = response["choices"][0]["message"]["content"]
        
        return {
            "answer": answer,
            "doc_id": request.doc_id,
            "question": request.question,
            "has_image": bool(request.image_base64),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI调用失败: {str(e)}")

@app.post("/summary")
async def generate_summary(request: SummaryRequest):
    """生成文档摘要"""
    if request.doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    doc = documents_store[request.doc_id]
    full_text = doc["data"]["full_text"]
    
    system_prompt = """你是专业的文档摘要专家。请为文档生成简洁的摘要。

要求：
1. 提取文档的核心观点和关键信息
2. 摘要应简洁明了，长度在200-500字
3. 使用要点形式组织内容
4. 生成3-5个相关问题建议"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请为以下文档生成摘要：\n\n{full_text[:8000]}"}
    ]
    
    try:
        response = await call_ai_api(
            messages, 
            request.api_key, 
            request.model, 
            request.api_provider
        )
        summary = response["choices"][0]["message"]["content"]
        
        # 生成建议问题
        questions_messages = [
            {
                "role": "system", 
                "content": "根据文档内容，生成5个有价值的问题。只输出问题列表，每行一个问题。"
            },
            {"role": "user", "content": f"文档内容：\n{full_text[:4000]}"}
        ]
        
        questions_response = await call_ai_api(
            questions_messages, 
            request.api_key, 
            request.model, 
            request.api_provider
        )
        suggested_questions = questions_response["choices"][0]["message"]["content"].split("\n")
        suggested_questions = [q.strip() for q in suggested_questions if q.strip()]
        
        return {
            "summary": summary,
            "suggested_questions": suggested_questions[:5],
            "doc_id": request.doc_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"摘要生成失败: {str(e)}")

# ==================== 文档上传和管理 ====================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """上传PDF文件"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    try:
        # 读取文件内容
        content = await file.read()
        pdf_file = io.BytesIO(content)

        # 提取文本
        extracted_data = extract_text_from_pdf(pdf_file)

        # 生成文档ID（使用文本内容生成唯一ID）
        doc_id = generate_doc_id(extracted_data["full_text"])

        # 保存PDF文件到磁盘
        pdf_filename = f"{doc_id}.pdf"
        pdf_path = os.path.join("uploads", pdf_filename)
        with open(pdf_path, "wb") as f:
            f.write(content)

        # 生成PDF访问URL
        pdf_url = f"/uploads/{pdf_filename}"

        # 存储文档（包含pdf_url）
        documents_store[doc_id] = {
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "data": extracted_data,
            "pdf_url": pdf_url
        }

        return {
            "message": "PDF上传成功",
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": extracted_data["total_pages"],
            "total_chars": len(extracted_data["full_text"]),
            "pdf_url": pdf_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF处理失败: {str(e)}")

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """获取文档详情"""
    if doc_id not in documents_store:
        raise HTTPException(status_code=404, detail="文档未找到")

    doc = documents_store[doc_id]
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "upload_time": doc["upload_time"],
        "total_pages": doc["data"]["total_pages"],
        "total_chars": len(doc["data"]["full_text"]),
        "pages": doc["data"]["pages"],
        "pdf_url": doc.get("pdf_url")
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/version")
async def get_version():
    """检查后端代码版本"""
    return {"version": "2.0.1", "build_time": "2025-11-25 19:30:00", "feature": "native_pdf_url"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
