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
import glob
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# Persistence configuration
DATA_DIR = "data"
DOCS_DIR = os.path.join(DATA_DIR, "docs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_stores")
# Ensure parent directory exists first
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

documents_store = {}

# Embedding Models Configuration
EMBEDDING_MODELS = {
    "local-minilm": {
        "name": "Local: MiniLM-L6 (Default)",
        "provider": "local",
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "max_tokens": 256,
        "price": "Free (Local)",
        "description": "Fast, general purpose"
    },
    "local-multilingual": {
        "name": "Local: Multilingual",
        "provider": "local",
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "max_tokens": 128,
        "price": "Free (Local)",
        "description": "Better for Chinese/multilingual"
    },
    "text-embedding-3-large": {
        "name": "OpenAI: text-embedding-3-large",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "dimension": 3072,
        "max_tokens": 8191,
        "price": "$0.13/M tokens",
        "description": "Best overall quality"
    },
    "text-embedding-3-small": {
        "name": "OpenAI: text-embedding-3-small",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "dimension": 1536,
        "max_tokens": 8191,
        "price": "$0.02/M tokens",
        "description": "Best value"
    },
    "text-embedding-v3": {
        "name": "Alibaba: text-embedding-v3",
        "provider": "openai",
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "$0.007/M tokens",
        "description": "Chinese optimized, cheapest"
    },
    "moonshot-embedding-v1": {
        "name": "Moonshot: moonshot-embedding-v1",
        "provider": "openai",
        "base_url": "https://api.moonshot.cn/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "$0.011/M tokens",
        "description": "Kimi, OpenAI compatible"
    },
    "deepseek-embedding-v1": {
        "name": "DeepSeek: deepseek-embedding-v1",
        "provider": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "$0.01/M tokens",
        "description": "Low cost OpenAI compatible"
    },
    "glm-embedding-2": {
        "name": "Zhipu: glm-embedding-2",
        "provider": "openai",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "$0.014/M tokens",
        "description": "GLM series"
    },
    "minimax-embedding-v2": {
        "name": "MiniMax: minimax-embedding-v2",
        "provider": "openai",
        "base_url": "https://api.minimaxi.chat/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "$0.014/M tokens",
        "description": "ABAB series"
    },
    "BAAI/bge-m3": {
        "name": "SiliconFlow: BAAI/bge-m3",
        "provider": "openai",
        "base_url": "https://api.siliconflow.cn/v1",
        "dimension": 1024,
        "max_tokens": 8192,
        "price": "$0.02/M tokens",
        "description": "Open source, hosted"
    }
}

# Lazy-loaded local embedding models cache
local_embedding_models = {}

def save_document(doc_id: str, data: dict):
    """Save document data to disk"""
    try:
        file_path = os.path.join(DOCS_DIR, f"{doc_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved document {doc_id} to {file_path}")
    except Exception as e:
        print(f"Error saving document {doc_id}: {e}")

def load_documents():
    """Load all documents from disk"""
    print("Loading documents from disk...")
    count = 0
    for file_path in glob.glob(os.path.join(DOCS_DIR, "*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                doc_id = os.path.splitext(os.path.basename(file_path))[0]
                documents_store[doc_id] = data
                count += 1
        except Exception as e:
            print(f"Error loading document from {file_path}: {e}")
    print(f"Loaded {count} documents.")

def get_embedding_function(embedding_model_id: str, api_key: str = None):
    """Get embedding function for the specified model"""
    if embedding_model_id not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model: {embedding_model_id}")
    
    config = EMBEDDING_MODELS[embedding_model_id]
    provider = config["provider"]
    
    if provider == "local":
        # Use local SentenceTransformer
        model_name = config["model_name"]
        if model_name not in local_embedding_models:
            print(f"Loading local embedding model: {model_name}")
            local_embedding_models[model_name] = SentenceTransformer(model_name)
        model = local_embedding_models[model_name]
        return lambda texts: model.encode(texts)
    
    elif provider == "openai":
        # OpenAI-compatible API (OpenAI, Alibaba, Moonshot, DeepSeek, etc.)
        if not api_key:
            raise ValueError(f"API key required for {embedding_model_id}")
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=config["base_url"])
        
        def embed_texts(texts):
            response = client.embeddings.create(
                model=embedding_model_id,
                input=texts
            )
            return np.array([item.embedding for item in response.data])
        
        return embed_texts
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def build_vector_index(doc_id: str, text: str, embedding_model_id: str = "local-minilm", api_key: str = None):
    """Build and save vector index for a document"""
    try:
        print(f"Building vector index for {doc_id}...")
        # 1. Chunk text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        print(f"Split into {len(chunks)} chunks.")
        
        if not chunks:
            return

        # 2. Generate embeddings using selected model
        embed_fn = get_embedding_function(embedding_model_id, api_key)
        embeddings = embed_fn(chunks)
        
        # 3. Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # 4. Save index and chunks
        index_path = os.path.join(VECTOR_STORE_DIR, f"{doc_id}.index")
        chunks_path = os.path.join(VECTOR_STORE_DIR, f"{doc_id}.pkl")
        
        faiss.write_index(index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump({"chunks": chunks, "embedding_model": embedding_model_id}, f)
            
        print(f"Vector index saved to {index_path}")
        
    except Exception as e:
        print(f"Error building vector index for {doc_id}: {e}")

def get_relevant_context(doc_id: str, query: str, api_key: str = None, top_k: int = 5) -> str:
    """Retrieve relevant context using vector search"""
    try:
        index_path = os.path.join(VECTOR_STORE_DIR, f"{doc_id}.index")
        chunks_path = os.path.join(VECTOR_STORE_DIR, f"{doc_id}.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            print(f"Vector index not found for {doc_id}")
            return ""
            
        # Load index and chunks with metadata
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
            
        # Handle old format (just chunks) or new format (dict with metadata)
        if isinstance(data, dict):
            chunks = data["chunks"]
            embedding_model_id = data.get("embedding_model", "local-minilm")
        else:
            chunks = data
            embedding_model_id = "local-minilm"  # Default for old indexes
            
        # Get embedding function using the same model that was used for indexing
        embed_fn = get_embedding_function(embedding_model_id, api_key)
        
        # Embed query
        query_vector = embed_fn([query])
        
        # Search
        D, I = index.search(np.array(query_vector).astype('float32'), top_k)
        
        # Retrieve chunks
        relevant_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        
        return "\n\n...\n\n".join(relevant_chunks)
        
    except Exception as e:
        print(f"Error retrieving context for {doc_id}: {e}")
        return ""

@app.on_event("startup")
async def startup_event():
    load_documents()

class ChatRequest(BaseModel):
    doc_id: str
    question: str
    api_key: str
    model: str
    api_provider: str
    selected_text: Optional[str] = None
    enable_vector_search: bool = False

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
    "openai": ["gpt-5.1-2025-11-13", "gpt-4.1", "gpt-5-nano", "o4-mini", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
    "anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
        "claude-haiku-4-5-20250219",
        "claude-3-opus-20240229", 
        "claude-3-sonnet-20240229", 
        "claude-3-haiku-20240307"
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.0-flash",
        "gemini-pro-vision"
    ],
    "grok": ["grok-4.1", "grok-4.1-fast", "grok-3", "grok-vision-beta"],
    "doubao": ["doubao-1.5-pro-256k", "doubao-1.5-pro-32k"],
    "qwen": ["qwen-max-2025-01-25", "qwen3-235b-a22b-instruct-2507", "qwen3-coder-plus-2025-09-23"],
    "minimax": ["abab6.5-chat", "abab6.5s-chat", "minimax-m2"]
}

# AI模型配置
AI_MODELS = {
    "openai": {
        "name": "OpenAI GPT",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "models": {
            "gpt-5.1-2025-11-13": "GPT-5.1 (视觉/MoE)",
            "gpt-4.1": "GPT-4.1 (视觉)",
            "gpt-5-nano": "GPT-5 Nano (视觉)",
            "o4-mini": "o4-mini (视觉)",
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
            "claude-opus-4-1-20250805": "Claude Opus 4.1 (视觉)",
            "claude-haiku-4-5-20250219": "Claude Haiku 4.5 (视觉)",
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
            "gemini-2.5-flash-lite-preview-09-2025": "Gemini 2.5 Flash-Lite (视觉)",
            "gemini-2.0-flash": "Gemini 2.0 Flash (视觉)",
            "gemini-pro-vision": "Gemini Pro Vision"
        }
    },
    "grok": {
        "name": "xAI Grok",
        "endpoint": "https://api.x.ai/v1/chat/completions",
        "models": {
            "grok-4.1": "Grok 4.1 (视觉)",
            "grok-4.1-fast": "Grok 4.1 Fast (视觉)",
            "grok-3": "Grok 3 (视觉)",
            "grok-vision-beta": "Grok Vision Beta"
        }
    },
    "doubao": {
        "name": "ByteDance Doubao (豆包)",
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "models": {
            "doubao-1.5-pro-256k": "Doubao 1.5 Pro 256K (视觉)",
            "doubao-1.5-pro-32k": "Doubao 1.5 Pro 32K (视觉)",
            "doubao-seed-code": "Doubao Seed Code"
        }
    },
    "qwen": {
        "name": "Alibaba Qwen (通义千问)",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "models": {
            "qwen-max-2025-01-25": "Qwen3-Max (视觉)",
            "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-A22B (视觉)",
            "qwen3-coder-plus-2025-09-23": "Qwen3-Coder-Plus (视觉)"
        }
    },
    "minimax": {
        "name": "MiniMax ABAB",
        "endpoint": "https://api.minimaxi.chat/v1/chat/completions",
        "models": {
            "abab6.5-chat": "ABAB 6.5 (视觉)",
            "abab6.5s-chat": "ABAB 6.5s (视觉)",
            "minimax-m2": "MiniMax-M2 (视觉)"
        }
    },
    "glm": {
        "name": "Zhipu GLM (智谱)",
        "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "models": {
            "glm-4.6": "GLM-4.6 (MoE)",
            "glm-4.5": "GLM-4.5",
            "glm-4.5-air": "GLM-4.5-Air"
        }
    },
    "deepseek": {
        "name": "DeepSeek",
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "models": {
            "deepseek-v3.2-exp": "DeepSeek V3.2-Exp",
            "deepseek-reasoner": "DeepSeek-R1 (推理)",
            "deepseek-chat": "DeepSeek Chat"
        }
    },
    "kimi": {
        "name": "Moonshot Kimi (月之暗面)",
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "models": {
            "kimi-k2-instruct-0905": "Kimi-K2-Instruct",
            "kimi-k2-thinking": "Kimi-K2-Thinking",
            "moonshot-v1": "Moonshot-V1"
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

@app.get("/embedding_models")
async def get_embedding_models():
    """获取可用嵌入模型列表"""
    return EMBEDDING_MODELS

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
    elif request.enable_vector_search:
        print(f"Using vector search for doc {request.doc_id}")
        relevant_text = get_relevant_context(request.doc_id, request.question, request.api_key)
        if relevant_text:
            context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
        else:
            context = doc["data"]["full_text"][:8000]
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
    elif request.enable_vector_search:
        print(f"Using vector search for doc {request.doc_id}")
        relevant_text = get_relevant_context(request.doc_id, request.question, request.api_key)
        if relevant_text:
            context = f"根据用户问题检索到的相关文档片段：\n\n{relevant_text}\n\n"
        else:
            context = doc["data"]["full_text"][:8000]
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
async def upload_pdf(
    file: UploadFile = File(...),
    embedding_model: str = "local-minilm",
    embedding_api_key: str = None
):
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

        # Persist to disk
        save_document(doc_id, documents_store[doc_id])
        
        # Build vector index with selected embedding model
        build_vector_index(doc_id, extracted_data["full_text"], embedding_model, embedding_api_key)

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

@app.get("/storage_info")
async def get_storage_info():
    """获取文件存储路径信息"""
    import platform

    # 获取绝对路径
    uploads_path = os.path.abspath("uploads")
    data_path = os.path.abspath(DATA_DIR)
    docs_path = os.path.abspath(DOCS_DIR)
    vector_stores_path = os.path.abspath(VECTOR_STORE_DIR)

    # 统计文件数量
    pdf_count = len(glob.glob(os.path.join(uploads_path, "*.pdf")))
    doc_count = len(glob.glob(os.path.join(docs_path, "*.json")))

    return {
        "uploads_dir": uploads_path,
        "data_dir": data_path,
        "docs_dir": docs_path,
        "vector_stores_dir": vector_stores_path,
        "pdf_count": pdf_count,
        "doc_count": doc_count,
        "platform": platform.system()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
