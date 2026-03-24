# filename: local_llm_api.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama's default endpoint
MODEL_NAME = "llama3"  # Replace with your model name

class ChatRequest(BaseModel):
    model: str
    messages: list  # expects a list of dicts with 'role' and 'content'
    temperature: float = 0.7

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Extract conversation
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in req.messages]) + "\nassistant:"

    ollama_payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": req.temperature,
        "stream": False
    }

    # Send to Ollama
    try:
        response = requests.post(OLLAMA_URL, json=ollama_payload)
        result = response.json()
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": result["response"]
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "model": MODEL_NAME,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        return {"error": str(e)}
