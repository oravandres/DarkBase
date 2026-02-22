"""
MiMi LLM Adapter Service
Wraps Ollama API behind a stable, model-agnostic HTTP interface.
"""

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("llm-adapter")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "llm_adapter_requests_total",
    "Total requests to LLM adapter",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "llm_adapter_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)
ADAPTER_STATE = Gauge(
    "llm_adapter_state",
    "Current adapter state (0=error, 1=loading, 2=ready, 3=busy)",
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
adapter_state = "loading"  # loading | ready | busy | error
active_requests = 0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    state: str
    ollama_reachable: bool
    models_loaded: int = 0
    active_requests: int = 0


class ModelInfo(BaseModel):
    id: str
    name: str
    size: int = 0
    parameter_size: str = ""


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_error(code: str, message: str, status_code: int = 500, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {}),
        metadata={
            "service": "llm-adapter",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": str(uuid.uuid4()),
        },
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _set_state(s: str):
    global adapter_state
    adapter_state = s
    ADAPTER_STATE.set({"error": 0, "loading": 1, "ready": 2, "busy": 3}.get(s, 0))


async def _check_ollama(client: httpx.AsyncClient) -> tuple[bool, int]:
    """Returns (reachable, model_count)."""
    try:
        r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=HEALTH_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            return True, len(data.get("models", []))
    except Exception:
        pass
    return False, 0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _set_state("loading")
    logger.info("LLM adapter starting, checking Ollama at %s", OLLAMA_BASE_URL)
    async with httpx.AsyncClient() as client:
        reachable, _ = await _check_ollama(client)
        if reachable:
            _set_state("ready")
            logger.info("Ollama is reachable – adapter ready")
        else:
            _set_state("error")
            logger.warning("Ollama not reachable at startup – will retry on health checks")
    yield
    logger.info("LLM adapter shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="MiMi LLM Adapter", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Middleware for metrics
# ---------------------------------------------------------------------------
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/chat/health")
async def health():
    async with httpx.AsyncClient() as client:
        reachable, model_count = await _check_ollama(client)
    if reachable and adapter_state != "busy":
        _set_state("ready")
    elif not reachable:
        _set_state("error")
    return HealthResponse(
        status="ok" if reachable else "degraded",
        state=adapter_state,
        ollama_reachable=reachable,
        models_loaded=model_count,
        active_requests=active_requests,
    )


@app.get("/api/v1/chat/models")
async def list_models():
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=HEALTH_TIMEOUT)
            r.raise_for_status()
        except Exception as exc:
            return make_error("OLLAMA_UNREACHABLE", str(exc), 502)
    data = r.json()
    models = [
        ModelInfo(
            id=m["name"],
            name=m["name"],
            size=m.get("size", 0),
            parameter_size=m.get("details", {}).get("parameter_size", ""),
        )
        for m in data.get("models", [])
    ]
    return ModelsResponse(models=models)


@app.post("/api/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    global active_requests

    active_requests += 1
    if adapter_state == "ready":
        _set_state("busy")

    try:
        ollama_payload = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": req.stream,
        }
        if req.temperature is not None:
            ollama_payload["options"] = ollama_payload.get("options", {})
            ollama_payload["options"]["temperature"] = req.temperature
        if req.max_tokens is not None:
            ollama_payload["options"] = ollama_payload.get("options", {})
            ollama_payload["options"]["num_predict"] = req.max_tokens

        if req.stream:
            return EventSourceResponse(_stream_chat(ollama_payload))
        else:
            return await _non_stream_chat(ollama_payload, req.model)
    except httpx.ConnectError:
        return make_error("OLLAMA_UNREACHABLE", "Cannot connect to Ollama", 502)
    except httpx.TimeoutException:
        return make_error("TIMEOUT", "Request to Ollama timed out", 504)
    except Exception as exc:
        logger.exception("Unexpected error in chat completions")
        return make_error("INTERNAL_ERROR", str(exc), 500)
    finally:
        active_requests -= 1
        if active_requests <= 0:
            active_requests = 0
            _set_state("ready")


async def _non_stream_chat(payload: dict, model: str) -> ChatResponse:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code != 200:
            error_text = r.text
            raise HTTPException(status_code=r.status_code, detail=error_text)
        data = r.json()

    content = data.get("message", {}).get("content", "")
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        ),
    )


async def _stream_chat(payload: dict) -> AsyncGenerator[dict, None]:
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                import json

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = chunk.get("message", {}).get("content", "")
                done = chunk.get("done", False)

                sse_data = {
                    "id": f"chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": "stop" if done else None,
                        }
                    ],
                }
                yield {"data": json.dumps(sse_data)}

                if done:
                    yield {"data": "[DONE]"}
                    return


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
