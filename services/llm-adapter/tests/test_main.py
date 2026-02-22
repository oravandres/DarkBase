"""Tests for the LLM adapter service."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, Response
from main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_ollama_reachable():
    mock_response = Response(
        200,
        json={"models": [{"name": "deepseek-r1:32b"}, {"name": "qwen3:32b"}]},
    )
    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        async with AsyncClient(app=app, base_url="http://test") as ac:
            r = await ac.get("/api/v1/chat/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ollama_reachable"] is True
        assert data["models_loaded"] == 2


@pytest.mark.anyio
async def test_health_ollama_unreachable():
    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(side_effect=Exception("Connection refused"))
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        async with AsyncClient(app=app, base_url="http://test") as ac:
            r = await ac.get("/api/v1/chat/health")
        assert r.status_code == 200
        data = r.json()
        assert data["ollama_reachable"] is False
        assert data["state"] == "error"


@pytest.mark.anyio
async def test_models_endpoint():
    mock_response = Response(
        200,
        json={
            "models": [
                {"name": "deepseek-r1:32b", "size": 19000000000, "details": {"parameter_size": "32B"}},
            ]
        },
    )
    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=mock_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        async with AsyncClient(app=app, base_url="http://test") as ac:
            r = await ac.get("/api/v1/chat/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["id"] == "deepseek-r1:32b"


@pytest.mark.anyio
async def test_chat_completions_non_stream():
    mock_ollama_response = Response(
        200,
        json={
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )
    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post = AsyncMock(return_value=mock_ollama_response)
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        async with AsyncClient(app=app, base_url="http://test") as ac:
            r = await ac.post(
                "/api/v1/chat/completions",
                json={
                    "model": "deepseek-r1:32b",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5


@pytest.mark.anyio
async def test_metrics_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/metrics")
    assert r.status_code == 200
    assert "llm_adapter_requests_total" in r.text
