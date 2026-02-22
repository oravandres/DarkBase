"""Tests for the image adapter service."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from httpx import AsyncClient, Response
from main import app, jobs, JobStatus


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clear_jobs():
    jobs.clear()
    yield
    jobs.clear()


@pytest.mark.anyio
async def test_health_comfyui_reachable():
    with patch("main._check_comfyui", return_value=True):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            r = await ac.get("/api/v1/images/health")
        assert r.status_code == 200
        data = r.json()
        assert data["comfyui_reachable"] is True


@pytest.mark.anyio
async def test_health_comfyui_unreachable():
    with patch("main._check_comfyui", return_value=False):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            r = await ac.get("/api/v1/images/health")
        assert r.status_code == 200
        data = r.json()
        assert data["comfyui_reachable"] is False
        assert data["state"] == "error"


@pytest.mark.anyio
async def test_generate_queues_job():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post(
            "/api/v1/images/generate",
            json={"prompt": "A test image", "width": 512, "height": 512, "steps": 10},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "queued"
    assert data["prompt"] == "A test image"
    assert data["width"] == 512
    assert data["id"] is not None


@pytest.mark.anyio
async def test_get_job_not_found():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/api/v1/images/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_queue_status():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Queue a job
        await ac.post(
            "/api/v1/images/generate",
            json={"prompt": "Test", "width": 512, "height": 512},
        )
        r = await ac.get("/api/v1/images/queue")
    assert r.status_code == 200
    data = r.json()
    assert data["max_depth"] == 10


@pytest.mark.anyio
async def test_outputs_not_found():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/api/v1/images/outputs/nonexistent.png")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_metrics_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/metrics")
    assert r.status_code == 200
    assert "image_adapter_requests_total" in r.text
