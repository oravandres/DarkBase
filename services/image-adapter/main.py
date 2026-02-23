"""
MiMi Image Adapter Service
Wraps ComfyUI API behind a stable image generation interface with queue management.
"""

import asyncio
import json
import os
import random
import re
import shutil
import time
import uuid
import logging
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import aiofiles
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
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
COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "http://localhost:8188")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/media/andres/data/ai-outputs/images")
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "10"))
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "5"))
GENERATION_TIMEOUT = float(os.getenv("GENERATION_TIMEOUT", "600"))
FLUX_MODEL_VERSION = os.getenv("FLUX_MODEL_VERSION", "dev")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("image-adapter")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "image_adapter_requests_total",
    "Total requests to image adapter",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "image_adapter_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.5, 1, 5, 10, 30, 60, 120, 300, 600],
)
QUEUE_DEPTH = Gauge("image_adapter_queue_depth", "Current image generation queue depth")
ADAPTER_STATE = Gauge(
    "image_adapter_state",
    "Current adapter state (0=error, 1=loading, 2=ready, 3=busy)",
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 20
    seed: int | None = None
    cfg_scale: float = 1.0


class JobInfo(BaseModel):
    id: str
    status: JobStatus
    prompt: str
    width: int
    height: int
    steps: int
    seed: int
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    image_url: str | None = None
    error_message: str | None = None
    queue_position: int | None = None


class QueueResponse(BaseModel):
    queue_depth: int
    max_depth: int
    current_job: JobInfo | None = None
    queued_jobs: list[JobInfo]


class HealthResponse(BaseModel):
    status: str
    state: str
    comfyui_reachable: bool
    queue_depth: int
    current_job_id: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
jobs: dict[str, dict[str, Any]] = {}
job_queue: asyncio.Queue | None = None
worker_task: asyncio.Task | None = None
adapter_state = "loading"
current_job_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_error(code: str, message: str, status_code: int = 500, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {}),
        metadata={
            "service": "image-adapter",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": str(uuid.uuid4()),
        },
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _set_state(s: str):
    global adapter_state
    adapter_state = s
    ADAPTER_STATE.set({"error": 0, "loading": 1, "ready": 2, "busy": 3}.get(s, 0))


async def _check_comfyui() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{COMFYUI_BASE_URL}/system_stats", timeout=HEALTH_TIMEOUT)
            return r.status_code == 200
    except Exception:
        return False


def _build_flux_workflow(
    prompt: str, width: int, height: int, steps: int, seed: int,
    denoise: float = 1.0, init_image: str | None = None
) -> dict:
    """Build a ComfyUI workflow for FLUX.1 image generation.

    Uses split-model nodes (UNETLoader, DualCLIPLoader, VAELoader) matching
    the directory layout created by the Ansible comfyui role:
      models/unet/flux1-{version}.safetensors
      models/clip/clip_l.safetensors + t5xxl_fp8_e4m3fn.safetensors
      models/vae/ae.safetensors
    """
    unet_name = f"flux1-{FLUX_MODEL_VERSION}.safetensors"

    workflow = {
        "prompt": {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": denoise,
                    "model": ["10", 0],  # Will be updated if LoRAs exist
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["13", 0] if init_image else ["5", 0],
                },
            },
            "5": {
                "class_type": "LoadImage" if init_image else "EmptyLatentImage",
                "inputs": {"image": init_image} if init_image else {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,  # Will be updated to stripped prompt
                    "clip": ["11", 0], # Will be updated if LoRAs exist
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["11", 0], # Will be updated if LoRAs exist
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["12", 0],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "mimi",
                    "images": ["8", 0],
                },
            },
            "10": {
                "class_type": "UNETLoaderNF4" if FLUX_MODEL_VERSION == "gaia" else "UNETLoader",
                "inputs": {"unet_name": unet_name} if FLUX_MODEL_VERSION == "gaia" else {
                    "unet_name": unet_name,
                    "weight_dtype": "default",
                },
            },
            "11": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "clip_l.safetensors",
                    "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                    "type": "flux",
                },
            },
            "12": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "ae.safetensors",
                },
            },
        }
    }
    
    # Inject VAE Encode if doing img2img
    if init_image:
        workflow["prompt"]["13"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["5", 0],
                "vae": ["12", 0]
            }
        }

    # Intercept <lora:filename:strength> tags from the prompt
    lora_matches = list(re.finditer(r"<lora:([^:>]+)(?::([0-9.]+))?>", prompt))
    clean_prompt = prompt

    last_model = ["10", 0]
    last_clip = ["11", 0]
    node_id_counter = 20

    for match in lora_matches:
        full_match = match.group(0)
        lora_name = match.group(1)
        strength = float(match.group(2)) if match.group(2) else 1.0

        if not lora_name.endswith(".safetensors"):
            lora_name += ".safetensors"

        clean_prompt = clean_prompt.replace(full_match, "")

        node_id = str(node_id_counter)
        workflow["prompt"][node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
                "model": last_model,
                "clip": last_clip,
            },
        }
        last_model = [node_id, 0]
        last_clip = [node_id, 1]
        node_id_counter += 1

    clean_prompt = clean_prompt.strip()

    # Update workflow with new links and clean prompt
    workflow["prompt"]["6"]["inputs"]["text"] = clean_prompt
    workflow["prompt"]["3"]["inputs"]["model"] = last_model
    workflow["prompt"]["6"]["inputs"]["clip"] = last_clip
    workflow["prompt"]["7"]["inputs"]["clip"] = last_clip

    return workflow


def _make_job_info(job: dict[str, Any]) -> JobInfo:
    return JobInfo(
        id=job["id"],
        status=job["status"],
        prompt=job["prompt"],
        width=job["width"],
        height=job["height"],
        steps=job["steps"],
        seed=job["seed"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        duration_seconds=job.get("duration_seconds"),
        image_url=job.get("image_url"),
        error_message=job.get("error_message"),
        queue_position=job.get("queue_position"),
    )


# ---------------------------------------------------------------------------
# Queue Worker
# ---------------------------------------------------------------------------
async def _queue_worker():
    """Process image generation jobs one at a time."""
    global current_job_id

    logger.info("Image generation queue worker started")
    while True:
        job_id = await job_queue.get()
        if job_id not in jobs:
            job_queue.task_done()
            continue

        job = jobs[job_id]
        current_job_id = job_id
        _set_state("busy")

        job["status"] = JobStatus.PROCESSING
        job["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        start_time = time.monotonic()

        # Update queue positions for remaining jobs
        _update_queue_positions()

        logger.info("Processing image job %s: '%s'", job_id, job["prompt"][:50])

        try:
            workflow = _build_flux_workflow(
                prompt=job["prompt"],
                width=job["width"],
                height=job["height"],
                steps=job["steps"],
                seed=job["seed"],
                denoise=job.get("denoise", 1.0),
                init_image=job.get("init_image", None)
            )

            # Submit to ComfyUI
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{COMFYUI_BASE_URL}/prompt",
                    json=workflow,
                    timeout=30,
                )
                if r.status_code != 200:
                    raise Exception(f"ComfyUI returned {r.status_code}: {r.text}")
                comfy_data = r.json()
                prompt_id = comfy_data.get("prompt_id")

            if not prompt_id:
                raise Exception("No prompt_id returned from ComfyUI")

            # Poll for completion
            image_filename = await _poll_comfyui_completion(prompt_id)

            # Move output to persistent storage
            output_path = await _save_output(image_filename, job_id)

            duration = time.monotonic() - start_time
            job["status"] = JobStatus.COMPLETED
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            job["duration_seconds"] = round(duration, 2)
            job["image_url"] = f"/api/v1/images/outputs/{os.path.basename(output_path)}"

            # Save metadata sidecar
            _save_image_metadata(job, output_path)

            logger.info("Job %s completed in %.1fs", job_id, duration)

        except Exception as exc:
            duration = time.monotonic() - start_time
            job["status"] = JobStatus.ERROR
            job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            job["duration_seconds"] = round(duration, 2)
            job["error_message"] = str(exc)
            logger.error("Job %s failed: %s", job_id, exc)

        finally:
            current_job_id = None
            QUEUE_DEPTH.set(job_queue.qsize())
            if job_queue.empty():
                _set_state("ready")
            job_queue.task_done()


async def _poll_comfyui_completion(prompt_id: str, timeout: float = None) -> str:
    """Poll ComfyUI /history endpoint until the prompt is done."""
    timeout = timeout or GENERATION_TIMEOUT
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{COMFYUI_BASE_URL}/history/{prompt_id}",
                    timeout=HEALTH_TIMEOUT,
                )
                if r.status_code != 200:
                    continue
                history = r.json()

            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                # Find SaveImage node output
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            return img["filename"]
                raise Exception("No image output found in ComfyUI response")
        except httpx.RequestError:
            continue

    raise Exception(f"ComfyUI generation timed out after {timeout}s")


async def _save_output(comfyui_filename: str, job_id: str) -> str:
    """Copy generated image from ComfyUI output to persistent storage."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ext = Path(comfyui_filename).suffix or ".png"
    output_filename = f"{job_id}{ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Download from ComfyUI /view endpoint
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{COMFYUI_BASE_URL}/view",
            params={"filename": comfyui_filename},
            timeout=30,
        )
        if r.status_code != 200:
            raise Exception(f"Failed to download image from ComfyUI: {r.status_code}")

        async with aiofiles.open(output_path, "wb") as f:
            await f.write(r.content)

    logger.info("Saved output to %s", output_path)
    return output_path


def _update_queue_positions():
    """Update queue_position for all queued jobs."""
    position = 1
    for job in jobs.values():
        if job["status"] == JobStatus.QUEUED:
            job["queue_position"] = position
            position += 1
        else:
            job["queue_position"] = None


def _save_image_metadata(job: dict, output_path: str):
    """Save a JSON metadata sidecar next to the generated image."""
    try:
        meta = {
            "id": job["id"],
            "type": "image",
            "prompt": job["prompt"],
            "width": job["width"],
            "height": job["height"],
            "steps": job["steps"],
            "seed": job["seed"],
            "created_at": job.get("completed_at", ""),
            "duration_seconds": job.get("duration_seconds"),
            "image_file": os.path.basename(output_path),
        }
        meta_path = os.path.splitext(output_path)[0] + ".json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved image metadata %s", meta_path)
    except Exception as exc:
        logger.warning("Failed to save image metadata: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue, worker_task
    _set_state("loading")
    logger.info("Image adapter starting, checking ComfyUI at %s", COMFYUI_BASE_URL)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize queue
    job_queue = asyncio.Queue(maxsize=MAX_QUEUE_DEPTH)

    # Start worker
    worker_task = asyncio.create_task(_queue_worker())

    reachable = await _check_comfyui()
    if reachable:
        _set_state("ready")
        logger.info("ComfyUI is reachable – adapter ready")
    else:
        _set_state("error")
        logger.warning("ComfyUI not reachable at startup – will retry on health checks")

    yield

    # Shutdown
    logger.info("Image adapter shutting down")
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="MiMi Image Adapter", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Middleware
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
@app.get("/api/v1/images/health")
async def health():
    reachable = await _check_comfyui()
    if reachable and adapter_state not in ("busy",):
        _set_state("ready")
    elif not reachable:
        _set_state("error")

    return HealthResponse(
        status="ok" if reachable else "degraded",
        state=adapter_state,
        comfyui_reachable=reachable,
        queue_depth=job_queue.qsize() if job_queue else 0,
        current_job_id=current_job_id,
    )


@app.post("/api/v1/images/generate")
async def generate_image(
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    steps: int = Form(20),
    seed: int | None = Form(None),
    cfg_scale: float = Form(1.0),
    denoise: float = Form(1.0),
    image: UploadFile | None = File(None)
):
    if job_queue is None:
        return make_error("NOT_READY", "Service not yet initialized", 503)

    if job_queue.full():
        return make_error(
            "QUEUE_FULL",
            f"Queue is full ({MAX_QUEUE_DEPTH} jobs). Try again later.",
            429,
        )

    seed_val = seed if seed is not None else random.randint(0, 2**32 - 1)
    job_id = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Handle image upload if provided
    init_image_name = None
    if image is not None and image.size > 0:
        try:
            # Send file to ComfyUI upload endpoint
            files = {"image": (image.filename, image.file, image.content_type)}
            data = {"overwrite": "true"}
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{COMFYUI_BASE_URL}/upload/image", files=files, data=data, timeout=30
                )
                if r.status_code == 200:
                    comfy_res = r.json()
                    init_image_name = comfy_res.get("name")
                else:
                    logger.error(f"Failed to upload image to ComfyUI: {r.status_code} {r.text}")
        except Exception as e:
            logger.error(f"Error handling image upload: {e}")

    job = {
        "id": job_id,
        "status": JobStatus.QUEUED,
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed_val,
        "cfg_scale": cfg_scale,
        "denoise": denoise,
        "init_image": init_image_name,
        "created_at": now,
        "queue_position": job_queue.qsize() + 1,
    }
    jobs[job_id] = job

    await job_queue.put(job_id)
    QUEUE_DEPTH.set(job_queue.qsize())

    logger.info("Queued image job %s (queue depth: %d)", job_id, job_queue.qsize())
    return _make_job_info(job)


@app.get("/api/v1/images/queue")
async def get_queue():
    queued = [_make_job_info(j) for j in jobs.values() if j["status"] == JobStatus.QUEUED]
    current = None
    if current_job_id and current_job_id in jobs:
        current = _make_job_info(jobs[current_job_id])

    return QueueResponse(
        queue_depth=job_queue.qsize() if job_queue else 0,
        max_depth=MAX_QUEUE_DEPTH,
        current_job=current,
        queued_jobs=queued,
    )


@app.get("/api/v1/images/outputs/{filename}")
async def serve_output(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return make_error("NOT_FOUND", f"Image {filename} not found", 404)
    return FileResponse(filepath, media_type="image/png")


@app.get("/api/v1/images/history")
async def list_image_history(limit: int = 50, offset: int = 0):
    """List past image generations, newest first."""
    try:
        output_path = Path(OUTPUT_DIR)
        if not output_path.exists():
            return {"images": [], "total": 0}

        files = sorted(output_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        total = len(files)
        page = files[offset : offset + limit]

        images = []
        for f in page:
            try:
                data = json.loads(f.read_text())
                data["image_url"] = f"/api/v1/images/outputs/{data.get('image_file', '')}"
                images.append(data)
            except Exception:
                continue

        return {"images": images, "total": total}
    except Exception as exc:
        return make_error("HISTORY_ERROR", str(exc), 500)


# This MUST be after all static /api/v1/images/* routes to avoid catching them
@app.get("/api/v1/images/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return make_error("NOT_FOUND", f"Job {job_id} not found", 404)
    return _make_job_info(jobs[job_id])


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

