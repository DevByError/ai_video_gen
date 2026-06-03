import asyncio
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from video_pipeline_improved import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_BASE_VIDEO_PATH,
    DEFAULT_WAV2LIP_PATH,
    DEFAULT_GFPGAN_PATH,
    generate_videos_from_script,
)

app = FastAPI(title="AI Video Generator API")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_queue: asyncio.Queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}


# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)

    async def send_progress(self, job_id: str, data: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(data)
                except:
                    pass


manager = ConnectionManager()


class GenerateRequest(BaseModel):
    script: str = Field(..., description="The text script to generate video from.")
    num_videos: int = Field(
        5, ge=1, le=10, description="How many separate videos to create."
    )
    combine_final: bool = Field(
        True, description="Also generate a combined side-by-side summary video."
    )
    base_video_path: Optional[str] = Field(
        None,
        description="Optional path to a local face video. If omitted, the repository base_video.mp4 is used.",
    )
    wav2lip_root: Optional[str] = Field(
        None,
        description="Optional Wav2Lip directory path. If omitted, the repository Wav2Lip folder is used.",
    )
    gfpgan_root: Optional[str] = Field(
        None,
        description="Optional GFPGAN directory path. If omitted, the repository GFPGAN folder is used.",
    )


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: Optional[str] = None
    current_step: Optional[str] = None
    step_progress: Optional[int] = None
    total_steps: Optional[int] = None
    current_segment: Optional[int] = None
    total_segments: Optional[int] = None
    error: Optional[str] = None
    job_dir: Optional[str] = None
    videos: Optional[List[str]] = None
    combined_video: Optional[str] = None
    video_urls: Optional[List[str]] = None
    combined_video_url: Optional[str] = None


@app.on_event("startup")
async def start_worker() -> None:
    asyncio.create_task(worker())


async def update_progress(job_id: str, **kwargs):
    """Update job progress and notify via WebSocket"""
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        # Send WebSocket update
        await manager.send_progress(
            job_id, {"type": "progress", "job_id": job_id, **kwargs}
        )


async def worker() -> None:
    while True:
        job_id = await job_queue.get()
        job = jobs.get(job_id)
        if job is None:
            job_queue.task_done()
            continue

        job["status"] = "running"
        job["started_at"] = datetime.utcnow()
        job["progress"] = "Initializing TTS and models..."
        job["current_step"] = "loading_models"
        job["step_progress"] = 0
        job["total_steps"] = 5
        job["current_segment"] = 0

        await update_progress(
            job_id,
            status="running",
            started_at=job["started_at"],
            progress="Initializing TTS and models...",
            current_step="loading_models",
            step_progress=0,
        )

        try:
            output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, "jobs")
            os.makedirs(output_dir, exist_ok=True)

            await update_progress(
                job_id,
                progress="Splitting script into segments...",
                current_step="splitting_script",
                step_progress=10,
            )

            result = await asyncio.to_thread(
                generate_videos_from_script,
                job["script"],
                output_dir,
                job["num_videos"],
                job["base_video_path"],
                job["wav2lip_root"],
                job["gfpgan_root"],
                job["combine_final"],
            )

            # Generate URLs for downloads
            video_urls = []
            job_dir_name = os.path.basename(result.get("job_dir", ""))
            for i, video_path in enumerate(result.get("videos", []), 1):
                video_filename = os.path.basename(video_path)
                video_url = f"/download/{job_dir_name}/segment_{i}/{video_filename}"
                video_urls.append(video_url)

            combined_url = None
            if result.get("combined_video"):
                combined_filename = os.path.basename(result["combined_video"])
                combined_url = f"/download/{job_dir_name}/{combined_filename}"

            job["status"] = "completed"
            job["finished_at"] = datetime.utcnow()
            job["progress"] = "Complete!"
            job["current_step"] = "done"
            job["step_progress"] = 100
            job["videos"] = result.get("videos", [])
            job["combined_video"] = result.get("combined_video")
            job["job_dir"] = result.get("job_dir")
            job["video_urls"] = video_urls
            job["combined_video_url"] = combined_url

            await update_progress(
                job_id,
                status="completed",
                finished_at=job["finished_at"],
                progress="Complete!",
                current_step="done",
                step_progress=100,
                videos=job["videos"],
                combined_video=job["combined_video"],
                video_urls=video_urls,
                combined_video_url=combined_url,
            )

        except Exception as exc:
            job["status"] = "failed"
            job["finished_at"] = datetime.utcnow()
            job["progress"] = "Failed"
            job["error"] = str(exc)

            await update_progress(
                job_id,
                status="failed",
                finished_at=job["finished_at"],
                progress="Failed",
                error=str(exc),
            )
        finally:
            job_queue.task_done()


@app.get("/", summary="Healthcheck")
def root() -> Dict[str, str]:
    return {"message": "AI Video Generator API is running"}


@app.post("/generate", summary="Queue a video generation job")
def generate(request: GenerateRequest) -> Dict[str, str]:
    if not request.script.strip():
        raise HTTPException(status_code=400, detail="Script text cannot be empty.")

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow(),
        "started_at": None,
        "finished_at": None,
        "progress": "Queued...",
        "current_step": "queued",
        "step_progress": 0,
        "current_segment": 0,
        "total_segments": request.num_videos,
        "error": None,
        "script": request.script,
        "num_videos": request.num_videos,
        "combine_final": request.combine_final,
        "base_video_path": request.base_video_path or DEFAULT_BASE_VIDEO_PATH,
        "wav2lip_root": request.wav2lip_root or DEFAULT_WAV2LIP_PATH,
        "gfpgan_root": request.gfpgan_root or DEFAULT_GFPGAN_PATH,
    }
    jobs[job_id] = job
    job_queue.put_nowait(job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "poll_url": f"/status/{job_id}",
        "websocket_url": f"ws://localhost:8000/ws/{job_id}",
    }


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str) -> JobStatusResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        progress=job.get("progress"),
        current_step=job.get("current_step"),
        step_progress=job.get("step_progress"),
        total_steps=job.get("total_steps"),
        current_segment=job.get("current_segment"),
        total_segments=job.get("total_segments"),
        error=job.get("error"),
        job_dir=job.get("job_dir"),
        videos=job.get("videos"),
        combined_video=job.get("combined_video"),
        video_urls=job.get("video_urls"),
        combined_video_url=job.get("combined_video_url"),
    )


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        # Send initial status
        if job_id in jobs:
            await websocket.send_json(
                {
                    "type": "connected",
                    "job_id": job_id,
                    "status": jobs[job_id]["status"],
                }
            )

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
            elif data == "get_status":
                if job_id in jobs:
                    await websocket.send_json({"type": "status", **jobs[job_id]})
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


@app.get("/download/{folder_name}/{file_path:path}")
def download_file(folder_name: str, file_path: str) -> FileResponse:
    # Find job by folder name
    job = None
    for j in jobs.values():
        if j.get("job_dir") and folder_name in j.get("job_dir"):
            job = j
            break

    if not job or "job_dir" not in job:
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    safe_path = os.path.normpath(os.path.join(job["job_dir"], file_path))
    if not safe_path.startswith(os.path.abspath(job["job_dir"])):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=safe_path, filename=os.path.basename(file_path), media_type="video/mp4"
    )
