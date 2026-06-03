# Runpod Deployment Guide for ai_video_gen

## 1. Install dependencies

On Runpod, in the repository root run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use a GPU image with CUDA, make sure the `torch` package version in `requirements.txt` matches that CUDA version.

## 2. Start the API server

Run the FastAPI app with a single uvicorn worker:

```bash
cd /workspace  # or the repository root
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use only one worker because the video pipeline uses the GPU and external subprocesses.

## 3. Call the API

### Create a job

```bash
curl -X POST http://0.0.0.0:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "script": "Your full video script text goes here.",
    "num_videos": 5,
    "combine_final": false
  }'
```

Response example:

```json
{
  "job_id": "...",
  "status": "queued"
}
```

### Check job status

```bash
curl http://0.0.0.0:8000/status/<job_id>
```

### Download finished videos

```bash
curl -O http://0.0.0.0:8000/download/<job_id>/segment_1/video_1.mp4
```

If `combine_final` was enabled, you can also download:

```bash
curl -O http://0.0.0.0:8000/download/<job_id>/final_combined.mp4
```

## 4. Notes for web integration

- Your website backend can call `/generate` to queue generation jobs.
- Use `/status/<job_id>` to poll progress.
- Use `/download/<job_id>/<filename>` to fetch completed video files.
- For a production website, make your UI send the script text and `num_videos: 5`.

## 5. Important Runpod tips

- Make sure `ffmpeg` is installed in the container.
- Make sure the `Wav2Lip` and `GFPGAN` folders are present in the repo.
- If you want to run multiple users at once, the queue in `api_server.py` will accept jobs but only process them one at a time to keep GPU usage safe.
- If you later need more throughput, use a larger GPU instance or add a second Runpod pod behind a queue.
