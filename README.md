# AI Video Generator

![Python](https://img.shields.io/badge/python-3.10-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C)
![TTS](https://img.shields.io/badge/TTS-Coqui-0F766E)
![Lip-sync](https://img.shields.io/badge/lip--sync-Wav2Lip-5C3EE8)
![API](https://img.shields.io/badge/API-FastAPI-009688)

Turn a script into a presenter video. You give it text, and the pipeline writes the speech, animates a face so the lips match the audio, and puts it together with slides into a finished video.

## How it works

```
text -> speech (Coqui TTS) -> talking face (Wav2Lip / SadTalker) -> face restoration (GFPGAN) -> slides + video (MoviePy)
```

1. **Speech.** Coqui TTS turns the script into natural-sounding audio.
2. **Lip-sync.** Wav2Lip and SadTalker animate a face image so the mouth follows the speech.
3. **Restoration.** GFPGAN sharpens the generated face.
4. **Composition.** The talking head is combined with slides and rendered with MoviePy.

## Features

- Text to speech with Coqui TTS
- Lip-synced talking head with Wav2Lip and SadTalker
- Face enhancement with GFPGAN
- Slide generation and video composition
- FastAPI server for generating videos over HTTP

## Project layout

| File | Purpose |
|---|---|
| `video_pipeline.py`, `video_pipeline_improved.py` | The generation pipeline |
| `api_server.py` | FastAPI service |
| `main.py` | Command-line entry point |
| `download_models.sh` | Downloads the model weights |
| `deploy_runpod.sh`, `RUNPOD_DEPLOYMENT.md` | Deployment on a RunPod GPU |
| `Wav2Lip/`, `SadTalker/`, `GFPGAN/` | Model repositories, as git submodules |

## Getting started

A CUDA GPU is strongly recommended.

```bash
git clone --recurse-submodules https://github.com/DevByError/ai_video_gen.git
cd ai_video_gen
pip install -r requirements.txt
bash download_models.sh
```

For a cloud GPU, follow [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md) or run `deploy_runpod.sh`.

## Built with

PyTorch, Coqui TTS, Wav2Lip, SadTalker, GFPGAN, MoviePy, OpenCV, librosa, FastAPI and Uvicorn.
