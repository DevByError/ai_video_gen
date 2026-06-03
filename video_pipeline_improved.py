import os
import uuid
import subprocess
from typing import List, Dict, Optional
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Constants
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
DEFAULT_SLIDES_ROOT = os.path.join(BASE_DIR, "slides")
DEFAULT_WAV2LIP_PATH = os.path.join(BASE_DIR, "Wav2Lip")
DEFAULT_GFPGAN_PATH = os.path.join(BASE_DIR, "GFPGAN")
DEFAULT_BASE_VIDEO_PATH = os.path.join(BASE_DIR, "assets", "base_presenter.mp4")
DEFAULT_TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

# Import needed functions from original
from video_pipeline import ensure_nltk, safe_text, split_into_video_chunks


class ProfessionalVideoGenerator:
    def __init__(self, base_video_path: Optional[str] = None):
        self.use_ai_avatar = False
        self.avatar_image = os.path.join(BASE_DIR, "assets", "avatar.png")
        self.background_music = os.path.join(BASE_DIR, "assets", "background_music.mp3")

        # Setup encoding for subprocesses
        self._setup_encoding()

        # Use the base_presenter.mp4 from assets folder
        if base_video_path and os.path.exists(base_video_path):
            self.base_video = base_video_path
        elif os.path.exists(os.path.join(BASE_DIR, "assets", "base_presenter.mp4")):
            self.base_video = os.path.join(BASE_DIR, "assets", "base_presenter.mp4")
        elif os.path.exists(os.path.join(BASE_DIR, "base_video_resized.mp4")):
            self.base_video = os.path.join(BASE_DIR, "base_video_resized.mp4")
        elif os.path.exists(os.path.join(BASE_DIR, "base_video.mp4")):
            self.base_video = os.path.join(BASE_DIR, "base_video.mp4")
        else:
            raise FileNotFoundError(
                "No base video found! Please add assets/base_presenter.mp4"
            )

        print(f" Using base video: {self.base_video}")
        self.tts_model = None

    def _setup_encoding(self):
        """Setup proper encoding for Windows compatibility"""
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONUTF8"] = "1"

        if os.name == "nt":
            try:
                subprocess.run("chcp 65001", shell=True, capture_output=True)
            except:
                pass

    def _get_tts_model(self):
        """Lazy load TTS model"""
        if self.tts_model is None:
            from TTS.api import TTS

            self.tts_model = TTS(model_name=DEFAULT_TTS_MODEL)
        return self.tts_model

    def create_professional_slide(
        self, text: str, index: int, output_dir: str, template: str = "modern"
    ) -> str:
        """Create beautiful slides with branding"""
        os.makedirs(output_dir, exist_ok=True)

        if template == "modern":
            img = Image.new("RGB", (1920, 1080), color=(15, 25, 45))
            draw = ImageDraw.Draw(img)

            draw.rectangle([(0, 0), (1920, 15)], fill=(100, 150, 255))
            draw.rectangle([(0, 1065), (1920, 1080)], fill=(100, 150, 255))

            try:
                title_font = ImageFont.truetype("arial.ttf", 72)
                body_font = ImageFont.truetype("arial.ttf", 48)
            except:
                title_font = ImageFont.load_default()
                body_font = ImageFont.load_default()

            words = text.split()
            lines = []
            current_line = ""
            max_chars = 30

            for word in words:
                if len(current_line + " " + word) <= max_chars:
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            title = f"Point {index + 1}"
            try:
                title_bbox = draw.textbbox((0, 0), title, font=title_font)
                title_width = title_bbox[2] - title_bbox[0]
                draw.text(
                    (1920 // 2 - title_width // 2, 100),
                    title,
                    fill=(100, 150, 255),
                    font=title_font,
                )
            except:
                draw.text((100, 100), title, fill=(100, 150, 255), font=title_font)

            y_pos = 250
            for line in lines:
                try:
                    line_bbox = draw.textbbox((0, 0), line, font=body_font)
                    line_width = line_bbox[2] - line_bbox[0]
                    draw.text(
                        (1920 // 2 - line_width // 2, y_pos),
                        line,
                        fill=(255, 255, 255),
                        font=body_font,
                    )
                except:
                    draw.text((100, y_pos), line, fill=(255, 255, 255), font=body_font)
                y_pos += 80

        logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                logo = Image.open(logo_path).resize((200, 200))
                img.paste(logo, (1720, 880), logo if logo.mode == "RGBA" else None)
            except:
                pass

        slide_path = os.path.join(output_dir, f"slide_{index + 1}.png")
        img.save(slide_path)
        return slide_path

    def _generate_audio(self, text: str, output_dir: str, index: int) -> str:
        """Generate audio using TTS"""
        audio_path = os.path.join(output_dir, f"audio_{index+1}.wav")
        tts = self._get_tts_model()
        tts.tts_to_file(text=text, file_path=audio_path)
        return audio_path

    def _generate_lipsync_video(self, audio_path: str, output_path: str) -> str:
        """Generate lip-sync with pre-recorded video"""
        if not os.path.exists(self.base_video):
            raise FileNotFoundError(f"Base video not found: {self.base_video}")

        print(f"  Using base video: {self.base_video}")

        inference_script = os.path.join(DEFAULT_WAV2LIP_PATH, "inference_fixed.py")
        if not os.path.exists(inference_script):
            inference_script = os.path.join(DEFAULT_WAV2LIP_PATH, "inference.py")

        cmd = [
            "python",
            inference_script,
            "--checkpoint_path",
            os.path.join(DEFAULT_WAV2LIP_PATH, "checkpoints", "wav2lip_gan.pth"),
            "--face",
            self.base_video,
            "--audio",
            audio_path,
            "--outfile",
            output_path,
            "--pads",
            "0",
            "15",
            "0",
            "10",
            "--face_det_batch_size",
            "8",
            "--wav2lip_batch_size",
            "16",
            "--resize_factor",
            "2",
        ]

        print(f"  Running Wav2Lip...")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=900,
            )

            if os.path.exists(output_path):
                print(f"  Wav2Lip completed successfully")
                return output_path

            alt_paths = [
                output_path.replace(".mp4", ".avi"),
                os.path.join(DEFAULT_WAV2LIP_PATH, "results", "result_voice.mp4"),
                os.path.join(DEFAULT_WAV2LIP_PATH, "test_output.mp4"),
                os.path.join(os.path.dirname(audio_path), "result_voice.mp4"),
            ]

            for alt in alt_paths:
                if os.path.exists(alt):
                    import shutil

                    shutil.copy2(alt, output_path)
                    print(f"  Found and copied output from: {alt}")
                    return output_path

            print(f"  Wav2Lip output not found, falling back to slideshow")
            return self._create_audio_slideshow(audio_path, output_path)

        except subprocess.TimeoutExpired:
            print(f"  Wav2Lip timed out, falling back to slideshow")
            return self._create_audio_slideshow(audio_path, output_path)
        except Exception as e:
            print(f"  Unexpected error: {str(e)}")
            return self._create_audio_slideshow(audio_path, output_path)

    def _create_audio_slideshow(self, audio_path: str, output_path: str) -> str:
        """Fallback: Create slideshow with audio (no ImageMagick required)"""
        from moviepy.editor import AudioFileClip, ColorClip, CompositeVideoClip

        audio = AudioFileClip(audio_path)

        # Create a simple colored background
        bg_clip = ColorClip(size=(1920, 1080), color=(15, 25, 45))
        bg_clip = bg_clip.set_duration(audio.duration)

        # Create text using PIL instead of TextClip
        img = Image.new("RGB", (1920, 1080), color=(15, 25, 45))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 70)
        except:
            font = ImageFont.load_default()

        text = "Generating video..."
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = 300
            text_height = 50

        x = (1920 - text_width) // 2
        y = (1080 - text_height) // 2
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        # Convert PIL image to numpy array for moviepy
        text_array = np.array(img)
        text_clip = ImageClip(text_array).set_duration(audio.duration)

        # Composite
        final = CompositeVideoClip([bg_clip, text_clip])
        final = final.set_audio(audio)

        # Write video
        final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

        # Cleanup
        audio.close()
        final.close()

        return output_path

    def create_engaging_videos(
        self, text_chunks: List[str], output_dir: str
    ) -> List[str]:
        """Main method to create engaging videos"""
        videos = []
        output_dir = os.path.abspath(output_dir)

        for i, chunk in enumerate(text_chunks):
            print(f"\nProcessing segment {i+1}/{len(text_chunks)}")
            segment_dir = os.path.join(output_dir, f"segment_{i+1}")
            os.makedirs(segment_dir, exist_ok=True)

            print(f"  Creating slide...")
            slide_path = self.create_professional_slide(chunk, i, segment_dir)

            print(f"  Generating audio...")
            audio_path = self._generate_audio(chunk, segment_dir, i)

            print(f"  Creating lip-sync video...")
            temp_video = os.path.join(segment_dir, f"temp_video_{i+1}.mp4")
            lip_sync_video = self._generate_lipsync_video(audio_path, temp_video)

            if not os.path.exists(lip_sync_video):
                print(f"  Error: Lip-sync video not created")
                continue

            print(f"  Adding slide overlay...")
            video_path = os.path.join(segment_dir, f"video_{i+1}.mp4")

            try:
                video_clip = VideoFileClip(lip_sync_video)
                slide_clip = ImageClip(slide_path).set_duration(video_clip.duration)
                slide_clip = slide_clip.resize(height=200)
                slide_clip = slide_clip.set_position(("right", "bottom")).margin(
                    right=20, bottom=20
                )
                final = CompositeVideoClip([video_clip, slide_clip])
                final.write_videofile(video_path, codec="libx264", audio_codec="aac")
                video_clip.close()
                slide_clip.close()
                final.close()
                videos.append(video_path)
                print(f"  Completed segment {i+1}")
            except Exception as e:
                print(f"  Error adding overlay: {e}")
                if os.path.exists(lip_sync_video):
                    import shutil

                    shutil.copy(lip_sync_video, video_path)
                    videos.append(video_path)
                    print(f"  Saved lip-sync video without overlay")

        return videos


def generate_videos_from_script(
    script_text: str,
    output_dir: str,
    num_videos: int = 5,
    base_video_path: Optional[str] = None,
    wav2lip_root: Optional[str] = None,
    gfpgan_root: Optional[str] = None,
    combine_final: bool = True,
    **kwargs,
) -> Dict[str, object]:
    """Generate videos using improved pipeline"""
    ensure_nltk()
    script_text = safe_text(script_text)
    if not script_text:
        raise ValueError("The script text is empty.")

    os.makedirs(output_dir, exist_ok=True)
    job_dir = os.path.join(output_dir, str(uuid.uuid4()))
    os.makedirs(job_dir, exist_ok=True)

    print(f"Job directory: {job_dir}")

    chunks = split_into_video_chunks(script_text, num_videos=num_videos)
    generator = ProfessionalVideoGenerator(base_video_path=base_video_path)
    videos = generator.create_engaging_videos(chunks, job_dir)

    if not videos:
        raise RuntimeError("No videos were created.")

    result = {
        "videos": videos,
        "job_dir": job_dir,
        "job_dir_name": os.path.basename(job_dir),
    }

    # ALWAYS create combined video if there's more than 1 video
    if len(videos) > 1:
        print("\nCreating combined video with FFmpeg...")
        final_combined = os.path.join(job_dir, "final_combined.mp4")

        # Create file list for FFmpeg
        filelist = os.path.join(job_dir, "filelist.txt")
        with open(filelist, "w") as f:
            for video in videos:
                abs_path = os.path.abspath(video)
                f.write(f"file '{abs_path}'\n")

        # Use FFmpeg to concatenate
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            filelist,
            "-c",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            final_combined,
            "-y",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  Combined video created: {final_combined}")
            result["combined_video"] = final_combined
            os.remove(filelist)
        except subprocess.CalledProcessError as e:
            print(f"  FFmpeg combine failed: {e}")
            print("  Falling back to MoviePy...")
            clips = []
            for v in videos:
                clip = VideoFileClip(v)
                if clip.audio is None:
                    clip = VideoFileClip(v, audio=True)
                clips.append(clip)

            combined = concatenate_videoclips(clips, method="compose")
            combined.write_videofile(
                final_combined,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp_audio.m4a",
                remove_temp=True,
                preset="medium",
            )
            result["combined_video"] = final_combined

            for clip in clips:
                clip.close()
            combined.close()
    else:
        # If only one video, just copy it as combined
        print("\nOnly one video segment, using as combined...")
        final_combined = os.path.join(job_dir, "final_combined.mp4")
        import shutil

        shutil.copy2(videos[0], final_combined)
        result["combined_video"] = final_combined

    return result


# Export constants for api_server
__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_BASE_VIDEO_PATH",
    "DEFAULT_WAV2LIP_PATH",
    "DEFAULT_GFPGAN_PATH",
    "generate_videos_from_script",
]
