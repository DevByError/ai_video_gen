"""
AI Video Generator CLI wrapper.
This file now keeps the existing script behavior while delegating the pipeline
work to video_pipeline.py.
"""

import os

from video_pipeline import generate_videos_from_script


if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    script_path = os.path.join(BASE_DIR, "script.txt")
    output_root = os.path.join(BASE_DIR, "output")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Missing script file: {script_path}")

    with open(script_path, "r", encoding="utf-8") as script_file:
        script_text = script_file.read().strip()

    if not script_text:
        raise ValueError("script.txt is empty. Add the text you want to turn into video.")

    result = generate_videos_from_script(
        script_text=script_text,
        output_dir=output_root,
        num_videos=5,
        base_video_path=os.path.join(BASE_DIR, "base_video.mp4"),
        combine_final=True,
    )

    print("\n✅ CLI generation complete.")
    print("Output folder:", result["job_dir"])
    print("Videos:")
    for video_path in result["videos"]:
        print(video_path)
    if result.get("combined_video"):
        print("Combined video:", result["combined_video"])
