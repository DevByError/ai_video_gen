"""Video pipeline helper functions"""

import os
import nltk
import math
from typing import List


def ensure_nltk():
    """Ensure NLTK data is downloaded"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def safe_text(text: str) -> str:
    """Clean and normalize text"""
    return text.strip().replace("\r", "\n")


def split_into_video_chunks(script_text: str, num_videos: int = 5) -> List[str]:
    """Split script into chunks for video segments"""
    ensure_nltk()

    script_text = safe_text(script_text)
    if not script_text:
        return []

    # Split into sentences
    sentences = [s.strip() for s in nltk.sent_tokenize(script_text) if s.strip()]

    if not sentences:
        return [script_text]

    if num_videos <= 1 or len(sentences) <= num_videos:
        return sentences

    # Group sentences into chunks
    chunk_size = math.ceil(len(sentences) / num_videos)
    chunks = [
        " ".join(sentences[i : i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]

    return chunks[:num_videos]
