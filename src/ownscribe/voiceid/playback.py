"""Extract and play representative speaker clips for interactive enrollment."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def representative_range(ranges: list[tuple[float, float]]) -> tuple[float, float]:
    return max(ranges, key=lambda r: r[1] - r[0])


def total_duration(ranges: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in ranges)


def extract_clip(audio_path: Path, start: float, end: float) -> Path:
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    out = Path(name)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-to", str(end),
            "-ac", "1", "-ar", "16000", str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def play_clip(clip_path: Path) -> bool:
    afplay = shutil.which("afplay")
    if not afplay:
        return False
    subprocess.run([afplay, str(clip_path)], check=False)
    return True
