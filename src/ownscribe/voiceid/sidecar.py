"""Machine-readable diarization sidecar.

Stores original SPEAKER_xx labels and their time-ranges so that enrollment can
correlate names back to audio after a user renames speakers in the transcript.
"""

from __future__ import annotations

import json
from pathlib import Path

from ownscribe.transcription.models import TranscriptResult

DIARIZATION_FILENAME = "diarization.json"


def write_sidecar(result: TranscriptResult, path: Path) -> None:
    data = {
        "segments": [
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker}
            for seg in result.segments
            if seg.speaker
        ]
    }
    path.write_text(json.dumps(data, indent=2))


def read_speaker_ranges(path: Path) -> dict[str, list[tuple[float, float]]]:
    data = json.loads(path.read_text())
    ranges: dict[str, list[tuple[float, float]]] = {}
    for seg in data.get("segments", []):
        speaker = seg.get("speaker")
        if speaker:
            ranges.setdefault(speaker, []).append((seg["start"], seg["end"]))
    return ranges
