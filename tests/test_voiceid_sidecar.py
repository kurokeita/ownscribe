"""Tests for the diarization sidecar."""

from __future__ import annotations

import json

from ownscribe.transcription.models import Segment, TranscriptResult
from ownscribe.voiceid.sidecar import (
    DIARIZATION_FILENAME,
    read_speaker_ranges,
    write_sidecar,
)


def _diarized() -> TranscriptResult:
    return TranscriptResult(
        segments=[
            Segment(text="a", start=0.0, end=2.0, speaker="SPEAKER_00"),
            Segment(text="b", start=2.0, end=3.5, speaker="SPEAKER_01"),
            Segment(text="c", start=3.5, end=6.0, speaker="SPEAKER_00"),
            Segment(text="no-speaker", start=6.0, end=7.0, speaker=None),
        ],
        language="en",
        duration=7.0,
    )


def test_filename_constant():
    assert DIARIZATION_FILENAME == "diarization.json"


def test_write_only_includes_speakered_segments(tmp_path):
    path = tmp_path / DIARIZATION_FILENAME
    write_sidecar(_diarized(), path)
    data = json.loads(path.read_text())
    assert len(data["segments"]) == 3
    assert all(s["speaker"] for s in data["segments"])


def test_read_groups_ranges_by_speaker(tmp_path):
    path = tmp_path / DIARIZATION_FILENAME
    write_sidecar(_diarized(), path)
    ranges = read_speaker_ranges(path)
    assert ranges["SPEAKER_00"] == [(0.0, 2.0), (3.5, 6.0)]
    assert ranges["SPEAKER_01"] == [(2.0, 3.5)]
