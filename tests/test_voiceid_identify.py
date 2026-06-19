"""Tests for identification relabeling."""

from __future__ import annotations

import numpy as np

from ownscribe.transcription.models import Segment, TranscriptResult, Word
from ownscribe.voiceid.identify import (
    apply_relabel_map,
    build_relabel_map,
    speaker_ranges,
)
from ownscribe.voiceid.store import VoiceStore


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


class _FakeEmbedder:
    def __init__(self, by_start: dict[float, np.ndarray]) -> None:
        self.by_start = by_start

    def embed(self, audio_path, ranges):
        return self.by_start[ranges[0][0]]


def _diarized() -> TranscriptResult:
    return TranscriptResult(
        segments=[
            Segment(
                text="a", start=0.0, end=2.0, speaker="SPEAKER_00",
                words=[Word(text="a", start=0.0, end=2.0, speaker="SPEAKER_00")],
            ),
            Segment(text="b", start=2.0, end=4.0, speaker="SPEAKER_01"),
        ],
        language="en",
        duration=4.0,
    )


def test_speaker_ranges_groups_by_label():
    ranges = speaker_ranges(_diarized())
    assert ranges == {"SPEAKER_00": [(0.0, 2.0)], "SPEAKER_01": [(2.0, 4.0)]}


def test_build_relabel_map_matches_enrolled(tmp_path):
    store = VoiceStore(tmp_path)
    store.enroll("Alice", _unit(1, 0, 0), source_dir="d", duration_s=5.0, now="t")
    embedder = _FakeEmbedder({0.0: _unit(0.95, 0.05, 0), 2.0: _unit(0, 0, 1)})
    mapping = build_relabel_map(_diarized(), "audio.wav", embedder, store, threshold=0.25)
    assert mapping == {"SPEAKER_00": "Alice"}


def test_apply_relabel_map_rewrites_segments_and_words():
    result = _diarized()
    apply_relabel_map(result, {"SPEAKER_00": "Alice"})
    assert result.segments[0].speaker == "Alice"
    assert result.segments[0].words[0].speaker == "Alice"
    assert result.segments[1].speaker == "SPEAKER_01"
