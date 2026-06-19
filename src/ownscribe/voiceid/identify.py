"""Match diarized speaker clusters against enrolled voices and relabel them."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ownscribe.transcription.models import TranscriptResult
from ownscribe.voiceid.store import VoiceStore


class Embedder(Protocol):
    def embed(self, audio_path: Path, segments: list[tuple[float, float]]) -> np.ndarray: ...


def speaker_ranges(result: TranscriptResult) -> dict[str, list[tuple[float, float]]]:
    ranges: dict[str, list[tuple[float, float]]] = {}
    for seg in result.segments:
        if seg.speaker:
            ranges.setdefault(seg.speaker, []).append((seg.start, seg.end))
    return ranges


def build_relabel_map(
    result: TranscriptResult,
    audio_path: Path,
    embedder: Embedder,
    store: VoiceStore,
    threshold: float,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for speaker, ranges in speaker_ranges(result).items():
        embedding = embedder.embed(audio_path, ranges)
        match = store.match(embedding, threshold)
        if match is not None:
            mapping[speaker] = match[0]
    return mapping


def apply_relabel_map(result: TranscriptResult, mapping: dict[str, str]) -> None:
    for seg in result.segments:
        if seg.speaker in mapping:
            seg.speaker = mapping[seg.speaker]
        for word in seg.words:
            if word.speaker in mapping:
                word.speaker = mapping[word.speaker]
