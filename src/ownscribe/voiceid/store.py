"""Persistent voice-profile store backed by per-person JSON files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "voice"


@dataclass
class VoiceSample:
    embedding: list[float]
    source_dir: str
    enrolled_at: str
    duration_s: float


@dataclass
class VoiceProfile:
    name: str
    samples: list[VoiceSample] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        mat = np.array([s.embedding for s in self.samples], dtype=np.float32)
        mean = mat.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean


class VoiceStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path_for(self, name: str) -> Path:
        return self.directory / f"{slugify_name(name)}.json"

    def _load(self, name: str) -> VoiceProfile | None:
        path = self._path_for(name)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        samples = [VoiceSample(**s) for s in data.get("samples", [])]
        return VoiceProfile(name=data["name"], samples=samples)

    def _all_profiles(self) -> list[VoiceProfile]:
        if not self.directory.exists():
            return []
        profiles = []
        for path in sorted(self.directory.glob("*.json")):
            data = json.loads(path.read_text())
            samples = [VoiceSample(**s) for s in data.get("samples", [])]
            profiles.append(VoiceProfile(name=data["name"], samples=samples))
        return profiles

    def list_names(self) -> list[str]:
        return [p.name for p in self._all_profiles()]

    def profile_vector(self, name: str) -> np.ndarray | None:
        profile = self._load(name)
        if profile is None or not profile.samples:
            return None
        return profile.vector()

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        *,
        source_dir: str,
        duration_s: float,
        now: str | None = None,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        profile = self._load(name) or VoiceProfile(name=name)
        profile.samples.append(
            VoiceSample(
                embedding=[float(x) for x in np.asarray(embedding).ravel()],
                source_dir=source_dir,
                enrolled_at=now or datetime.now().isoformat(timespec="seconds"),
                duration_s=float(duration_s),
            )
        )
        data = {
            "name": profile.name,
            "samples": [vars(s) for s in profile.samples],
        }
        self._path_for(name).write_text(json.dumps(data, indent=2))

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str, float] | None:
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        best: tuple[str, float] | None = None
        for profile in self._all_profiles():
            if not profile.samples:
                continue
            score = float(np.dot(emb, profile.vector()))
            if best is None or score > best[1]:
                best = (profile.name, score)
        if best is not None and best[1] >= threshold:
            return best
        return None
