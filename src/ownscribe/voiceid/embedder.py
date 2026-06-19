"""SpeechBrain ECAPA-TDNN speaker-embedding wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_audio_16k(audio_path: Path) -> np.ndarray:
    import whisperx

    return whisperx.load_audio(str(audio_path))


class EcapaEmbedder:
    _SAMPLE_RATE = 16000

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def ensure_available(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "speechbrain is required for voice identification. "
                "Install with: uv pip install 'ownscribe[voiceid]'"
            ) from exc

    def _load(self) -> None:
        if self._model is not None:
            return
        self.ensure_available()
        from speechbrain.inference.speaker import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(source=self._model_name)

    def embed(self, audio_path: Path, segments: list[tuple[float, float]]) -> np.ndarray:
        import torch

        self._load()
        audio = _load_audio_16k(audio_path)

        chunks = []
        for start, end in segments:
            i0 = int(start * self._SAMPLE_RATE)
            i1 = int(end * self._SAMPLE_RATE)
            if i1 > i0:
                chunks.append(audio[i0:i1])
        clip = np.concatenate(chunks) if chunks else audio

        wav = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0)
        emb = self._model.encode_batch(wav).squeeze().detach().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb
