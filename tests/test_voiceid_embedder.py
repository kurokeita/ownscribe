"""Tests for the ECAPA embedder (model and audio loading mocked)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ownscribe.voiceid import embedder as embedder_mod
from ownscribe.voiceid.embedder import EcapaEmbedder


class _FakeModel:
    def encode_batch(self, wav):
        n = wav.shape[-1]
        vec = torch.tensor([float(n), 1.0, 2.0, 3.0])
        return vec.reshape(1, 1, 4)


def test_embed_returns_normalized_vector(tmp_path, monkeypatch):
    fake_audio = np.ones(16000, dtype=np.float32)
    monkeypatch.setattr(embedder_mod, "_load_audio_16k", lambda p: fake_audio)

    emb = EcapaEmbedder("fake/model")
    monkeypatch.setattr(emb, "_load", lambda: setattr(emb, "_model", _FakeModel()))

    out = emb.embed("audio.wav", [(0.0, 0.5)])
    assert out.shape == (4,)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_missing_speechbrain_raises_with_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("speechbrain"):
            raise ImportError("no speechbrain")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    emb = EcapaEmbedder("fake/model")
    with pytest.raises(ImportError, match="voiceid"):
        emb._load()
