"""Tests for the voice profile store."""

from __future__ import annotations

import numpy as np

from ownscribe.voiceid.store import VoiceStore, slugify_name


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_slugify_name():
    assert slugify_name("Alice Smith") == "alice-smith"
    assert slugify_name("  Bob!! ") == "bob"
    assert slugify_name("???") == "voice"


def test_enroll_creates_file_and_lists_name(tmp_path):
    store = VoiceStore(tmp_path)
    store.enroll("Alice", _unit(1, 0, 0), source_dir="d1", duration_s=5.0, now="2026-06-19T10:00:00")
    assert store.list_names() == ["Alice"]
    assert (tmp_path / "alice.json").exists()


def test_profile_vector_is_mean_of_samples(tmp_path):
    store = VoiceStore(tmp_path)
    store.enroll("Alice", _unit(1, 0, 0), source_dir="d1", duration_s=5.0, now="t1")
    store.enroll("Alice", _unit(0, 1, 0), source_dir="d2", duration_s=5.0, now="t2")
    vec = store.profile_vector("Alice")
    assert np.allclose(vec, _unit(1, 1, 0), atol=1e-5)


def test_profile_vector_missing_returns_none(tmp_path):
    assert VoiceStore(tmp_path).profile_vector("Nobody") is None


def test_match_returns_best_above_threshold(tmp_path):
    store = VoiceStore(tmp_path)
    store.enroll("Alice", _unit(1, 0, 0), source_dir="d", duration_s=5.0, now="t")
    store.enroll("Bob", _unit(0, 1, 0), source_dir="d", duration_s=5.0, now="t")
    result = store.match(_unit(0.9, 0.1, 0), threshold=0.25)
    assert result is not None
    name, score = result
    assert name == "Alice"
    assert score > 0.9


def test_match_below_threshold_returns_none(tmp_path):
    store = VoiceStore(tmp_path)
    store.enroll("Alice", _unit(1, 0, 0), source_dir="d", duration_s=5.0, now="t")
    assert store.match(_unit(0, 0, 1), threshold=0.25) is None


def test_match_empty_store_returns_none(tmp_path):
    assert VoiceStore(tmp_path).match(_unit(1, 0, 0), threshold=0.25) is None


def test_enroll_same_slug_appends_samples(tmp_path):
    store = VoiceStore(tmp_path)
    store.enroll("Alice", _unit(1, 0, 0), source_dir="d", duration_s=5.0, now="t")
    store.enroll("alice", _unit(0, 1, 0), source_dir="d", duration_s=5.0, now="t")
    assert len(store._load("alice").samples) == 2
