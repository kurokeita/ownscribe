"""Tests for clip selection, extraction, and playback."""

from __future__ import annotations

from ownscribe.voiceid import playback


def test_representative_range_picks_longest():
    assert playback.representative_range([(0.0, 1.0), (2.0, 6.0), (7.0, 7.5)]) == (2.0, 6.0)


def test_total_duration_sums_ranges():
    assert playback.total_duration([(0.0, 1.0), (2.0, 4.5)]) == 3.5


def test_extract_clip_invokes_ffmpeg(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return None

    monkeypatch.setattr(playback.subprocess, "run", fake_run)
    out = playback.extract_clip(tmp_path / "rec.wav", 1.0, 3.0)
    assert "ffmpeg" in calls["cmd"][0]
    assert "-ss" in calls["cmd"] and "1.0" in calls["cmd"]
    assert "-to" in calls["cmd"] and "3.0" in calls["cmd"]
    assert str(out).endswith(".wav")


def test_play_clip_returns_false_when_afplay_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(playback.shutil, "which", lambda name: None)
    assert playback.play_clip(tmp_path / "clip.wav") is False


def test_play_clip_runs_afplay_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(playback.shutil, "which", lambda name: "/usr/bin/afplay")
    ran = {}
    monkeypatch.setattr(playback.subprocess, "run", lambda cmd, **kw: ran.update(cmd=cmd))
    assert playback.play_clip(tmp_path / "clip.wav") is True
    assert ran["cmd"][0] == "/usr/bin/afplay"
