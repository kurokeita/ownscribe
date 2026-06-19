"""Tests for pipeline orchestration."""

from __future__ import annotations

import contextlib
import json
from unittest import mock

import pytest

from ownscribe.config import Config
from ownscribe.transcription.models import Segment, TranscriptResult


class TestCreateRecorder:
    def test_coreaudio_when_available(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = ""

        with mock.patch("ownscribe.audio.coreaudio.CoreAudioRecorder") as mock_cls:
            mock_cls.return_value.is_available.return_value = True
            recorder = _create_recorder(config)
            assert recorder == mock_cls.return_value

    def test_fallback_to_sounddevice(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = ""

        with (
            mock.patch("ownscribe.audio.coreaudio.CoreAudioRecorder") as mock_ca,
            mock.patch("ownscribe.audio.sounddevice_recorder.SoundDeviceRecorder") as mock_sd,
        ):
            mock_ca.return_value.is_available.return_value = False
            recorder = _create_recorder(config)
            assert recorder == mock_sd.return_value

    def test_sounddevice_when_device_set(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = "USB Mic"

        with mock.patch("ownscribe.audio.sounddevice_recorder.SoundDeviceRecorder") as mock_sd:
            recorder = _create_recorder(config)
            assert recorder == mock_sd.return_value

    def test_silence_timeout_passed_to_coreaudio(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = ""
        config.audio.silence_timeout = 120

        with mock.patch("ownscribe.audio.coreaudio.CoreAudioRecorder") as mock_cls:
            mock_cls.return_value.is_available.return_value = True
            _create_recorder(config)
            mock_cls.assert_called_once_with(mic=False, mic_device="", capture_mode="picker", silence_timeout=120)

    def test_capture_mode_passed_to_coreaudio(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "coreaudio"
        config.audio.device = ""
        config.audio.capture_mode = "all"

        with mock.patch("ownscribe.audio.coreaudio.CoreAudioRecorder") as mock_cls:
            mock_cls.return_value.is_available.return_value = True
            _create_recorder(config)
            mock_cls.assert_called_once_with(mic=False, mic_device="", capture_mode="all", silence_timeout=300)

    def test_silence_timeout_passed_to_sounddevice(self):
        from ownscribe.pipeline import _create_recorder

        config = Config()
        config.audio.backend = "sounddevice"
        config.audio.device = "USB Mic"
        config.audio.silence_timeout = 60

        with mock.patch("ownscribe.audio.sounddevice_recorder.SoundDeviceRecorder") as mock_sd:
            _create_recorder(config)
            mock_sd.assert_called_once_with(device="USB Mic", silence_timeout=60)


class TestFormatOutput:
    def test_markdown_format(self, sample_transcript):
        from ownscribe.pipeline import _format_output

        config = Config()
        config.output.format = "markdown"

        transcript_str, summary_str = _format_output(config, sample_transcript)
        assert "# Transcript" in transcript_str
        assert summary_str is None

    def test_markdown_with_summary(self, sample_transcript):
        from ownscribe.pipeline import _format_output

        config = Config()
        config.output.format = "markdown"

        transcript_str, summary_str = _format_output(config, sample_transcript, "A great meeting.")
        assert "# Transcript" in transcript_str
        assert "# Meeting Summary" in summary_str
        assert "A great meeting." in summary_str

    def test_json_format(self, sample_transcript):
        from ownscribe.pipeline import _format_output

        config = Config()
        config.output.format = "json"

        transcript_str, _summary_str = _format_output(config, sample_transcript)
        parsed = json.loads(transcript_str)
        assert "segments" in parsed


class TestSlugify:
    def test_basic(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("Q3 Budget Planning Review") == "q3-budget-planning-review"

    def test_strips_special_chars(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("Hello, World! @#$") == "hello-world"

    def test_truncates_to_max_length(self):
        from ownscribe.pipeline import _slugify

        result = _slugify("a " * 100, max_length=10)
        assert len(result) <= 10

    def test_empty_input(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("") == ""

    def test_colons_removed(self):
        from ownscribe.pipeline import _slugify

        assert _slugify("Meeting: Budget Review") == "meeting-budget-review"


class TestGenerateTitleSlug:
    def test_returns_slug(self):
        from ownscribe.pipeline import _generate_title_slug

        mock_summarizer = mock.MagicMock()
        mock_summarizer.generate_title.return_value = "Budget Review"

        assert _generate_title_slug("summary text", mock_summarizer) == "budget-review"

    def test_returns_empty_on_empty_slug(self):
        from ownscribe.pipeline import _generate_title_slug

        mock_summarizer = mock.MagicMock()
        mock_summarizer.generate_title.return_value = "!!!"  # slugifies to empty

        assert _generate_title_slug("summary", mock_summarizer) == ""

    def test_returns_empty_on_llm_failure(self):
        from ownscribe.pipeline import _generate_title_slug

        mock_summarizer = mock.MagicMock()
        mock_summarizer.generate_title.side_effect = Exception("LLM down")

        assert _generate_title_slug("summary", mock_summarizer) == ""


class TestDoTranscribeAndSummarize:
    """Test _do_transcribe_and_summarize with mocked transcriber/summarizer."""

    def _make_transcript(self) -> TranscriptResult:
        return TranscriptResult(
            segments=[Segment(text="Hello world.", start=0.0, end=1.5)],
            language="en",
            duration=1.5,
        )

    def test_transcribe_only(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.md").exists()
        assert not (tmp_path / "summary.md").exists()

    def test_transcribe_and_summarize(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.summarization.enabled = True
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = True
        mock_summarizer.summarize.return_value = "## Summary\nGood meeting."

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline.create_summarizer", return_value=mock_summarizer),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
        ):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=True)

        assert (tmp_path / "transcript.md").exists()
        assert (tmp_path / "summary.md").exists()
        assert "Summary" in (tmp_path / "summary.md").read_text()

    def test_summarizer_unavailable_skips_gracefully(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.summarization.enabled = True
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = False

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline.create_summarizer", return_value=mock_summarizer),
        ):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=True)

        assert (tmp_path / "transcript.md").exists()
        assert not (tmp_path / "summary.md").exists()

    def test_json_output_format(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "json"
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.json").exists()
        assert not (tmp_path / "transcript.md").exists()

    def test_keep_recording_false_deletes_wav(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.output.keep_recording = False
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake audio data")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.md").exists()
        assert not audio_path.exists()

    def test_keep_recording_true_keeps_wav(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.output.keep_recording = True
        audio_path = tmp_path / "recording.wav"
        audio_path.write_bytes(b"fake audio data")

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=False)

        assert (tmp_path / "transcript.md").exists()
        assert audio_path.exists()

    def test_summarization_failure_preserves_transcript(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize

        config = Config()
        config.output.format = "markdown"
        config.summarization.enabled = True
        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = self._make_transcript()

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = True
        mock_summarizer.summarize.side_effect = Exception("GPU OOM")

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline.create_summarizer", return_value=mock_summarizer),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
        ):
            _do_transcribe_and_summarize(config, audio_path, tmp_path, summarize=True)

        assert (tmp_path / "transcript.md").exists()
        assert "Hello world." in (tmp_path / "transcript.md").read_text()
        assert not (tmp_path / "summary.md").exists()


class TestRunWarmup:
    def test_run_warmup_calls_prepare_models(self):
        from ownscribe.pipeline import run_warmup

        config = Config()
        config.transcription.language = "en"

        mock_transcriber = mock.MagicMock()

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
        ):
            run_warmup(config)

        mock_transcriber.prepare_models.assert_called_once_with(language="en")

    def test_run_warmup_enables_prepare_step_in_progress(self):
        from ownscribe.pipeline import run_warmup

        config = Config()
        mock_transcriber = mock.MagicMock()
        fake_progress = mock.MagicMock()

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline.PipelineProgress") as mock_progress_cls,
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
        ):
            mock_progress_cls.return_value.__enter__.return_value = fake_progress
            run_warmup(config)

        _, kwargs = mock_progress_cls.call_args
        assert kwargs["include_prepare"] is True
        assert kwargs["transcribe"] is False
        assert kwargs["download_summarizer"] is True

    def test_run_warmup_downloads_summarizer_with_progress(self):
        from ownscribe.pipeline import run_warmup

        config = Config()
        config.summarization.enabled = True
        config.summarization.backend = "local"
        config.summarization.model = "phi-4-mini"

        mock_transcriber = mock.MagicMock()

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model") as mock_ensure,
        ):
            run_warmup(config)

        mock_ensure.assert_called_once()
        _, kwargs = mock_ensure.call_args
        assert kwargs.get("on_progress") is not None

    def test_run_warmup_skips_summarizer_download_when_not_local(self):
        from ownscribe.pipeline import run_warmup

        config = Config()
        config.summarization.enabled = True
        config.summarization.backend = "ollama"

        mock_transcriber = mock.MagicMock()

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model") as mock_ensure,
        ):
            run_warmup(config)

        mock_ensure.assert_not_called()


class TestRunTranscribeOutputDir:
    """run_transcribe copies the source into a fresh, named ownscribe output dir."""

    def test_transcript_saved_in_named_output_dir(self, tmp_path):
        from ownscribe.pipeline import run_transcribe

        source_dir = tmp_path / "downloads"
        source_dir.mkdir()
        audio_path = source_dir / "Team Meeting.wav"
        audio_path.touch()

        out_base = tmp_path / "out"
        config = Config()
        config.output.dir = str(out_base)
        config.output.format = "markdown"

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptResult(
            segments=[Segment(text="Test.", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
        )

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline._check_audio_silence"),
        ):
            run_transcribe(config, str(audio_path))

        assert audio_path.exists()
        subdirs = [p for p in out_base.iterdir() if p.is_dir()]
        assert len(subdirs) == 1
        out_dir = subdirs[0]
        assert out_dir.name.endswith("_team-meeting")
        assert (out_dir / "transcript.md").exists()
        assert (out_dir / "recording.wav").exists()

    def test_transcript_in_place_when_within_output_dir(self, tmp_path):
        from ownscribe.pipeline import run_transcribe

        out_base = tmp_path / "out"
        existing = out_base / "2026-01-01_1200_team-meeting"
        existing.mkdir(parents=True)
        audio_path = existing / "recording.wav"
        audio_path.touch()

        config = Config()
        config.output.dir = str(out_base)
        config.output.format = "markdown"

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptResult(
            segments=[Segment(text="Test.", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
        )

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline._check_audio_silence"),
        ):
            run_transcribe(config, str(audio_path))

        assert [p for p in out_base.iterdir() if p.is_dir()] == [existing]
        assert (existing / "transcript.md").exists()

    def test_summarize_flag_also_writes_summary(self, tmp_path):
        from ownscribe.pipeline import run_transcribe

        source_dir = tmp_path / "downloads"
        source_dir.mkdir()
        audio_path = source_dir / "Team Meeting.wav"
        audio_path.touch()

        out_base = tmp_path / "out"
        config = Config()
        config.output.dir = str(out_base)
        config.output.format = "markdown"
        config.summarization.enabled = True

        mock_transcriber = mock.MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptResult(
            segments=[Segment(text="Test.", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
        )
        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = True
        mock_summarizer.summarize.return_value = "## Summary\nGood meeting."
        mock_summarizer.generate_title.return_value = "team-sync"

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=mock_transcriber),
            mock.patch("ownscribe.pipeline.create_summarizer", return_value=mock_summarizer),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
            mock.patch("ownscribe.pipeline._check_audio_silence"),
        ):
            run_transcribe(config, str(audio_path), summarize=True)

        subdirs = [p for p in out_base.iterdir() if p.is_dir()]
        assert len(subdirs) == 1
        out_dir = subdirs[0]
        assert out_dir.name.endswith("_team-sync")
        assert (out_dir / "transcript.md").exists()
        assert (out_dir / "summary.md").exists()


class TestRunSummarizeOutputDir:
    """run_summarize copies the source into a fresh, named ownscribe output dir."""

    def test_summary_saved_in_named_output_dir(self, tmp_path):
        from ownscribe.pipeline import run_summarize

        source_dir = tmp_path / "notes"
        source_dir.mkdir()
        tx_path = source_dir / "Standup Notes.md"
        tx_path.write_text("# Transcript\nHello world.")

        out_base = tmp_path / "out"
        config = Config()
        config.output.dir = str(out_base)
        config.summarization.enabled = True

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = True
        mock_summarizer.summarize.return_value = "## Summary\nGood meeting."
        mock_summarizer.generate_title.return_value = "test-title"

        with (
            mock.patch("ownscribe.pipeline.create_summarizer", return_value=mock_summarizer),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
        ):
            run_summarize(config, str(tx_path))

        assert tx_path.exists()
        subdirs = [p for p in out_base.iterdir() if p.is_dir()]
        assert len(subdirs) == 1
        out_dir = subdirs[0]
        assert out_dir.name.endswith("_test-title")
        assert "standup-notes" not in out_dir.name
        assert (out_dir / "summary.md").exists()
        assert (out_dir / "transcript.md").exists()

    def test_in_place_summarizes_in_source_dir(self, tmp_path):
        from ownscribe.pipeline import run_summarize

        tx_dir = tmp_path / "meetings" / "2026-01-01_1200"
        tx_dir.mkdir(parents=True)
        tx_path = tx_dir / "transcript.md"
        tx_path.write_text("# Transcript\nHello world.")

        config = Config()
        config.summarization.enabled = True

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = True
        mock_summarizer.summarize.return_value = "## Summary\nGood meeting."
        mock_summarizer.generate_title.return_value = "test-title"

        with (
            mock.patch("ownscribe.pipeline.create_summarizer", return_value=mock_summarizer),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
        ):
            run_summarize(config, str(tx_path), in_place=True)

        renamed_dir = tx_dir.parent / f"{tx_dir.name}_test-title"
        assert (renamed_dir / "summary.md").exists()

    def test_summary_in_place_when_within_output_dir(self, tmp_path):
        from ownscribe.pipeline import run_summarize

        out_base = tmp_path / "out"
        existing = out_base / "2026-01-01_1200_team-meeting"
        existing.mkdir(parents=True)
        tx_path = existing / "transcript.md"
        tx_path.write_text("# Transcript\nHello world.")

        config = Config()
        config.output.dir = str(out_base)
        config.summarization.enabled = True

        mock_summarizer = mock.MagicMock()
        mock_summarizer.is_available.return_value = True
        mock_summarizer.summarize.return_value = "## Summary\nGood meeting."
        mock_summarizer.generate_title.return_value = "final-title"

        with (
            mock.patch("ownscribe.pipeline.create_summarizer", return_value=mock_summarizer),
            mock.patch("ownscribe.summarization.llama_cpp_summarizer._ensure_model"),
        ):
            run_summarize(config, str(tx_path))

        renamed = out_base / "2026-01-01_1200_final-title"
        assert [p for p in out_base.iterdir() if p.is_dir()] == [renamed]
        assert (renamed / "summary.md").exists()


class TestResume:
    """Test run_resume artifact detection and dispatch."""

    def test_nothing_to_resume(self, tmp_path):
        from ownscribe.pipeline import run_resume

        (tmp_path / "transcript.md").write_text("hello")
        (tmp_path / "summary.md").write_text("summary")

        config = Config()
        run_resume(config, str(tmp_path))
        # Should exit cleanly without error

    def test_error_no_audio_no_transcript(self, tmp_path):
        from ownscribe.pipeline import run_resume

        config = Config()
        with mock.patch("sys.exit", side_effect=SystemExit(1)), contextlib.suppress(SystemExit):
            run_resume(config, str(tmp_path))

    def test_resumes_summarize_only(self, tmp_path):
        from ownscribe.pipeline import run_resume

        (tmp_path / "transcript.md").write_text("# Transcript\nHello.")

        config = Config()
        config.summarization.enabled = True

        with mock.patch("ownscribe.pipeline.run_summarize") as mock_sum:
            run_resume(config, str(tmp_path))
            mock_sum.assert_called_once_with(config, str(tmp_path / "transcript.md"), in_place=True)

    def test_resumes_transcribe_and_summarize(self, tmp_path):
        from ownscribe.pipeline import run_resume

        audio_path = tmp_path / "recording.wav"
        audio_path.touch()

        config = Config()

        with mock.patch("ownscribe.pipeline._do_transcribe_and_summarize") as mock_ts:
            run_resume(config, str(tmp_path))
            mock_ts.assert_called_once_with(config, audio_path, tmp_path)

    def test_finds_non_wav_audio(self, tmp_path):
        from ownscribe.pipeline import run_resume

        audio_path = tmp_path / "meeting.mp3"
        audio_path.touch()

        config = Config()

        with mock.patch("ownscribe.pipeline._do_transcribe_and_summarize") as mock_ts:
            run_resume(config, str(tmp_path))
            mock_ts.assert_called_once_with(config, audio_path, tmp_path)

    def test_finds_json_transcript(self, tmp_path):
        from ownscribe.pipeline import run_resume

        (tmp_path / "transcript.json").write_text('{"segments": []}')

        config = Config()

        with mock.patch("ownscribe.pipeline.run_summarize") as mock_sum:
            run_resume(config, str(tmp_path))
            mock_sum.assert_called_once_with(config, str(tmp_path / "transcript.json"), in_place=True)


class TestVoiceIdentification:
    """Sidecar writing and opt-in speaker relabeling in the transcribe path."""

    def _diarized_result(self) -> TranscriptResult:
        return TranscriptResult(
            segments=[
                Segment(text="hi", start=0.0, end=2.0, speaker="SPEAKER_00"),
                Segment(text="yo", start=2.0, end=4.0, speaker="SPEAKER_01"),
            ],
            language="en",
            duration=4.0,
        )

    def test_do_transcribe_writes_sidecar_when_diarized(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize
        from ownscribe.voiceid.sidecar import DIARIZATION_FILENAME

        config = Config()
        config.summarization.enabled = False
        audio = tmp_path / "recording.wav"
        audio.write_bytes(b"x" * 100)

        transcriber = mock.MagicMock()
        transcriber.transcribe.return_value = self._diarized_result()
        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=transcriber):
            _do_transcribe_and_summarize(config, audio, tmp_path, summarize=False)

        assert (tmp_path / DIARIZATION_FILENAME).exists()

    def test_do_transcribe_no_sidecar_without_speakers(self, tmp_path):
        from ownscribe.pipeline import _do_transcribe_and_summarize
        from ownscribe.voiceid.sidecar import DIARIZATION_FILENAME

        config = Config()
        config.summarization.enabled = False
        audio = tmp_path / "recording.wav"
        audio.write_bytes(b"x" * 100)

        transcriber = mock.MagicMock()
        transcriber.transcribe.return_value = TranscriptResult(
            segments=[Segment(text="hi", start=0.0, end=2.0)],
            language="en",
            duration=2.0,
        )
        with mock.patch("ownscribe.pipeline._create_transcriber", return_value=transcriber):
            _do_transcribe_and_summarize(config, audio, tmp_path, summarize=False)

        assert not (tmp_path / DIARIZATION_FILENAME).exists()

    def test_do_transcribe_relabels_when_identify(self, tmp_path):
        from ownscribe import pipeline

        config = Config()
        config.summarization.enabled = False
        audio = tmp_path / "recording.wav"
        audio.write_bytes(b"x" * 100)

        transcriber = mock.MagicMock()
        transcriber.transcribe.return_value = self._diarized_result()

        store = mock.MagicMock()
        store.list_names.return_value = ["Alice"]

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=transcriber),
            mock.patch.object(pipeline, "_build_identify_tools", return_value=(mock.MagicMock(), store)),
            mock.patch(
                "ownscribe.voiceid.identify.build_relabel_map",
                return_value={"SPEAKER_00": "Alice"},
            ),
        ):
            pipeline._do_transcribe_and_summarize(
                config, audio, tmp_path, summarize=False, identify=True
            )

        transcript = (tmp_path / "transcript.md").read_text()
        assert "Alice" in transcript
        assert "SPEAKER_00" not in transcript

    def test_auto_identify_relabels_when_identify_none(self, tmp_path):
        from ownscribe import pipeline

        config = Config()
        config.summarization.enabled = False
        config.voice.auto_identify = True
        audio = tmp_path / "recording.wav"
        audio.write_bytes(b"x" * 100)

        transcriber = mock.MagicMock()
        transcriber.transcribe.return_value = self._diarized_result()

        store = mock.MagicMock()
        store.list_names.return_value = ["Alice"]

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=transcriber),
            mock.patch.object(pipeline, "_build_identify_tools", return_value=(mock.MagicMock(), store)),
            mock.patch(
                "ownscribe.voiceid.identify.build_relabel_map",
                return_value={"SPEAKER_00": "Alice"},
            ),
        ):
            pipeline._do_transcribe_and_summarize(
                config, audio, tmp_path, summarize=False, identify=None
            )

        assert "Alice" in (tmp_path / "transcript.md").read_text()

    def test_no_identify_overrides_auto_identify(self, tmp_path):
        from ownscribe import pipeline

        config = Config()
        config.summarization.enabled = False
        config.voice.auto_identify = True
        audio = tmp_path / "recording.wav"
        audio.write_bytes(b"x" * 100)

        transcriber = mock.MagicMock()
        transcriber.transcribe.return_value = self._diarized_result()

        with (
            mock.patch("ownscribe.pipeline._create_transcriber", return_value=transcriber),
            mock.patch.object(pipeline, "_build_identify_tools") as build_tools,
        ):
            pipeline._do_transcribe_and_summarize(
                config, audio, tmp_path, summarize=False, identify=False
            )

        build_tools.assert_not_called()
        assert "SPEAKER_00" in (tmp_path / "transcript.md").read_text()

    def test_maybe_identify_empty_store_notifies(self, tmp_path, capsys):
        from ownscribe import pipeline

        config = Config()
        config.voice.auto_identify = True
        config.voice.dir = str(tmp_path / "voices")
        result = self._diarized_result()

        pipeline._maybe_identify(config, result, tmp_path / "recording.wav", None)

        assert result.segments[0].speaker == "SPEAKER_00"
        assert "analyze" in capsys.readouterr().err.lower()

    def test_maybe_identify_missing_speechbrain_warns(self, capsys):
        from ownscribe import pipeline

        config = Config()
        config.voice.auto_identify = True
        result = self._diarized_result()

        embedder = mock.MagicMock()
        embedder.ensure_available.side_effect = ImportError(
            "speechbrain is required ... ownscribe[voiceid]"
        )
        store = mock.MagicMock()
        store.list_names.return_value = ["Alice"]

        with mock.patch.object(pipeline, "_build_identify_tools", return_value=(embedder, store)):
            pipeline._maybe_identify(config, result, "recording.wav", None)

        assert result.segments[0].speaker == "SPEAKER_00"
        assert "skipped" in capsys.readouterr().err.lower()


class TestRunAnalyze:
    """Interactive enrollment from a diarized meeting directory."""

    def test_transcript_name_suggestions_maps_renamed_headers(self, tmp_path):
        from ownscribe import pipeline

        (tmp_path / "transcript.md").write_text(
            "# Transcript\n\n**Alice** [00:00]\nhi\n\n**SPEAKER_01** [00:02]\nyo\n"
        )
        ranges = {"SPEAKER_00": [(0.0, 2.0)], "SPEAKER_01": [(2.0, 4.0)]}
        suggestions = pipeline._transcript_name_suggestions(tmp_path, ranges)
        assert suggestions == {"SPEAKER_00": "Alice"}

    def test_run_analyze_enrolls_named_speakers(self, tmp_path, monkeypatch):
        from ownscribe import pipeline
        from ownscribe.voiceid.sidecar import DIARIZATION_FILENAME
        from ownscribe.voiceid.store import VoiceStore

        config = Config()
        config.voice.dir = str(tmp_path / "voices")
        (tmp_path / DIARIZATION_FILENAME).write_text(
            '{"segments": [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]}'
        )
        (tmp_path / "recording.wav").write_bytes(b"x" * 100)

        import numpy as np

        embedder = mock.MagicMock()
        embedder.embed.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store = VoiceStore(config.voice.resolved_dir)

        monkeypatch.setattr(pipeline, "_build_identify_tools", lambda cfg: (embedder, store))
        monkeypatch.setattr("ownscribe.voiceid.playback.extract_clip", lambda *a: tmp_path / "clip.wav")
        monkeypatch.setattr("ownscribe.voiceid.playback.play_clip", lambda *a: True)
        monkeypatch.setattr("click.prompt", lambda *a, **k: "Alice")

        pipeline.run_analyze(config, str(tmp_path))

        assert VoiceStore(config.voice.resolved_dir).list_names() == ["Alice"]

    def test_run_analyze_missing_speechbrain_exits_cleanly(self, tmp_path, monkeypatch, capsys):
        from ownscribe import pipeline
        from ownscribe.voiceid.sidecar import DIARIZATION_FILENAME

        config = Config()
        config.voice.dir = str(tmp_path / "voices")
        (tmp_path / DIARIZATION_FILENAME).write_text(
            '{"segments": [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]}'
        )
        (tmp_path / "recording.wav").write_bytes(b"x" * 100)

        embedder = mock.MagicMock()
        embedder.ensure_available.side_effect = ImportError(
            "speechbrain is required ... ownscribe[voiceid]"
        )
        store = mock.MagicMock()
        monkeypatch.setattr(pipeline, "_build_identify_tools", lambda cfg: (embedder, store))

        with pytest.raises(SystemExit):
            pipeline.run_analyze(config, str(tmp_path))

        assert "voiceid" in capsys.readouterr().err.lower()
