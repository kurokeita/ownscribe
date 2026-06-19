"""Pipeline orchestration: record -> transcribe -> summarize -> output."""

from __future__ import annotations

import logging
import re
import select
import shutil
import signal
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import click

from ownscribe.config import Config
from ownscribe.progress import (
    DownloadProgressEvent,
    PipelineProgress,
    download_event_fraction,
    format_download_progress,
)
from ownscribe.summarization import create_summarizer

# A standard WAV file header (RIFF + fmt + data chunk header) is 44 bytes.
# Files at or below this size contain no audio frames.
_WAV_HEADER_SIZE = 44


def _check_audio_silence(audio_path: Path) -> None:
    """Check if the recorded audio is silent and warn the user."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return  # Skip check if deps not available

    try:
        # Read up to 5 seconds
        info = sf.info(audio_path)
        frames_to_read = min(int(info.samplerate * 5), info.frames)
        data, _ = sf.read(audio_path, frames=frames_to_read, dtype="float32")
        peak = float(np.max(np.abs(data)))
    except Exception:
        return  # Don't block pipeline on check failure

    if peak < 1e-6:
        click.echo(
            "\nError: Recorded audio is completely silent (peak amplitude ~0).\n"
            "This usually means Screen Recording permission is missing.\n"
            "Fix: System Settings > Privacy & Security > Screen Recording "
            "— enable your terminal app, then restart it.\n",
            err=True,
        )
        raise SystemExit(1)


def _get_output_dir(config: Config) -> Path:
    """Create and return a timestamped output directory."""
    base = config.output.resolved_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _get_named_output_dir(config: Config, name: str) -> Path:
    """Create a timestamped output directory named after a source file."""
    base = config.output.resolved_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    slug = _slugify(name) or "file"
    out_dir = base / f"{timestamp}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _within_output_dir(config: Config, path: Path) -> bool:
    """True if path already lives inside the configured output directory."""
    try:
        path.resolve().relative_to(config.output.resolved_dir.resolve())
        return True
    except ValueError:
        return False


def _create_recorder(config: Config):
    """Create the appropriate audio recorder based on config."""
    if config.audio.backend == "coreaudio" and not config.audio.device:
        from ownscribe.audio.coreaudio import CoreAudioRecorder

        recorder = CoreAudioRecorder(
            mic=config.audio.mic,
            mic_device=config.audio.mic_device,
            capture_mode=config.audio.capture_mode,
            silence_timeout=config.audio.silence_timeout,
        )
        if recorder.is_available():
            return recorder
        click.echo("Core Audio helper not found, falling back to sounddevice.")

    from ownscribe.audio.sounddevice_recorder import SoundDeviceRecorder

    device = config.audio.device or None
    # Try to parse as int (device index)
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    return SoundDeviceRecorder(device=device, silence_timeout=config.audio.silence_timeout)


def _create_transcriber(config: Config, progress=None):
    """Create the WhisperX transcriber."""
    from ownscribe.transcription.whisperx_transcriber import WhisperXTranscriber

    diar_config = config.diarization if config.diarization.enabled else None
    return WhisperXTranscriber(config.transcription, diar_config, progress=progress)


def _download_summarization_model(
    model_name: str,
    progress: PipelineProgress,
    step_key: str,
) -> Path:
    """Download the summarization GGUF model, feeding progress into the TUI."""
    from ownscribe.summarization.llama_cpp_summarizer import _ensure_model

    def _on_progress(event: DownloadProgressEvent) -> None:
        fraction = download_event_fraction(event)
        if fraction is not None:
            progress.update(step_key, fraction)
        formatted = format_download_progress(event, include_percent=fraction is None)
        if formatted:
            progress.set_detail(step_key, formatted)

    return _ensure_model(model_name, on_progress=_on_progress)


def _format_output(config: Config, transcript_result, summary_text: str | None = None) -> tuple[str, str | None]:
    """Format transcript and optional summary. Returns (transcript_str, summary_str)."""
    if config.output.format == "json":
        from ownscribe.output.json_output import format_transcript_json

        return format_transcript_json(transcript_result), summary_text
    else:
        from ownscribe.output.markdown import format_summary, format_transcript

        tx = format_transcript(transcript_result)
        sm = format_summary(summary_text) if summary_text else None
        return tx, sm


def _build_identify_tools(config: Config):
    from ownscribe.voiceid.embedder import EcapaEmbedder
    from ownscribe.voiceid.store import VoiceStore

    embedder = EcapaEmbedder(config.voice.model)
    store = VoiceStore(config.voice.resolved_dir)
    return embedder, store


def _maybe_identify(config: Config, result, audio_path: Path, identify: bool | None) -> None:
    want = config.voice.auto_identify if identify is None else identify
    if not (want and result.has_speakers):
        return

    embedder, store = _build_identify_tools(config)
    if not store.list_names():
        click.echo(
            "\nVoice identification is on, but no voices are enrolled yet. "
            "Run 'ownscribe analyze <meeting-dir>' to enroll speakers.",
            err=True,
        )
        return
    try:
        embedder.ensure_available()
    except ImportError as exc:
        click.echo(f"\nWarning: voice identification skipped, {exc}", err=True)
        return

    from ownscribe.voiceid.identify import apply_relabel_map, build_relabel_map

    mapping = build_relabel_map(result, audio_path, embedder, store, config.voice.threshold)
    if mapping:
        apply_relabel_map(result, mapping)


def _slugify(text: str, max_length: int = 50) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:max_length].rstrip("-")


def _generate_title_slug(summary: str, summarizer) -> str:
    """Generate a title slug from a summary. Returns empty string on failure."""
    try:
        title = summarizer.generate_title(summary)
        return _slugify(title)
    except Exception:
        logging.getLogger(__name__).warning("Could not generate title", exc_info=True)
        return ""


_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}")


def _rename_with_title(out_dir: Path, title_slug: str) -> Path:
    """Rename out_dir to '<timestamp>_<title_slug>', preserving the timestamp prefix."""
    match = _TIMESTAMP_PREFIX.match(out_dir.name)
    prefix = match.group(0) if match else out_dir.name
    new_dir = out_dir.parent / f"{prefix}_{title_slug}"
    if new_dir == out_dir:
        return out_dir
    try:
        out_dir.rename(new_dir)
        return new_dir
    except Exception:
        logging.getLogger(__name__).warning("Could not rename output directory", exc_info=True)
        return out_dir


def run_pipeline(config: Config) -> None:
    """Run the full pipeline: record, transcribe, summarize, output."""
    out_dir = _get_output_dir(config)
    audio_path = out_dir / "recording.wav"

    # 1. Record
    recorder = _create_recorder(config)

    from ownscribe.audio.coreaudio import CoreAudioRecorder

    can_mute = isinstance(recorder, CoreAudioRecorder) and config.audio.mic
    is_tty = sys.stdin.isatty()

    hints = []
    if can_mute and is_tty:
        hints.append("Press 'm' to mute/unmute mic.")
    silence_timeout = config.audio.silence_timeout
    if silence_timeout > 0:
        mins, secs = divmod(int(silence_timeout), 60)
        if mins > 0 and secs > 0:
            hints.append(f"Auto-stops after {mins}m {secs}s of silence.")
        elif mins > 0:
            hints.append(f"Auto-stops after {mins}m of silence.")
        else:
            hints.append(f"Auto-stops after {silence_timeout}s of silence.")
    hints.append("Press Ctrl+C to stop.")
    click.echo(f"Starting recording... {' '.join(hints)}\n")
    recorder.start(audio_path)

    start_time = time.time()
    stop_event = False

    def on_interrupt(sig, frame):
        nonlocal stop_event
        stop_event = True

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, on_interrupt)

    old_termios = None
    if can_mute and is_tty:
        old_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    warned_no_data = False
    try:
        while not stop_event and recorder.is_recording:
            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            mute_indicator = "  [MIC MUTED]" if recorder.is_muted else ""
            click.echo(
                f"\r  Recording: {mins:02d}:{secs:02d}{mute_indicator}\033[K",
                nl=False,
            )
            if (
                not warned_no_data
                and elapsed >= 3
                and audio_path.exists()
                and audio_path.stat().st_size <= _WAV_HEADER_SIZE
            ):
                click.echo(
                    "\n\n  Warning: No audio data received yet.\n",
                    err=True,
                )
                warned_no_data = True
            if old_termios is not None:
                readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                if readable:
                    ch = sys.stdin.read(1)
                    if ch in ("m", "M"):
                        recorder.toggle_mute()
                    elif ch == "\x03":
                        stop_event = True
            else:
                time.sleep(0.5)
    finally:
        if old_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)
        signal.signal(signal.SIGINT, original_handler)

    recorder.stop()
    if getattr(recorder, "silence_timed_out", False):
        click.echo("\n\nRecording auto-stopped after silence timeout.")
    else:
        click.echo("\n\nStopping recording...")

    if not audio_path.exists() or audio_path.stat().st_size <= _WAV_HEADER_SIZE:
        click.echo(
            "Error: No audio was captured. Make sure audio is playing on your system, "
            "or use --device to capture mic-only.",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Audio saved to {audio_path}\n")

    # Check for silent audio before spending time on transcription
    # Skip if the recorder already reported a silence warning (CoreAudio helper)
    if not getattr(recorder, "silence_warning", False):
        _check_audio_silence(audio_path)

    # 2. Transcribe
    _do_transcribe_and_summarize(config, audio_path, out_dir)


def run_transcribe(
    config: Config, audio_file: str, *, summarize: bool = False, identify: bool | None = None
) -> None:
    """Transcribe an audio file into a fresh ownscribe output directory."""
    source_path = Path(audio_file).resolve()
    _check_audio_silence(source_path)
    if _within_output_dir(config, source_path):
        out_dir = source_path.parent
        audio_path = source_path
    else:
        out_dir = _get_named_output_dir(config, source_path.stem)
        audio_path = out_dir / f"recording{source_path.suffix}"
        shutil.copy2(source_path, audio_path)
    _do_transcribe_and_summarize(
        config, audio_path, out_dir, summarize=summarize, identify=identify
    )


def run_warmup(config: Config) -> None:
    """Prefetch transcription/diarization models without processing audio."""
    diar_enabled = config.diarization.enabled and bool(config.diarization.hf_token)
    hf_token_warning = config.diarization.enabled and not config.diarization.hf_token
    local_sum = config.summarization.enabled and config.summarization.backend == "local"

    with PipelineProgress(
        diarize=False,
        summarize=False,
        transcribe=False,
        include_prepare=True,
        download_summarizer=local_sum,
    ) as progress:
        try:
            transcriber = _create_transcriber(config, progress=progress)
        except ImportError:
            click.echo(
                "Error: WhisperX is not installed. Install with:\n  uv pip install 'ownscribe[transcription]'",
                err=True,
            )
            raise SystemExit(1) from None

        transcriber.prepare_models(language=config.transcription.language or None)

        if local_sum:
            progress.begin("downloading_model")
            try:
                _download_summarization_model(config.summarization.model, progress, "downloading_model")
                progress.complete("downloading_model")
            except Exception as exc:
                progress.fail("downloading_model")
                click.echo(f"Error: {exc}", err=True)
                raise SystemExit(1) from None

    click.echo(f"Whisper model ready: {config.transcription.model}")
    if config.transcription.language:
        click.echo(f"Alignment model ready: {config.transcription.language}")
    else:
        click.echo("Alignment model not preloaded (language auto-detect).")

    if diar_enabled:
        click.echo("Diarization pipeline ready.")
    elif hf_token_warning:
        click.echo(
            "Warning: Diarization enabled but no HF token configured. Skipping diarization warmup.",
            err=True,
        )

    if local_sum:
        click.echo(f"Summarization model ready: {config.summarization.model}")


def run_summarize(config: Config, transcript_file: str, *, in_place: bool = False) -> None:
    """Summarize a transcript file into a fresh ownscribe output directory.

    When ``in_place`` is set, summarize within the transcript's own directory
    (used by ``resume`` to complete an existing meeting directory).
    """
    transcript_path = Path(transcript_file).resolve()
    transcript_text = transcript_path.read_text()

    try:
        summarizer = create_summarizer(config)
    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None
    if not summarizer.is_available():
        if config.summarization.backend == "local":
            click.echo(
                f"Error: Local summarization model '{config.summarization.model}' is not available.",
                err=True,
            )
        else:
            click.echo(
                f"Error: {config.summarization.backend} is not reachable at {config.summarization.host}. "
                "Is the server running?",
                err=True,
            )
        raise SystemExit(1)

    from ownscribe.output.markdown import format_summary

    if in_place or _within_output_dir(config, transcript_path):
        out_dir = transcript_path.parent
    else:
        out_dir = _get_named_output_dir(config, transcript_path.stem)
        shutil.copy2(transcript_path, out_dir / f"transcript{transcript_path.suffix}")
    local_sum = config.summarization.backend == "local"

    try:
        with PipelineProgress(
            transcribe=False,
            diarize=False,
            summarize=True,
            download_summarizer=local_sum,
        ) as progress:
            progress.begin("summarizing")
            if local_sum:
                progress.begin("downloading_model")
                try:
                    _download_summarization_model(
                        config.summarization.model,
                        progress,
                        "downloading_model",
                    )
                    progress.complete("downloading_model")
                except Exception:
                    progress.fail("downloading_model")
                    click.echo(
                        f"Error: Failed to download summarization model '{config.summarization.model}'.\n"
                        "Check your internet connection and try again.",
                        err=True,
                    )
                    raise SystemExit(1) from None
            summary = summarizer.summarize(transcript_text)
            title_slug = _generate_title_slug(summary, summarizer)
            progress.complete("summarizing")
    finally:
        summarizer.close()

    summary_md = format_summary(summary)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary_md)

    if title_slug:
        out_dir = _rename_with_title(out_dir, title_slug)

    summary_path = out_dir / "summary.md"

    click.echo(f"\n{summary_md}")
    click.echo(f"Summary saved to {summary_path}")


def _do_transcribe_and_summarize(
    config: Config,
    audio_path: Path,
    out_dir: Path,
    summarize: bool = True,
    identify: bool | None = None,
) -> None:
    """Shared logic for transcribe + optional summarize."""
    diar_enabled = config.diarization.enabled and bool(config.diarization.hf_token)
    sum_enabled = summarize and config.summarization.enabled

    summary = None
    summary_str = None
    title_slug = ""
    sum_unavailable = False
    sum_failed = False

    local_sum = sum_enabled and config.summarization.backend == "local"

    with PipelineProgress(
        diarize=diar_enabled,
        summarize=sum_enabled,
        download_summarizer=local_sum,
    ) as progress:
        try:
            transcriber = _create_transcriber(config, progress=progress)
        except ImportError:
            click.echo(
                "Error: WhisperX is not installed. Install with:\n  uv pip install 'ownscribe[transcription]'",
                err=True,
            )
            raise SystemExit(1) from None

        result = transcriber.transcribe(audio_path)

        if result.has_speakers:
            from ownscribe.voiceid.sidecar import DIARIZATION_FILENAME, write_sidecar

            write_sidecar(result, out_dir / DIARIZATION_FILENAME)

        _maybe_identify(config, result, audio_path, identify)

        # Save transcript — silent, no echo
        transcript_str, _ = _format_output(config, result)
        ext = "json" if config.output.format == "json" else "md"
        transcript_path = out_dir / f"transcript.{ext}"
        transcript_path.write_text(transcript_str)

        if sum_enabled:
            try:
                summarizer = create_summarizer(config)
            except ImportError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise SystemExit(1) from None
            try:
                if not summarizer.is_available():
                    sum_unavailable = True
                else:
                    try:
                        progress.begin("summarizing")
                        if local_sum:
                            progress.begin("downloading_model")
                            _download_summarization_model(
                                config.summarization.model,
                                progress,
                                "downloading_model",
                            )
                            progress.complete("downloading_model")
                        summary = summarizer.summarize(result.full_text)
                        _, summary_str = _format_output(config, result, summary)
                        summary_path = out_dir / f"summary.{ext}"
                        summary_path.write_text(summary_str or summary)
                        title_slug = _generate_title_slug(summary, summarizer)
                        progress.complete("summarizing")
                    except Exception:
                        progress.fail("summarizing")
                        sum_failed = True
            finally:
                summarizer.close()

    # --- All user-facing output after TUI exits ---
    click.echo(f"Transcript saved to {transcript_path}")

    if sum_unavailable:
        if config.summarization.backend == "local":
            click.echo(
                f"\nWarning: Local summarization model '{config.summarization.model}' is not available. "
                "Skipping summarization.",
                err=True,
            )
        else:
            click.echo(
                f"\nWarning: {config.summarization.backend} is not reachable "
                f"at {config.summarization.host}. "
                "Skipping summarization. Is the server running?",
                err=True,
            )
    elif sum_failed:
        click.echo(
            f"\nWarning: Summarization failed. "
            f"Transcript is saved at {transcript_path}\n"
            f"Resume with: ownscribe resume {out_dir}",
            err=True,
        )

    if summary is not None:
        click.echo(f"Summary saved to {out_dir / f'summary.{ext}'}")
        click.echo(f"\n{summary_str or summary}")
        if title_slug:
            out_dir = _rename_with_title(out_dir, title_slug)
    elif not summarize:
        click.echo(f"\n{transcript_str}")

    # Delete recording if configured — use the (possibly renamed) out_dir
    if not config.output.keep_recording:
        actual_audio_path = out_dir / audio_path.name
        if actual_audio_path.exists():
            actual_audio_path.unlink()
            click.echo(f"Recording deleted (keep_recording=false): {actual_audio_path}")


_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def _find_audio(directory: Path) -> Path | None:
    """Find an audio file in directory, preferring 'recording.wav'."""
    recording = directory / "recording.wav"
    if recording.exists():
        return recording
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS:
            return f
    return None


def _find_transcript(directory: Path) -> Path | None:
    """Find a transcript file in directory."""
    for ext in ("md", "json"):
        path = directory / f"transcript.{ext}"
        if path.exists():
            return path
    return None


def _find_summary(directory: Path) -> Path | None:
    """Find a summary file in directory."""
    for ext in ("md", "json"):
        path = directory / f"summary.{ext}"
        if path.exists():
            return path
    return None


def run_resume(config: Config, directory: str) -> None:
    """Resume a partially-completed pipeline in the given directory."""
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        click.echo(f"Error: {dir_path} is not a directory.", err=True)
        raise SystemExit(1)

    audio = _find_audio(dir_path)
    transcript = _find_transcript(dir_path)
    summary = _find_summary(dir_path)

    if transcript and summary:
        click.echo("Nothing to resume — transcript and summary already exist.")
        return

    if not audio and not transcript:
        click.echo(
            f"Error: No audio or transcript found in {dir_path}.\nA recording or transcript is needed to resume.",
            err=True,
        )
        raise SystemExit(1)

    if transcript:
        # Have transcript, missing summary — summarize only
        click.echo(f"Found transcript: {transcript}")
        click.echo("Resuming: summarize only.\n")
        run_summarize(config, str(transcript), in_place=True)
    else:
        # Have audio, missing transcript (and summary) — full transcribe + summarize
        click.echo(f"Found audio: {audio}")
        click.echo("Resuming: transcribe + summarize.\n")
        _do_transcribe_and_summarize(config, audio, dir_path)


_HEADER_RE = re.compile(r"^\*\*(.+?)\*\*\s*\[(\d{2}):(\d{2})(?::(\d{2}))?\]")
_SPEAKER_LABEL_RE = re.compile(r"^SPEAKER_\d+$")


def _transcript_name_suggestions(
    directory: Path, ranges: dict[str, list[tuple[float, float]]]
) -> dict[str, str]:
    transcript = directory / "transcript.md"
    if not transcript.exists():
        return {}
    suggestions: dict[str, str] = {}
    for line in transcript.read_text().splitlines():
        match = _HEADER_RE.match(line.strip())
        if not match:
            continue
        name = match.group(1).strip()
        if _SPEAKER_LABEL_RE.match(name):
            continue
        a, b = int(match.group(2)), int(match.group(3))
        c = int(match.group(4)) if match.group(4) else None
        timestamp = (a * 3600 + b * 60 + c) if c is not None else (a * 60 + b)
        for speaker, spk_ranges in ranges.items():
            if any(start <= timestamp <= end for start, end in spk_ranges):
                suggestions.setdefault(speaker, name)
                break
    return suggestions


def run_analyze(config: Config, directory: str) -> None:
    """Interactively enroll named voices from a diarized meeting directory."""
    from ownscribe.voiceid.playback import (
        extract_clip,
        play_clip,
        representative_range,
        total_duration,
    )
    from ownscribe.voiceid.sidecar import DIARIZATION_FILENAME, read_speaker_ranges

    dir_path = Path(directory).resolve()
    sidecar = dir_path / DIARIZATION_FILENAME
    if not sidecar.exists():
        click.echo(
            f"Error: no {DIARIZATION_FILENAME} in {dir_path}.\n"
            "Run a diarized transcription first (transcribe --diarize).",
            err=True,
        )
        raise SystemExit(1)

    audio = _find_audio(dir_path)
    if not audio:
        click.echo(f"Error: no audio file found in {dir_path}.", err=True)
        raise SystemExit(1)

    ranges_by_speaker = read_speaker_ranges(sidecar)
    suggestions = _transcript_name_suggestions(dir_path, ranges_by_speaker)

    try:
        embedder, store = _build_identify_tools(config)
        embedder.ensure_available()
    except ImportError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None

    for speaker, ranges in ranges_by_speaker.items():
        start, end = representative_range(ranges)
        if (end - start) < config.voice.min_clip_s:
            click.echo(f"Note: {speaker}'s longest clip is short (<{config.voice.min_clip_s}s).")
        try:
            clip = extract_clip(audio, start, end)
        except Exception:
            click.echo(f"Could not extract a clip for {speaker}; skipping.", err=True)
            continue

        default = suggestions.get(speaker, "")
        while True:
            played = play_clip(clip)
            hint = "" if played else " (playback unavailable)"
            answer = click.prompt(
                f"{speaker}{hint} — name, 'p' to replay, Enter to skip",
                default=default,
                show_default=bool(default),
            ).strip()
            if answer.lower() == "p":
                continue
            name = answer
            break

        clip.unlink(missing_ok=True)
        if not name:
            click.echo(f"Skipped {speaker}.")
            continue

        embedding = embedder.embed(audio, ranges)
        store.enroll(
            name,
            embedding,
            source_dir=dir_path.name,
            duration_s=total_duration(ranges),
        )
        click.echo(f"Enrolled {speaker} as '{name}'.")
