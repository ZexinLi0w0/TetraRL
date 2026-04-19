"""Week 7 Task 5: FFmpeg co-runner interference harness tests."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# --------------------------- Cycle 1: LatencyRecorder ----------------------


def test_latency_recorder_percentiles():
    """percentiles() returns hand-computed values for a known sample."""
    from tetrarl.eval.ffmpeg_interference import LatencyRecorder

    rec = LatencyRecorder()
    # Inject a known monotonic sample: 1, 2, 3, ..., 100 ms.
    # We bypass mark() since timing is non-deterministic.
    rec._samples_ms = [float(i) for i in range(1, 101)]

    pcts = rec.percentiles([50.0, 90.0, 99.0, 99.9])
    # Linear interpolation between order statistics on sorted [1..100].
    # For p=50 over n=100: rank index r = (p/100)*(n-1) = 49.5 -> between
    # samples[49]=50.0 and samples[50]=51.0 -> 50.5.
    assert pcts[50.0] == pytest.approx(50.5, abs=1e-9)
    # p90: r = 0.90 * 99 = 89.1 -> between samples[89]=90.0 and [90]=91.0
    # -> 90.0 + 0.1*1.0 = 90.1
    assert pcts[90.0] == pytest.approx(90.1, abs=1e-9)
    # p99: r = 0.99 * 99 = 98.01 -> between [98]=99.0 and [99]=100.0
    # -> 99.01
    assert pcts[99.0] == pytest.approx(99.01, abs=1e-9)
    # p99.9: r = 0.999 * 99 = 98.901 -> 99.0 + 0.901*1.0 = 99.901
    assert pcts[99.9] == pytest.approx(99.901, abs=1e-9)


def test_latency_recorder_to_jsonl_roundtrip(tmp_path: Path):
    """to_jsonl writes one JSON object per sample with sample_ms+idx fields."""
    from tetrarl.eval.ffmpeg_interference import LatencyRecorder

    rec = LatencyRecorder()
    rec._samples_ms = [0.5, 1.25, 7.0]
    out = tmp_path / "samples.jsonl"
    rec.to_jsonl(str(out))

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["sample_ms"] for p in parsed] == [0.5, 1.25, 7.0]
    assert [p["idx"] for p in parsed] == [0, 1, 2]


def test_latency_recorder_uses_perf_counter():
    """mark() must produce strictly non-negative samples and use perf_counter_ns."""
    from tetrarl.eval.ffmpeg_interference import LatencyRecorder

    rec = LatencyRecorder()
    rec.start()
    for _ in range(5):
        # tiny busy-loop to force a measurable delta
        end = time.perf_counter_ns() + 50_000  # 50 microseconds
        while time.perf_counter_ns() < end:
            pass
        rec.mark()
    samples = rec.samples_ms
    assert len(samples) == 5
    # Every sample is non-negative; perf_counter_ns is monotonic.
    for s in samples:
        assert s >= 0.0
    # At least one sample should be positive (we spun for 50us).
    assert any(s > 0.0 for s in samples)


# ----------------------- Cycle 2: FFmpegInterference -----------------------


def test_ffmpeg_available_returns_bool():
    from tetrarl.eval.ffmpeg_interference import ffmpeg_available

    val = ffmpeg_available()
    assert isinstance(val, bool)


def test_resolution_to_wh_mapping():
    from tetrarl.eval.ffmpeg_interference import FFmpegInterference

    assert FFmpegInterference._resolution_to_wh("720p") == (1280, 720)
    assert FFmpegInterference._resolution_to_wh("1080p") == (1920, 1080)
    assert FFmpegInterference._resolution_to_wh("2K") == (2560, 1440)


def test_none_condition_is_noop():
    """resolution='none' must not spawn a subprocess."""
    from tetrarl.eval.ffmpeg_interference import FFmpegInterference

    with FFmpegInterference(resolution="none") as ff:
        assert ff.process is None


def test_synthetic_testsrc_argv_when_no_video_path():
    """If video_path is None we must use ``-f lavfi`` testsrc source."""
    from tetrarl.eval.ffmpeg_interference import FFmpegInterference

    argv = FFmpegInterference.build_argv(
        resolution="720p", video_path=None, hw_decode=False
    )
    # The argv list should contain a -f flag followed by 'lavfi' and a -i
    # flag whose value starts with 'testsrc='.
    assert "-f" in argv
    f_idx = argv.index("-f")
    assert argv[f_idx + 1] == "lavfi"
    assert "-i" in argv
    i_idx = argv.index("-i")
    assert argv[i_idx + 1].startswith("testsrc=")
    # Resolution propagated as 1280x720, framerate 30.
    assert "size=1280x720" in argv[i_idx + 1]
    assert "rate=30" in argv[i_idx + 1]
    # Output discarded via -f null -.
    assert argv[-2:] == ["-f", "null"] or argv[-2:] == ["null", "-"] or argv[-3:] == ["-f", "null", "-"]


def test_video_path_argv_uses_stream_loop():
    """If video_path is given, argv must include ``-stream_loop -1`` + path."""
    from tetrarl.eval.ffmpeg_interference import FFmpegInterference

    argv = FFmpegInterference.build_argv(
        resolution="1080p", video_path="/tmp/foo.h264", hw_decode=False
    )
    assert "-stream_loop" in argv
    sl_idx = argv.index("-stream_loop")
    assert argv[sl_idx + 1] == "-1"
    assert "/tmp/foo.h264" in argv


def test_hw_decode_suppressed_when_unavailable(monkeypatch):
    """hw_decode=True with no nvv4l2dec available must NOT add -c:v flag."""
    from tetrarl.eval import ffmpeg_interference as ffi

    monkeypatch.setattr(ffi.FFmpegInterference, "_hw_decode_available", staticmethod(lambda: False))
    argv = ffi.FFmpegInterference.build_argv(
        resolution="1080p", video_path="/tmp/foo.h264", hw_decode=True
    )
    assert "h264_nvv4l2dec" not in argv


def test_hw_decode_added_when_available(monkeypatch):
    """hw_decode=True with nvv4l2dec available must add ``-c:v h264_nvv4l2dec`` before -i."""
    from tetrarl.eval import ffmpeg_interference as ffi

    monkeypatch.setattr(ffi.FFmpegInterference, "_hw_decode_available", staticmethod(lambda: True))
    argv = ffi.FFmpegInterference.build_argv(
        resolution="1080p", video_path="/tmp/foo.h264", hw_decode=True
    )
    assert "h264_nvv4l2dec" in argv
    cv_idx = argv.index("-c:v")
    i_idx = argv.index("-i")
    assert cv_idx < i_idx
    assert argv[cv_idx + 1] == "h264_nvv4l2dec"


def test_context_manager_spawns_and_reaps():
    """End-to-end: spawn ffmpeg with testsrc, verify it is reaped on exit."""
    from tetrarl.eval.ffmpeg_interference import FFmpegInterference, ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("ffmpeg not on PATH")

    with FFmpegInterference(resolution="720p") as ff:
        assert ff.process is not None
        # Subprocess should be alive immediately after __enter__.
        assert ff.process.poll() is None
        proc = ff.process
        # Hold the reference so we can poll it after exit.

    # On exit, the subprocess must be reaped (terminated and waited on).
    # poll() returning a non-None code means the process finished.
    assert proc.poll() is not None


# ----------------------------- Cycle 3: summarize -------------------------


def _seeded_recorder(values_ms: list[float]):
    from tetrarl.eval.ffmpeg_interference import LatencyRecorder

    rec = LatencyRecorder()
    rec._samples_ms = list(values_ms)
    return rec


def test_summarize_columns_present():
    """Markdown header lists all required columns and one row per condition."""
    from tetrarl.eval.ffmpeg_interference import summarize

    results = {
        "none": _seeded_recorder([1.0, 2.0, 3.0, 4.0]),
        "720p": _seeded_recorder([2.0, 4.0, 6.0, 8.0]),
    }
    md = summarize(results)
    # Header columns
    for col in ("condition", "n", "p50_ms", "p90_ms", "p99_ms", "p99.9_ms", "slowdown_p99"):
        assert col in md, f"missing column header {col!r}"
    # One row per condition (look for the leading pipe + condition name)
    assert "| none" in md
    assert "| 720p" in md


def test_summarize_slowdown_relative_to_baseline():
    """slowdown_p99 = condition_p99 / baseline_p99 (baseline = 'none')."""
    from tetrarl.eval.ffmpeg_interference import summarize

    # Baseline samples: p99 of [1..100] is 99.01 (type-7 interpolation).
    baseline_vals = [float(i) for i in range(1, 101)]
    # Condition samples: 2x baseline, so p99 = 198.02.
    condition_vals = [2.0 * v for v in baseline_vals]
    results = {
        "none": _seeded_recorder(baseline_vals),
        "720p": _seeded_recorder(condition_vals),
    }
    md = summarize(results)
    # Baseline row should show 1.00x slowdown (numerically 1.0).
    none_row = next(line for line in md.splitlines() if line.startswith("| none"))
    assert "1.00x" in none_row
    # 720p row should show 2.00x slowdown.
    cond_row = next(line for line in md.splitlines() if line.startswith("| 720p"))
    assert "2.00x" in cond_row


def test_summarize_no_baseline_marks_slowdown_na():
    """If the 'none' baseline is absent, slowdown_p99 column should read N/A."""
    from tetrarl.eval.ffmpeg_interference import summarize

    results = {"720p": _seeded_recorder([1.0, 2.0, 3.0])}
    md = summarize(results)
    cond_row = next(line for line in md.splitlines() if line.startswith("| 720p"))
    assert "N/A" in cond_row
