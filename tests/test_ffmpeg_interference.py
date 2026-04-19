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
