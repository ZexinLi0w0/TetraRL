"""Week 7 Task 8a: tests for the FFmpeg co-runner CDF plotter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.week7_make_cdf import main


def _write_jsonl(path: Path, latencies: list[float]) -> None:
    """Write a JSONL file with ``{"step": i, "latency_ms": v}`` per sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, v in enumerate(latencies):
            f.write(json.dumps({"step": i, "latency_ms": float(v)}) + "\n")


def test_cdf_runs_on_synthetic_jsonl(tmp_path: Path):
    """Two well-formed JSONL inputs produce a non-empty PNG and exit 0."""
    _write_jsonl(tmp_path / "none.jsonl", [float(i) for i in range(100)])
    _write_jsonl(tmp_path / "720p.jsonl", [float(i) for i in range(100)])

    out_png = tmp_path / "cdf.png"
    rc = main(
        [
            "--in-dir",
            str(tmp_path),
            "--conditions",
            "none,720p",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_cdf_skips_missing_condition(tmp_path: Path):
    """Missing condition JSONLs are warned about but do not crash the run."""
    _write_jsonl(tmp_path / "none.jsonl", [float(i) for i in range(50)])

    out_png = tmp_path / "cdf.png"
    rc = main(
        [
            "--in-dir",
            str(tmp_path),
            "--conditions",
            "none,720p,1080p",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_cdf_exits_1_when_no_data(tmp_path: Path):
    """An empty input directory must exit 1 with no plot written."""
    out_png = tmp_path / "cdf.png"
    rc = main(
        [
            "--in-dir",
            str(tmp_path),
            "--conditions",
            "none",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 1
    # No plot should have been written when no data was available.
    assert not out_png.exists()


def test_cdf_p99_table_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Stdout must include a Markdown table with a ~99 ms p99 for [1..100]."""
    latencies = [float(i) for i in range(1, 101)]  # 1..100 ms
    _write_jsonl(tmp_path / "none.jsonl", latencies)

    rc = main(
        [
            "--in-dir",
            str(tmp_path),
            "--conditions",
            "none",
            "--out-png",
            str(tmp_path / "cdf.png"),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    out = captured.out

    assert "p99" in out, f"expected 'p99' header in stdout, got:\n{out}"
    assert "p50" in out and "p95" in out

    # Locate the data row for the 'none' condition and pull the p99 cell.
    none_rows = [
        line for line in out.splitlines() if line.startswith("| none ")
    ]
    assert none_rows, f"expected a markdown row for 'none', got:\n{out}"
    cells = [c.strip() for c in none_rows[0].strip("|").split("|")]
    # Columns: condition | n | p50_ms | p95_ms | p99_ms
    assert len(cells) == 5
    p99_value = float(cells[4])
    # numpy's linear-interpolation p99 of [1..100] is 99.01.
    assert p99_value == pytest.approx(99.0, abs=0.5)


def test_cdf_accepts_recorder_native_schema(tmp_path: Path):
    """LatencyRecorder writes ``sample_ms`` keys; the plotter must accept them."""
    path = tmp_path / "none.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"sample_ms": float(i + 1), "idx": i}) + "\n")

    out_png = tmp_path / "cdf.png"
    rc = main(
        [
            "--in-dir",
            str(tmp_path),
            "--conditions",
            "none",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_cdf_missing_in_dir_exits_1_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A nonexistent --in-dir must exit 1 with a friendly stderr message."""
    missing = tmp_path / "definitely_not_here"
    rc = main(
        [
            "--in-dir",
            str(missing),
            "--conditions",
            "none",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
