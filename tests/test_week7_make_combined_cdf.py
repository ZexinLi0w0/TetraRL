"""Week 7 cleanup Task D: tests for the 2-panel (Orin + Nano) combined CDF plotter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.week7_make_combined_cdf import main


def _write_jsonl(path: Path, latencies: list[float]) -> None:
    """Write a JSONL file with ``{"step": i, "latency_ms": v}`` per sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, v in enumerate(latencies):
            f.write(json.dumps({"step": i, "latency_ms": float(v)}) + "\n")


def _write_jsonl_recorder_schema(path: Path, latencies: list[float]) -> None:
    """Write a JSONL file with ``{"sample_ms": v, "idx": i}`` per sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, v in enumerate(latencies):
            f.write(json.dumps({"sample_ms": float(v), "idx": i}) + "\n")


def test_combined_cdf_writes_two_panel_png(tmp_path: Path):
    """Two valid in-dirs each with 2 conditions produce a non-empty 2-panel PNG."""
    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _write_jsonl(orin_dir / "none.jsonl", [float(i) for i in range(50)])
    _write_jsonl(orin_dir / "720p.jsonl", [float(i) * 1.5 for i in range(50)])
    _write_jsonl(nano_dir / "none.jsonl", [float(i) * 2.0 for i in range(50)])
    _write_jsonl(nano_dir / "720p.jsonl", [float(i) * 3.0 for i in range(50)])

    out_png = tmp_path / "combined.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none,720p",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0, "expected exit 0 on a successful run"
    assert out_png.exists(), "expected the output PNG to be written"
    assert out_png.stat().st_size > 0, "expected a non-empty PNG"


def test_combined_cdf_writes_svg_when_requested(tmp_path: Path):
    """The --out-svg flag writes an additional SVG copy alongside the PNG."""
    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _write_jsonl(orin_dir / "none.jsonl", [float(i) for i in range(20)])
    _write_jsonl(nano_dir / "none.jsonl", [float(i) * 2.0 for i in range(20)])

    out_png = tmp_path / "combined.png"
    out_svg = tmp_path / "combined.svg"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none",
            "--out-png",
            str(out_png),
            "--out-svg",
            str(out_svg),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_svg.exists() and out_svg.stat().st_size > 0


def test_combined_cdf_skips_missing_condition_per_panel(tmp_path: Path):
    """A condition missing on one side must not abort the other panel."""
    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    # Orin has 2K, Nano does not — both should still render.
    _write_jsonl(orin_dir / "none.jsonl", [float(i) for i in range(30)])
    _write_jsonl(orin_dir / "720p.jsonl", [float(i) for i in range(30)])
    _write_jsonl(orin_dir / "2K.jsonl", [float(i) for i in range(30)])
    _write_jsonl(nano_dir / "none.jsonl", [float(i) for i in range(30)])
    _write_jsonl(nano_dir / "720p.jsonl", [float(i) for i in range(30)])

    out_png = tmp_path / "combined.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none,720p,2K",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0


def test_combined_cdf_accepts_recorder_native_schema(tmp_path: Path):
    """LatencyRecorder writes ``sample_ms``; the combined plotter must accept it."""
    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _write_jsonl_recorder_schema(orin_dir / "none.jsonl", [float(i + 1) for i in range(20)])
    _write_jsonl_recorder_schema(nano_dir / "none.jsonl", [float(i + 1) * 2 for i in range(20)])

    out_png = tmp_path / "combined.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0


def test_combined_cdf_missing_orin_dir_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A nonexistent --orin-dir must exit 1 with a friendly stderr message."""
    nano_dir = tmp_path / "nano"
    _write_jsonl(nano_dir / "none.jsonl", [float(i) for i in range(10)])
    missing = tmp_path / "definitely_not_here_orin"
    rc = main(
        [
            "--orin-dir",
            str(missing),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none",
            "--out-png",
            str(tmp_path / "combined.png"),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_combined_cdf_missing_nano_dir_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A nonexistent --nano-dir must exit 1 with a friendly stderr message."""
    orin_dir = tmp_path / "orin"
    _write_jsonl(orin_dir / "none.jsonl", [float(i) for i in range(10)])
    missing = tmp_path / "definitely_not_here_nano"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(missing),
            "--conditions",
            "none",
            "--out-png",
            str(tmp_path / "combined.png"),
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_combined_cdf_exits_1_when_no_data_anywhere(tmp_path: Path):
    """Both in-dirs empty -> exit 1, no plot."""
    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    orin_dir.mkdir()
    nano_dir.mkdir()
    out_png = tmp_path / "combined.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 1
    assert not out_png.exists()


def test_combined_cdf_one_panel_empty_other_renders(tmp_path: Path):
    """If one panel has data and the other doesn't, exit 0 and still render the figure."""
    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _write_jsonl(orin_dir / "none.jsonl", [float(i) for i in range(40)])
    nano_dir.mkdir()  # no JSONL files inside

    out_png = tmp_path / "combined.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none",
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0, "should still succeed when one side has at least one usable curve"
    assert out_png.exists() and out_png.stat().st_size > 0


def test_combined_cdf_summary_table_includes_both_devices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Stdout must include a Markdown table tagged with both 'orin' and 'nano' rows."""
    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _write_jsonl(orin_dir / "none.jsonl", [float(i) for i in range(1, 101)])  # p99 ~= 99
    _write_jsonl(nano_dir / "none.jsonl", [float(i) * 2 for i in range(1, 101)])  # p99 ~= 198

    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--conditions",
            "none",
            "--out-png",
            str(tmp_path / "combined.png"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "p99" in out
    # Both device labels must appear in the summary table.
    assert "orin" in out.lower()
    assert "nano" in out.lower()
