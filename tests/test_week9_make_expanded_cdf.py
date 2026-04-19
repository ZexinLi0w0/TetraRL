"""Week 9 cross-platform expanded CDF: tests for the per-ω 12-curve plotter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Default omega vectors and conditions used to build complete grids in tests.
_DEFAULT_OMEGAS = ("energy_corner", "memory_corner", "center")
_DEFAULT_CONDITIONS = ("none", "720p", "1080p", "2K")


def _write_jsonl(path: Path, latencies: list[float]) -> None:
    """Write a JSONL file with ``{"sample_ms": v, "idx": i}`` per sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, v in enumerate(latencies):
            f.write(json.dumps({"sample_ms": float(v), "idx": i}) + "\n")


def _build_full_grid(
    root: Path,
    omegas: tuple[str, ...] = _DEFAULT_OMEGAS,
    conditions: tuple[str, ...] = _DEFAULT_CONDITIONS,
    base: float = 1.0,
) -> None:
    """Populate ``root/<omega>/<condition>.jsonl`` with 100 synthetic samples each.

    The mean latency is varied per (omega, condition) so the resulting
    empirical CDF curves are visibly distinct in a smoke render.
    """
    root.mkdir(parents=True, exist_ok=True)
    for oi, omega in enumerate(omegas):
        for ci, cond in enumerate(conditions):
            mean = base + 0.5 * oi + 0.25 * ci
            latencies = [mean + 0.01 * i for i in range(100)]
            _write_jsonl(root / omega / f"{cond}.jsonl", latencies)


def test_imports_main_entrypoint() -> None:
    """The script must expose a ``main`` callable as a stable importable entrypoint."""
    from scripts.week9_make_expanded_cdf import main  # noqa: F401

    assert callable(main)


def test_full_grid_writes_png_and_summary_md(tmp_path: Path) -> None:
    """Complete 3×4×2 grid renders a PNG and a 24-row summary.md."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _build_full_grid(orin_dir, base=1.0)
    _build_full_grid(nano_dir, base=2.0)

    out_png = tmp_path / "expanded.png"

    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0, "full grid must exit 0"
    assert out_png.exists() and out_png.stat().st_size > 0

    summary_md = out_png.with_suffix(out_png.suffix + ".summary.md")
    assert summary_md.exists(), "expected sidecar summary.md to be written"
    md_text = summary_md.read_text(encoding="utf-8")
    assert "| platform | omega | condition | n | p50_ms | p95_ms | p99_ms |" in md_text

    # 3 omegas * 4 conditions * 2 platforms = 24 data rows.
    data_rows = [
        line
        for line in md_text.splitlines()
        if line.startswith("|")
        and not line.startswith("| platform")
        and not line.startswith("|----")
    ]
    assert len(data_rows) == 24, f"expected 24 data rows, got {len(data_rows)}"


def test_writes_svg_when_requested(tmp_path: Path) -> None:
    """The --out-svg flag emits an SVG copy alongside the PNG."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _build_full_grid(orin_dir, base=1.0)
    _build_full_grid(nano_dir, base=2.0)

    out_png = tmp_path / "expanded.png"
    out_svg = tmp_path / "expanded.svg"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(out_png),
            "--out-svg",
            str(out_svg),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_svg.exists() and out_svg.stat().st_size > 0


def test_missing_omega_dir_skipped_with_warning_per_panel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing ω dir on one side warns and skips the whole ω for that panel."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _build_full_grid(orin_dir, base=1.0)
    # Build all but one omega on the Nano side.
    _build_full_grid(
        nano_dir, omegas=("energy_corner", "center"), base=2.0
    )

    out_png = tmp_path / "expanded.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0
    err = capsys.readouterr().err
    assert "missing omega dir for nano/memory_corner" in err


def test_missing_single_jsonl_skipped_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing single (ω, condition) JSONL warns but does not abort."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _build_full_grid(orin_dir, base=1.0)
    _build_full_grid(nano_dir, base=2.0)

    # Remove a single JSONL on the Orin side.
    missing = orin_dir / "energy_corner" / "1080p.jsonl"
    missing.unlink()

    out_png = tmp_path / "expanded.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0
    err = capsys.readouterr().err
    assert "missing JSONL for orin/energy_corner/1080p" in err


def test_orin_dir_missing_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A nonexistent --orin-dir hard-fails with exit 1 and a stderr message."""
    from scripts.week9_make_expanded_cdf import main

    nano_dir = tmp_path / "nano"
    _build_full_grid(nano_dir, base=2.0)
    missing = tmp_path / "definitely_not_here_orin"

    rc = main(
        [
            "--orin-dir",
            str(missing),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(tmp_path / "expanded.png"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert str(missing) in err


def test_nano_dir_missing_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A nonexistent --nano-dir hard-fails with exit 1 and a stderr message."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    _build_full_grid(orin_dir, base=1.0)
    missing = tmp_path / "definitely_not_here_nano"

    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(missing),
            "--out-png",
            str(tmp_path / "expanded.png"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert str(missing) in err


def test_no_data_anywhere_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both root dirs exist but contain no usable JSONLs -> exit 1."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    orin_dir.mkdir()
    nano_dir.mkdir()

    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(tmp_path / "expanded.png"),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "no (omega, condition) pairs" in err


def test_one_panel_empty_other_renders(tmp_path: Path) -> None:
    """If the Nano panel is empty but the Orin panel has data, still render."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _build_full_grid(orin_dir, base=1.0)
    nano_dir.mkdir()  # exists but no per-omega subdirs

    out_png = tmp_path / "expanded.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0


def test_summary_table_includes_both_devices_and_omegas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Markdown table must mention both devices and at least 2 distinct omegas."""
    from scripts.week9_make_expanded_cdf import main

    orin_dir = tmp_path / "orin"
    nano_dir = tmp_path / "nano"
    _build_full_grid(orin_dir, base=1.0)
    _build_full_grid(nano_dir, base=2.0)

    out_png = tmp_path / "expanded.png"
    rc = main(
        [
            "--orin-dir",
            str(orin_dir),
            "--nano-dir",
            str(nano_dir),
            "--out-png",
            str(out_png),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "orin" in out.lower()
    assert "nano" in out.lower()
    distinct_omegas = sum(1 for omega in _DEFAULT_OMEGAS if omega in out)
    assert distinct_omegas >= 2, (
        f"expected >=2 distinct omega names in stdout, found {distinct_omegas}"
    )
