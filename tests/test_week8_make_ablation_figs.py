"""Week 8 ablation figure script: CLI-arg adaptation tests.

The script was originally hard-coded to read ``runs/w8_ablation_orin/summary.csv``.
The W8 real-Orin re-run requires the same script to point at a different
runs directory and emit its figures there, so we expose ``main(argv)`` and
add ``--runs-dir`` plus ``--footer``.
"""
from __future__ import annotations

import csv
from pathlib import Path

from scripts.week8_make_ablation_figs import main


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _three_seed_rows(arm: str, n_steps: int, override: int) -> list[dict[str, str]]:
    return [
        {
            "env_name": "CartPole-v1",
            "agent_type": "preference_ppo",
            "ablation": arm,
            "platform": "orin_agx",
            "seed": str(s),
            "n_episodes": "200",
            "n_steps": str(n_steps + s),
            "mean_reward": "1.0",
            "override_fire_count": str(override + s),
            "tail_p99_ms": "0.06",
            "mean_energy_j": "0.001",
            "mean_memory_util": "0.108",
            "wall_time_s": "0.24",
        }
        for s in range(3)
    ]


def _five_arm_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows += _three_seed_rows("none", 3000, 50)
    rows += _three_seed_rows("preference_plane", 4400, 290)
    rows += _three_seed_rows("resource_manager", 3000, 50)
    rows += _three_seed_rows("rl_arbiter", 4300, 290)
    rows += _three_seed_rows("override_layer", 3100, 0)
    return rows


def test_runs_dir_flag_writes_to_new_dir(tmp_path: Path):
    """``--runs-dir X`` reads ``X/summary.csv`` and writes both PNGs into ``X``."""
    runs_dir = tmp_path / "w8_ablation_orin_real"
    _write_summary(runs_dir / "summary.csv", _five_arm_rows())

    rc = main(["--runs-dir", str(runs_dir)])

    assert rc == 0
    bar = runs_dir / "violation_rate_bar.png"
    hv = runs_dir / "hv_comparison.png"
    assert bar.exists() and bar.stat().st_size > 0
    assert hv.exists() and hv.stat().st_size > 0


def test_footer_flag_overrides_default_text(tmp_path: Path):
    """``--footer "..."`` overrides the default Mac-substitute footer.

    Smoke check via successful exit + figure presence; deeper image-content
    inspection is out of scope for unit tests.
    """
    runs_dir = tmp_path / "w8_ablation_orin_real"
    _write_summary(runs_dir / "summary.csv", _five_arm_rows())

    rc = main(
        [
            "--runs-dir",
            str(runs_dir),
            "--footer",
            "Real Orin AGX run, week8 re-run",
        ]
    )

    assert rc == 0
    assert (runs_dir / "violation_rate_bar.png").stat().st_size > 0


def test_default_args_still_use_legacy_dir(tmp_path: Path, monkeypatch):
    """No-args invocation must still target ``runs/w8_ablation_orin/`` (back-compat).

    We patch the script's repo_root to ``tmp_path`` and seed a summary.csv
    at the legacy path; main() with empty argv should write figures there.
    """
    import scripts.week8_make_ablation_figs as mod

    legacy = tmp_path / "runs" / "w8_ablation_orin"
    _write_summary(legacy / "summary.csv", _five_arm_rows())

    monkeypatch.setattr(mod, "_REPO_ROOT", tmp_path)

    rc = main([])

    assert rc == 0
    assert (legacy / "violation_rate_bar.png").stat().st_size > 0
