"""Tests for scripts/week10_lagrangian_violation_table.py."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "week10_lagrangian_violation_table.py"


def _write_jsonl(path: Path, *, n_steps: int, latency_ms: float,
                 memory_util: float, energy_j: float = 1e-3) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for st in range(n_steps):
            rec = {
                "episode": 0, "step": st, "action": 0,
                "reward": 1.0, "latency_ms": float(latency_ms),
                "energy_j": float(energy_j), "memory_util": float(memory_util),
                "omega": [0.25, 0.25, 0.25, 0.25],
            }
            f.write(json.dumps(rec) + "\n")


def _build_two_variants(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    runs.mkdir()
    yml_entries = []
    # 3 seeds for each variant.
    for sd in range(3):
        # No-override variant: high latency + high memory.
        f_no = runs / f"override_layer__preference_ppo__seed{sd}.jsonl"
        _write_jsonl(f_no, n_steps=100, latency_ms=5.0, memory_util=0.8)
        yml_entries.append({
            "env_name": "dag_scheduler_mo-v0", "agent_type": "preference_ppo",
            "ablation": "override_layer", "platform": "mac_stub",
            "n_episodes": 1, "seed": sd, "out_dir": str(runs),
            "extra": {}, "n_envs": 1,
        })
        # Override-on variant: low latency + low memory.
        f_yes = runs / f"none__preference_ppo__seed{sd}.jsonl"
        _write_jsonl(f_yes, n_steps=100, latency_ms=0.5, memory_util=0.1)
        yml_entries.append({
            "env_name": "dag_scheduler_mo-v0", "agent_type": "preference_ppo",
            "ablation": "none", "platform": "mac_stub",
            "n_episodes": 1, "seed": sd, "out_dir": str(runs),
            "extra": {}, "n_envs": 1,
        })
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"configs": yml_entries}, sort_keys=False))
    return runs, matrix


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True,
    )


def test_help_lists_required_flags():
    res = _run(["--help"])
    assert res.returncode == 0
    for flag in ("--matrix-yaml", "--runs-dir", "--out-dir",
                 "--latency-threshold-ms", "--memory-threshold",
                 "--energy-budget-j"):
        assert flag in res.stdout


def test_emits_md_and_csv(tmp_path: Path):
    runs, matrix = _build_two_variants(tmp_path)
    res = _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--latency-threshold-ms", "2.0",
        "--memory-threshold", "0.5",
        "--energy-budget-j", "1000.0",
    ])
    assert res.returncode == 0, res.stderr
    assert (runs / "lagrangian_violation_table.md").exists()
    assert (runs / "lagrangian_violation_table.csv").exists()


def test_md_contains_two_rows(tmp_path: Path):
    runs, matrix = _build_two_variants(tmp_path)
    _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--latency-threshold-ms", "2.0",
        "--memory-threshold", "0.5",
        "--energy-budget-j", "1000.0",
    ])
    md = (runs / "lagrangian_violation_table.md").read_text()
    assert "Lagrangian only" in md or "no override" in md.lower()
    assert "Lagrangian + override" in md or "override on" in md.lower()


def test_no_override_variant_has_higher_violation_rate(tmp_path: Path):
    runs, matrix = _build_two_variants(tmp_path)
    _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--latency-threshold-ms", "2.0",
        "--memory-threshold", "0.5",
        "--energy-budget-j", "1000.0",
    ])
    rows = list(csv.DictReader((runs / "lagrangian_violation_table.csv").open()))
    by_var = {r["variant"]: float(r["violation_rate"]) for r in rows}
    # The no-override variant should violate at much higher rate
    # (synthetic data: latency=5 > 2 AND memory=0.8 > 0.5 every step).
    assert by_var.get("override_off", -1) > by_var.get("override_on", -1)
    # And the override-on variant should have ~0 violation rate.
    assert by_var.get("override_on", 1.0) < 0.05


def test_csv_columns(tmp_path: Path):
    runs, matrix = _build_two_variants(tmp_path)
    _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--latency-threshold-ms", "2.0",
        "--memory-threshold", "0.5",
        "--energy-budget-j", "1000.0",
    ])
    rows = list(csv.DictReader((runs / "lagrangian_violation_table.csv").open()))
    assert len(rows) == 2
    cols = set(rows[0].keys())
    for col in ("variant", "violation_rate", "std", "n_runs"):
        assert col in cols
