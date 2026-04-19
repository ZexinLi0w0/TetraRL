"""Tests for scripts/week10_make_hv_comparison.py — HV bar chart + Welch."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "week10_make_hv_comparison.py"


def _write_jsonl(path: Path, *, n_episodes: int = 3, n_steps: int = 5,
                 reward: float, latency: float = 5.0, memory: float = 0.2,
                 energy: float = 1e-3, omega=(0.25, 0.25, 0.25, 0.25)) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            for st in range(n_steps):
                rec = {
                    "episode": ep, "step": st, "action": 0,
                    "reward": float(reward), "latency_ms": float(latency),
                    "energy_j": float(energy), "memory_util": float(memory),
                    "omega": list(map(float, omega)),
                }
                f.write(json.dumps(rec) + "\n")


def _build_matrix(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny synthetic 'runs/' dir + matrix YAML with 2 agents
    × 1 env × 2 omegas × 2 seeds = 8 entries; preference_ppo dominates."""
    runs = tmp_path / "runs_w10"
    runs.mkdir()
    agents = ["preference_ppo", "max_p"]
    omegas = [(1.0, 0.0, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25)]
    seeds = [0, 1]
    yml_entries = []
    for ag in agents:
        for o_idx, om in enumerate(omegas):
            for sd in seeds:
                # File naming convention: ablation__agent__seedN.jsonl
                # plus an omega-index suffix to disambiguate per-omega
                # runs (the runner does NOT include omega in the name;
                # the matrix yaml does — the analysis script must
                # reconcile via the manifest).
                fname = f"none__{ag}__seed{sd}__o{o_idx}.jsonl"
                p = runs / fname
                # preference_ppo gets a higher reward proxy than max_p.
                rew = 5.0 if ag == "preference_ppo" else 1.0
                _write_jsonl(p, reward=rew, omega=om)
                yml_entries.append({
                    "env_name": "dag_scheduler_mo-v0",
                    "agent_type": ag,
                    "ablation": "none",
                    "platform": "mac_stub",
                    "n_episodes": 3,
                    "seed": sd,
                    "out_dir": str(runs),
                    "extra": {"omega": list(om), "omega_idx": o_idx,
                              "jsonl_name": fname},
                    "n_envs": 1,
                })
    yml = tmp_path / "matrix.yaml"
    yml.write_text(yaml.safe_dump({"configs": yml_entries}, sort_keys=False))
    return runs, yml


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True,
    )


def test_help_lists_required_flags():
    res = _run(["--help"])
    assert res.returncode == 0
    for flag in ("--matrix-yaml", "--runs-dir", "--out-dir",
                 "--ref-point", "--tetrarl-agent"):
        assert flag in res.stdout


def test_emits_png_svg_md_csv(tmp_path: Path):
    runs, yml = _build_matrix(tmp_path)
    res = _run([
        "--matrix-yaml", str(yml),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--ref-point", "0,-100,-1,-1",
    ])
    assert res.returncode == 0, res.stderr
    for fname in ("hv_comparison.png", "hv_comparison.svg",
                  "hv_comparison.md", "hv_comparison.csv"):
        assert (runs / fname).exists(), fname


def test_csv_has_one_row_per_run(tmp_path: Path):
    runs, yml = _build_matrix(tmp_path)
    _run([
        "--matrix-yaml", str(yml),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--ref-point", "0,-100,-1,-1",
    ])
    rows = list(csv.DictReader((runs / "hv_comparison.csv").open()))
    # 2 agents × 2 omegas × 2 seeds = 8 entries.
    assert len(rows) == 8
    cols = set(rows[0].keys())
    for col in ("agent", "env", "seed", "omega_idx", "hv"):
        assert col in cols


def test_md_table_includes_pvalue_and_tetrarl(tmp_path: Path):
    runs, yml = _build_matrix(tmp_path)
    _run([
        "--matrix-yaml", str(yml),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--ref-point", "0,-100,-1,-1",
    ])
    md = (runs / "hv_comparison.md").read_text()
    assert "preference_ppo" in md
    assert "max_p" in md
    assert "p_value" in md or "p-value" in md
    # TetraRL row should NOT have a self-comparison p-value.
    # Other rows should.
    assert "vs preference_ppo" in md or "vs TetraRL" in md or "vs tetrarl" in md.lower()


def test_summary_printed_to_stdout(tmp_path: Path):
    runs, yml = _build_matrix(tmp_path)
    res = _run([
        "--matrix-yaml", str(yml),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--ref-point", "0,-100,-1,-1",
    ])
    assert "preference_ppo" in res.stdout
    assert "HV" in res.stdout or "hv" in res.stdout.lower()
