"""Tests for scripts/week10_make_reward_curves.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "week10_make_reward_curves.py"


def _write_jsonl(path: Path, *, n_episodes: int = 3, n_steps: int = 5,
                 reward: float = 1.0, latency: float = 5.0,
                 energy: float = 1e-3, memory: float = 0.2,
                 omega=(0.25, 0.25, 0.25, 0.25)) -> None:
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


def _build_minimal(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    runs.mkdir()
    agents = ["preference_ppo", "max_p"]
    omegas = [(1.0, 0.0, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25)]
    seeds = [0, 1]
    yml = []
    for ag in agents:
        for o_idx, om in enumerate(omegas):
            for sd in seeds:
                fname = f"none__{ag}__seed{sd}__o{o_idx}.jsonl"
                _write_jsonl(runs / fname, omega=om,
                             reward=2.0 if ag == "preference_ppo" else 1.0)
                yml.append({
                    "env_name": "dag_scheduler_mo-v0",
                    "agent_type": ag, "ablation": "none",
                    "platform": "mac_stub", "n_episodes": 3,
                    "seed": sd, "out_dir": str(runs),
                    "extra": {"omega": list(om), "omega_idx": o_idx,
                              "jsonl_name": fname},
                    "n_envs": 1,
                })
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"configs": yml}, sort_keys=False))
    return runs, matrix


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True,
    )


def test_help_lists_required_flags():
    res = _run(["--help"])
    assert res.returncode == 0
    for flag in ("--matrix-yaml", "--runs-dir", "--out-dir"):
        assert flag in res.stdout


def test_emits_png_svg_for_walltime_and_energy(tmp_path: Path):
    runs, matrix = _build_minimal(tmp_path)
    res = _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
    ])
    assert res.returncode == 0, res.stderr
    for fname in ("reward_vs_walltime.png", "reward_vs_walltime.svg",
                  "reward_vs_energy.png", "reward_vs_energy.svg"):
        assert (runs / fname).exists(), fname


def test_does_not_use_step_axis(tmp_path: Path):
    """We assert the script does NOT emit a reward_vs_steps figure
    (per §9.6 pitfall — unfair to on-policy)."""
    runs, matrix = _build_minimal(tmp_path)
    _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
    ])
    assert not (runs / "reward_vs_steps.png").exists()


def test_handles_missing_jsonl_gracefully(tmp_path: Path):
    """If a manifest entry refers to a JSONL that doesn't exist, the
    script should warn and skip — not crash."""
    runs, matrix = _build_minimal(tmp_path)
    # Inject a bogus entry into the matrix.
    doc = yaml.safe_load(matrix.read_text())
    doc["configs"].append({
        "env_name": "dag_scheduler_mo-v0",
        "agent_type": "missing_agent", "ablation": "none",
        "platform": "mac_stub", "n_episodes": 1, "seed": 99,
        "out_dir": str(runs),
        "extra": {"omega": [1, 0, 0, 0], "omega_idx": 0,
                  "jsonl_name": "none__missing_agent__seed99__o0.jsonl"},
        "n_envs": 1,
    })
    matrix.write_text(yaml.safe_dump(doc, sort_keys=False))
    res = _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
    ])
    assert res.returncode == 0
    assert (runs / "reward_vs_walltime.png").exists()


def test_smoothing_window_flag(tmp_path: Path):
    """--smoothing-window controls the rolling-mean window size."""
    runs, matrix = _build_minimal(tmp_path)
    res = _run([
        "--matrix-yaml", str(matrix),
        "--runs-dir", str(runs),
        "--out-dir", str(runs),
        "--smoothing-window", "5",
    ])
    assert res.returncode == 0
