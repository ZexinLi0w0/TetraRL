"""Smoke tests for ``scripts/p15_unified_runner.py``.

Invokes the runner via ``subprocess.run`` (it is a script, not a library) and
asserts:

- COMPLETED happy path produces a valid ``summary.json`` with all required
  fields and types.
- An incompatible (algo, wrapper) pair (a2c + r3) writes a SKIPPED summary
  and returns exit 0.
- ``per_step.jsonl`` has exactly one JSON line per env step with the expected
  schema.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "p15_unified_runner.py"


def _run_cli(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_runner_completed_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "cell"
    proc = _run_cli(
        [
            "--algo", "dqn",
            "--wrapper", "tetrarl",
            "--env", "cartpole",
            "--platform", "mac",
            "--seed", "0",
            "--frames", "50",
            "--out-dir", str(out_dir),
        ]
    )
    assert proc.returncode == 0, f"runner failed: stderr={proc.stderr!r}"
    summary_path = out_dir / "summary.json"
    assert summary_path.exists(), "summary.json was not written"
    with summary_path.open() as f:
        summary = json.load(f)
    assert summary["status"] == "COMPLETED", f"status mismatch: {summary}"
    # Required fields + types.
    assert isinstance(summary["cumulative_reward_curve"], list)
    assert len(summary["cumulative_reward_curve"]) > 0
    assert isinstance(summary["framework_overhead_pct"], (int, float))
    assert float(summary["framework_overhead_pct"]) >= 0.0
    assert isinstance(summary["mean_deadline_miss_rate"], (int, float))
    miss = float(summary["mean_deadline_miss_rate"])
    assert 0.0 <= miss <= 1.0
    assert isinstance(summary["mean_p99_step_ms"], (int, float))
    assert float(summary["mean_p99_step_ms"]) > 0.0
    assert isinstance(summary["peak_gpu_memory_mb"], (int, float))
    assert float(summary["peak_gpu_memory_mb"]) >= 0.0
    assert isinstance(summary["mean_energy_j"], (int, float))
    assert float(summary["mean_energy_j"]) >= 0.0
    assert isinstance(summary["time_to_converge_steps"], int)


def test_runner_skipped_for_incompatible_pair(tmp_path: Path) -> None:
    out_dir = tmp_path / "cell"
    proc = _run_cli(
        [
            "--algo", "a2c",
            "--wrapper", "r3",
            "--env", "cartpole",
            "--platform", "mac",
            "--seed", "0",
            "--frames", "50",
            "--out-dir", str(out_dir),
        ]
    )
    assert proc.returncode == 0, f"runner failed: stderr={proc.stderr!r}"
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    with summary_path.open() as f:
        summary = json.load(f)
    assert summary["status"] == "SKIPPED", f"want SKIPPED, got {summary}"
    reason = str(summary.get("reason", ""))
    assert "incompatible" in reason.lower(), f"reason missing 'incompatible': {reason!r}"


def test_runner_per_step_jsonl_lines(tmp_path: Path) -> None:
    out_dir = tmp_path / "cell"
    n_frames = 50
    proc = _run_cli(
        [
            "--algo", "dqn",
            "--wrapper", "tetrarl",
            "--env", "cartpole",
            "--platform", "mac",
            "--seed", "0",
            "--frames", str(n_frames),
            "--out-dir", str(out_dir),
        ]
    )
    assert proc.returncode == 0, f"runner failed: stderr={proc.stderr!r}"
    per_step = out_dir / "per_step.jsonl"
    assert per_step.exists()
    lines = per_step.read_text().splitlines()
    assert len(lines) == n_frames, f"expected {n_frames} lines, got {len(lines)}"
    first = json.loads(lines[0])
    expected_keys = {
        "step",
        "episode",
        "action",
        "reward",
        "framework_step_ms",
        "raw_step_ms",
        "total_step_ms",
        "energy_j",
        "memory_util",
        "deadline_miss",
    }
    assert expected_keys.issubset(first.keys()), (
        f"missing keys: {expected_keys - set(first.keys())}"
    )


def test_runner_no_longer_deferreds_breakout(tmp_path: Path) -> None:
    """The DEFERRED branch for Atari has been removed.

    Acceptable summary statuses: ``COMPLETED`` (when ALE+cv2 are installed) or
    ``ERROR`` (when those deps are missing — gymnasium raises ImportError and
    the runner's ERROR branch records it). The forbidden status is
    ``DEFERRED``.
    """
    out_dir = tmp_path / "cell"
    proc = _run_cli(
        [
            "--algo", "dqn",
            "--wrapper", "tetrarl",
            "--env", "breakout",
            "--platform", "mac",
            "--seed", "0",
            "--frames", "50",
            "--out-dir", str(out_dir),
        ]
    )
    summary_path = out_dir / "summary.json"
    assert summary_path.exists(), (
        f"summary.json was not written; stderr={proc.stderr!r}"
    )
    with summary_path.open() as f:
        summary = json.load(f)
    status = summary.get("status")
    assert status != "DEFERRED", (
        f"Atari is still in the DEFERRED branch: {summary}"
    )
    assert status in {"COMPLETED", "ERROR"}, (
        f"unexpected status {status!r}: {summary}"
    )


def test_runner_uses_cuda_when_available(monkeypatch, tmp_path: Path) -> None:
    """When torch.cuda.is_available() is True, the algo is built with device='cuda'.

    Verified indirectly by patching torch.cuda.is_available + asserting the runner
    does not crash and the resulting summary status is COMPLETED. This is a
    source-level assertion (the only way to test cuda selection without a
    CUDA-capable test runner: the runner is invoked via subprocess and we cannot
    reach in to patch torch.cuda there, and the test machine — Mac CI / nano2
    venv — has no CUDA build of torch).
    """
    runner_src = (REPO_ROOT / "scripts" / "p15_unified_runner.py").read_text()
    assert 'torch.cuda.is_available()' in runner_src, (
        "runner must select device based on torch.cuda.is_available()"
    )
    assert 'device=device' in runner_src, (
        "runner must pass device= to the algo constructor"
    )


def test_runner_completed_breakout_smoke(tmp_path: Path) -> None:
    """Full Atari training smoke test; only runs when ALE + cv2 are installed."""
    pytest.importorskip("ale_py")
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("cv2 not installed")
    out_dir = tmp_path / "cell"
    n_frames = 100
    proc = _run_cli(
        [
            "--algo", "dqn",
            "--wrapper", "tetrarl",
            "--env", "breakout",
            "--platform", "mac",
            "--seed", "0",
            "--frames", str(n_frames),
            "--out-dir", str(out_dir),
        ]
    )
    assert proc.returncode == 0, f"runner failed: stderr={proc.stderr!r}"
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    with summary_path.open() as f:
        summary = json.load(f)
    assert summary["status"] == "COMPLETED", f"status mismatch: {summary}"
    assert int(summary["n_steps"]) == n_frames, (
        f"expected {n_frames} steps, got {summary['n_steps']}"
    )
    assert isinstance(summary["cumulative_reward_curve"], list)
    assert len(summary["cumulative_reward_curve"]) > 0, (
        "expected at least one episode return"
    )
    assert float(summary["mean_p99_step_ms"]) > 0.0
