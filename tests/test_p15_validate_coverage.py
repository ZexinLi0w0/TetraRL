"""End-to-end tests for ``scripts/p15_validate_coverage.py``.

Each test:
1. Builds a tiny inline matrix YAML under ``tmp_path``.
2. Runs ``scripts/p15_unified_runner.py`` for each cell in the matrix to
   populate ``runs_root`` with real summary.json files.
3. Optionally tampers with the resulting directory (delete / overwrite a
   summary).
4. Runs ``scripts/p15_validate_coverage.py`` and asserts on its exit code +
   stdout markers.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "p15_unified_runner.py"
VALIDATOR = REPO_ROOT / "scripts" / "p15_validate_coverage.py"


def _cell_dir_name(env: str, hw: str, algo: str, wrapper: str, seed: int) -> str:
    return f"{env}__{hw}__{algo}__{wrapper}__seed{seed}"


def _run_cell(out_dir: Path, *, algo: str, wrapper: str, env: str,
              platform: str, seed: int, frames: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable, str(RUNNER),
            "--algo", algo,
            "--wrapper", wrapper,
            "--env", env,
            "--platform", platform,
            "--seed", str(seed),
            "--frames", str(frames),
            "--out-dir", str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"runner failed for {out_dir.name}: {proc.stderr!r}"


def _run_validator(matrix_path: Path, runs_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(VALIDATOR),
            "--matrix", str(matrix_path),
            "--runs-root", str(runs_root),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _write_two_cell_matrix(matrix_path: Path) -> list[dict]:
    """Build a matrix with one COMPLETED cell and one SKIPPED cell."""
    cells = [
        {"env": "cartpole", "hw": "mac", "algo": "dqn", "wrapper": "tetrarl",
         "seed": 0, "frames": 50, "expected": "COMPLETED"},
        {"env": "cartpole", "hw": "mac", "algo": "a2c", "wrapper": "r3",
         "seed": 0, "frames": 50, "expected": "SKIPPED"},
    ]
    yaml_text = (
        "metadata:\n"
        "  total_runs: 2\n"
        "  expected_completed: 1\n"
        "  expected_skipped: 1\n"
        "runs:\n"
        "  - {env: cartpole, hw: mac, algo: dqn, wrapper: tetrarl, seed: 0, frames: 50, expected: COMPLETED}\n"
        "  - {env: cartpole, hw: mac, algo: a2c, wrapper: r3, seed: 0, frames: 50, expected: SKIPPED}\n"
    )
    matrix_path.write_text(yaml_text)
    return cells


def test_validator_all_ok(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    runs_root = tmp_path / "runs"
    cells = _write_two_cell_matrix(matrix_path)
    for c in cells:
        cell_dir = runs_root / _cell_dir_name(
            c["env"], c["hw"], c["algo"], c["wrapper"], c["seed"]
        )
        _run_cell(
            cell_dir,
            algo=c["algo"], wrapper=c["wrapper"], env=c["env"],
            platform=c["hw"], seed=c["seed"], frames=c["frames"],
        )

    proc = _run_validator(matrix_path, runs_root)
    assert proc.returncode == 0, (
        f"validator failed (rc={proc.returncode}): "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ALL OK" in proc.stdout, f"missing ALL OK marker: {proc.stdout!r}"


def test_validator_detects_missing(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    runs_root = tmp_path / "runs"
    cells = _write_two_cell_matrix(matrix_path)
    for c in cells:
        cell_dir = runs_root / _cell_dir_name(
            c["env"], c["hw"], c["algo"], c["wrapper"], c["seed"]
        )
        _run_cell(
            cell_dir,
            algo=c["algo"], wrapper=c["wrapper"], env=c["env"],
            platform=c["hw"], seed=c["seed"], frames=c["frames"],
        )
    # Delete the COMPLETED cell's summary.json.
    completed = cells[0]
    completed_summary = (
        runs_root
        / _cell_dir_name(completed["env"], completed["hw"], completed["algo"],
                         completed["wrapper"], completed["seed"])
        / "summary.json"
    )
    assert completed_summary.exists()
    completed_summary.unlink()

    proc = _run_validator(matrix_path, runs_root)
    assert proc.returncode == 1, f"want exit 1, got {proc.returncode}; stdout={proc.stdout!r}"
    assert "MISSING" in proc.stdout, f"missing MISSING marker: {proc.stdout!r}"
    assert "FAILED" in proc.stdout, f"missing FAILED marker: {proc.stdout!r}"


def test_validator_detects_status_mismatch(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    runs_root = tmp_path / "runs"
    yaml_text = (
        "metadata:\n"
        "  total_runs: 1\n"
        "  expected_completed: 1\n"
        "  expected_skipped: 0\n"
        "runs:\n"
        "  - {env: cartpole, hw: mac, algo: dqn, wrapper: tetrarl, seed: 0, frames: 50, expected: COMPLETED}\n"
    )
    matrix_path.write_text(yaml_text)
    cell = {"env": "cartpole", "hw": "mac", "algo": "dqn", "wrapper": "tetrarl",
            "seed": 0, "frames": 50}
    cell_dir = runs_root / _cell_dir_name(
        cell["env"], cell["hw"], cell["algo"], cell["wrapper"], cell["seed"]
    )
    _run_cell(
        cell_dir,
        algo=cell["algo"], wrapper=cell["wrapper"], env=cell["env"],
        platform=cell["hw"], seed=cell["seed"], frames=cell["frames"],
    )
    summary_path = cell_dir / "summary.json"
    assert summary_path.exists()
    summary_path.write_text(json.dumps({"status": "BROKEN"}))

    proc = _run_validator(matrix_path, runs_root)
    assert proc.returncode == 1, (
        f"want exit 1, got {proc.returncode}; stdout={proc.stdout!r}"
    )
    assert "MISMATCH" in proc.stdout, f"missing MISMATCH marker: {proc.stdout!r}"
