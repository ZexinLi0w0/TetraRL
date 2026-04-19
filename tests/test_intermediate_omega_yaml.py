"""Tests for scripts/week10_make_intermediate_omega_yaml.py.

Covers Task 4 of the W10 ω-sensitivity sweep: the four intermediate
omegas (indices 5..8), the default 48-row matrix, and the per-entry
``omega_idx`` invariant that prevents collision with the original
five anchor omegas (0..4) in the same run dir.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tetrarl.eval.runner import EvalConfig, load_sweep_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "week10_make_intermediate_omega_yaml.py"


def _import_script_module():
    """Load the script as a module so we can introspect its constants."""
    spec = importlib.util.spec_from_file_location(
        "week10_make_intermediate_omega_yaml", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cli(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_intermediate_omegas_4d_length_and_simplex():
    mod = _import_script_module()
    omegas = mod.INTERMEDIATE_OMEGAS_4D
    assert len(omegas) == 4
    for w in omegas:
        assert len(w) == 4
        assert sum(w) == pytest.approx(1.0, abs=1e-9)
        for v in w:
            assert 0.0 <= v <= 1.0


def test_intermediate_omega_index_offset_is_5():
    """Indices 5..8 must not collide with the anchor sweep (0..4)."""
    mod = _import_script_module()
    assert mod.INTERMEDIATE_OMEGA_INDEX_OFFSET == 5


def test_build_assigns_indices_5_through_8():
    mod = _import_script_module()
    cfgs: list[EvalConfig] = mod.build_intermediate_omega_configs(
        agents=["preference_ppo"],
        envs=["dag_scheduler_mo-v0"],
        seeds=[0],
        n_episodes=10,
        platform="orin_agx",
        out_dir=Path("runs/w10_orin_full_fixed"),
        ablation="none",
    )
    seen = sorted({int(c.extra["omega_idx"]) for c in cfgs})
    assert seen == [5, 6, 7, 8]
    for c in cfgs:
        assert int(c.extra["omega_idx"]) in {5, 6, 7, 8}
        # jsonl_name must keep the __o<idx> suffix to avoid collision
        # with the existing __o0..__o4 files in the same run dir.
        assert "__o" in c.extra["jsonl_name"]
        assert c.extra["jsonl_name"].endswith(".jsonl")


def test_default_cli_yields_48_entries(tmp_path: Path):
    out_yaml = tmp_path / "w10_intermediate.yaml"
    res = _run_cli(["--out", str(out_yaml)])
    assert res.returncode == 0, res.stderr
    cfgs = load_sweep_yaml(out_yaml)
    # Default: 4 agents (preference_ppo,pcn,pd_morl,max_performance) *
    # 1 env (dag_scheduler_mo-v0) * 4 omegas * 3 seeds = 48.
    assert len(cfgs) == 48


def test_default_cli_uses_orin_agx_and_dag_only(tmp_path: Path):
    out_yaml = tmp_path / "w10_intermediate.yaml"
    res = _run_cli(["--out", str(out_yaml)])
    assert res.returncode == 0, res.stderr
    cfgs = load_sweep_yaml(out_yaml)
    assert all(c.platform == "orin_agx" for c in cfgs)
    assert all(c.env_name == "dag_scheduler_mo-v0" for c in cfgs)


def test_default_cli_emits_correct_jsonl_suffixes(tmp_path: Path):
    out_yaml = tmp_path / "w10_intermediate.yaml"
    res = _run_cli(["--out", str(out_yaml)])
    assert res.returncode == 0, res.stderr
    cfgs = load_sweep_yaml(out_yaml)
    suffixes = {c.extra["jsonl_name"].split("__")[-1] for c in cfgs}
    # Each filename ends in __o<5..8>.jsonl
    assert suffixes == {"o5.jsonl", "o6.jsonl", "o7.jsonl", "o8.jsonl"}


def test_help_lists_required_flags():
    res = _run_cli(["--help"])
    assert res.returncode == 0
    for flag in ("--out", "--agents", "--seeds", "--n-episodes",
                 "--platform", "--out-dir", "--ablation"):
        assert flag in res.stdout
