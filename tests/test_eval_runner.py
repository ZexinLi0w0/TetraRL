"""Tests for tetrarl.eval.runner — unified eval harness (Week 8)."""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from tetrarl.core.framework import StaticPreferencePlane  # noqa: F401
from tetrarl.eval.runner import (
    EvalConfig,
    EvalRunner,
    RunResult,
    _make_telemetry,
    load_sweep_yaml,
)
from tetrarl.morl.native.override import HardwareTelemetry


def _make_cfg(out_dir: Path, **overrides) -> EvalConfig:
    """Helper to build an EvalConfig with sane defaults; not a fixture."""
    defaults = dict(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=out_dir,
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def test_eval_config_round_trip_dict():
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=3,
        seed=7,
        out_dir=Path("runs/eval"),
    )
    d = cfg.to_dict()
    cfg2 = EvalConfig.from_dict(d)
    assert cfg2.env_name == cfg.env_name
    assert cfg2.agent_type == cfg.agent_type
    assert cfg2.ablation == cfg.ablation
    assert cfg2.platform == cfg.platform
    assert cfg2.n_episodes == cfg.n_episodes
    assert cfg2.seed == cfg.seed
    assert Path(cfg2.out_dir) == Path(cfg.out_dir)


def test_eval_config_round_trip_yaml(tmp_path):
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=2,
        seed=42,
        out_dir=tmp_path / "runs",
    )
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    cfg2 = EvalConfig.from_yaml(path)
    assert cfg2.env_name == cfg.env_name
    assert cfg2.agent_type == cfg.agent_type
    assert cfg2.ablation == cfg.ablation
    assert cfg2.platform == cfg.platform
    assert cfg2.n_episodes == cfg.n_episodes
    assert cfg2.seed == cfg.seed
    assert Path(cfg2.out_dir) == Path(cfg.out_dir)


def test_load_sweep_yaml(tmp_path):
    yaml_text = """
configs:
  - env_name: CartPole-v1
    agent_type: random
    ablation: none
    platform: mac_stub
    n_episodes: 1
    seed: 0
    out_dir: runs/a
  - env_name: CartPole-v1
    agent_type: preference_ppo
    ablation: preference_plane
    platform: mac_stub
    n_episodes: 1
    seed: 1
    out_dir: runs/b
"""
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml_text)
    cfgs = load_sweep_yaml(path)
    assert len(cfgs) == 2
    assert cfgs[0].agent_type == "random"
    assert cfgs[0].ablation == "none"
    assert cfgs[1].agent_type == "preference_ppo"
    assert cfgs[1].ablation == "preference_plane"


def test_run_random_ablation_none_returns_run_result(tmp_path):
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=2,
        seed=0,
        out_dir=tmp_path,
    )
    result = EvalRunner().run(cfg)
    assert isinstance(result, RunResult)
    assert result.n_steps > 0
    assert result.n_episodes == 2
    assert result.mean_reward > 0
    assert result.tail_p99_ms >= 0
    assert result.wall_time_s > 0


def test_ablation_none_uses_real_preference_plane(tmp_path):
    cfg = _make_cfg(tmp_path, ablation="none")
    framework = EvalRunner()._build_framework(cfg)
    assert framework.preference_plane.__class__.__name__ == "StaticPreferencePlane"


def test_ablation_preference_plane_uses_null_variant(tmp_path):
    cfg = _make_cfg(tmp_path, ablation="preference_plane")
    framework = EvalRunner()._build_framework(cfg)
    assert framework.preference_plane.__class__.__name__ == "_NullPreferencePlane"
    omega = framework.preference_plane.get()
    assert np.isclose(np.sum(omega), 1.0, atol=1e-6)
    assert np.allclose(omega, omega[0])  # uniform


def test_ablation_resource_manager_uses_null_variant(tmp_path):
    cfg = _make_cfg(tmp_path, ablation="resource_manager")
    framework = EvalRunner()._build_framework(cfg)
    assert framework.resource_manager.__class__.__name__ == "_NullResourceManager"
    idx = framework.resource_manager.decide_dvfs(
        HardwareTelemetry(memory_util=0.99), n_levels=10
    )
    assert idx == 9


def test_ablation_rl_arbiter_substitutes_random(tmp_path):
    cfg_abl = _make_cfg(
        tmp_path / "abl", agent_type="preference_ppo", ablation="rl_arbiter"
    )
    cfg_real = _make_cfg(
        tmp_path / "real", agent_type="preference_ppo", ablation="none"
    )
    fw_abl = EvalRunner()._build_framework(cfg_abl)
    fw_real = EvalRunner()._build_framework(cfg_real)
    assert fw_abl.rl_arbiter.__class__.__name__ == "_RandomArbiter"
    assert fw_real.rl_arbiter.__class__.__name__ != "_RandomArbiter"


def test_ablation_override_layer_never_fires(tmp_path):
    cfg = _make_cfg(tmp_path, ablation="override_layer")
    framework = EvalRunner()._build_framework(cfg)
    fired, action = framework.override_layer.step(
        HardwareTelemetry(latency_ema_ms=1e9)
    )
    assert fired is False
    assert action is None
    assert framework.override_layer.fire_count == 0


def test_run_sweep_writes_jsonl_per_config_and_aggregated_csv(tmp_path):
    configs = [
        _make_cfg(tmp_path, ablation="none", seed=0),
        _make_cfg(tmp_path, ablation="preference_plane", seed=1),
        _make_cfg(tmp_path, ablation="resource_manager", seed=2),
    ]
    EvalRunner().run_sweep(configs)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 3
    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 4  # header + 3 data rows


def test_seed_reproducibility_same_config_same_rewards(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cfg_a = _make_cfg(out_a, n_episodes=1, seed=42)
    cfg_b = _make_cfg(out_b, n_episodes=1, seed=42)
    EvalRunner().run(cfg_a)
    EvalRunner().run(cfg_b)
    jsonl_a = next(out_a.glob("*.jsonl"))
    jsonl_b = next(out_b.glob("*.jsonl"))
    rewards_a = [json.loads(line).get("reward") for line in jsonl_a.read_text().splitlines()]
    rewards_b = [json.loads(line).get("reward") for line in jsonl_b.read_text().splitlines()]
    assert rewards_a == rewards_b
    assert len(rewards_a) > 0


def test_cli_help_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "tetrarl.eval.runner", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_run_sweep_empty_list_returns_empty():
    out = EvalRunner().run_sweep([])
    assert out == []


def test_invalid_env_name_raises_clear_error(tmp_path):
    cfg = _make_cfg(tmp_path, env_name="NotAGymEnv-v0")
    with pytest.raises(Exception) as exc_info:
        EvalRunner().run(cfg)
    assert "NotAGymEnv" in str(exc_info.value)


def test_mac_stub_platform_no_dvfs_writes(tmp_path):
    cfg = _make_cfg(tmp_path, platform="mac_stub")
    framework = EvalRunner()._build_framework(cfg)
    dvfs = getattr(framework, "dvfs_controller", None)
    assert dvfs is None or getattr(dvfs, "stub", False) is True


def test_out_dir_created_if_missing(tmp_path):
    out_dir = tmp_path / "deeply" / "nested" / "missing"
    assert not out_dir.exists()
    cfg = _make_cfg(out_dir, n_episodes=1, seed=0)
    EvalRunner().run(cfg)
    assert out_dir.exists()
    jsonl_files = list(out_dir.glob("*.jsonl"))
    assert len(jsonl_files) >= 1


def test_make_telemetry_orin_agx_warns_about_stub_fallback():
    """W9 Task A: orin_agx must emit a WARN when the harness silently
    falls back to the Mac stub instead of wiring up a real tegrastats
    daemon. The function still returns the Mac stub source (behaviour
    preserved), but the user is now told the truth."""
    with pytest.warns(RuntimeWarning) as record:
        source, adapter = _make_telemetry("orin_agx")
    assert len(record) == 1
    msg = str(record[0].message).lower()
    assert "orin" in msg
    assert "stub" in msg
    # Behaviour preserved: still returns the Mac stub source.
    assert source.__class__.__name__ == "_MacStubTelemetry"


def test_make_telemetry_orin_nano_also_warns_about_stub_fallback():
    """Both orin_* variants must trigger the W9 stub-fallback WARN."""
    with pytest.warns(RuntimeWarning) as record:
        source, adapter = _make_telemetry("orin_nano")
    assert len(record) == 1
    msg = str(record[0].message).lower()
    assert "orin" in msg
    assert "stub" in msg
    assert source.__class__.__name__ == "_MacStubTelemetry"


def test_make_telemetry_mac_stub_does_not_warn():
    """The legitimate mac_stub path must remain warning-free so the
    eval harness on developer laptops stays quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        source, adapter = _make_telemetry("mac_stub")
    assert source.__class__.__name__ == "_MacStubTelemetry"
