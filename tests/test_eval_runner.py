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


# -----------------------------------------------------------------------------
# Week 9: n_envs (multi-env scaling) tests
# -----------------------------------------------------------------------------


def test_eval_config_default_n_envs_is_one():
    """W9: n_envs defaults to 1 for backward compatibility."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=Path("runs/eval"),
    )
    assert cfg.n_envs == 1


def test_eval_config_round_trip_dict_preserves_n_envs():
    """W9: dict round-trip preserves n_envs > 1."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=3,
        seed=7,
        out_dir=Path("runs/eval"),
        n_envs=4,
    )
    d = cfg.to_dict()
    assert d["n_envs"] == 4
    cfg2 = EvalConfig.from_dict(d)
    assert cfg2.n_envs == 4


def test_eval_config_round_trip_yaml_preserves_n_envs(tmp_path):
    """W9: YAML round-trip preserves n_envs > 1."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=2,
        seed=42,
        out_dir=tmp_path / "runs",
        n_envs=2,
    )
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    cfg2 = EvalConfig.from_yaml(path)
    assert cfg2.n_envs == 2


def test_load_sweep_yaml_omitting_n_envs_defaults_to_one(tmp_path):
    """W9 back-compat: pre-W9 sweep YAMLs without n_envs still parse."""
    yaml_text = """
configs:
  - env_name: CartPole-v1
    agent_type: random
    ablation: none
    platform: mac_stub
    n_episodes: 1
    seed: 0
    out_dir: runs/a
"""
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml_text)
    cfgs = load_sweep_yaml(path)
    assert len(cfgs) == 1
    assert cfgs[0].n_envs == 1


def test_load_sweep_yaml_with_n_envs_field(tmp_path):
    """W9: sweep YAMLs may set n_envs explicitly."""
    yaml_text = """
configs:
  - env_name: CartPole-v1
    agent_type: random
    ablation: none
    platform: mac_stub
    n_episodes: 1
    seed: 0
    out_dir: runs/a
    n_envs: 4
"""
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml_text)
    cfgs = load_sweep_yaml(path)
    assert len(cfgs) == 1
    assert cfgs[0].n_envs == 4


def test_run_with_n_envs_2_total_episodes_is_2x(tmp_path):
    """W9: with n_envs=2 and n_episodes=2 (per-env), total = 4 episodes."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=2,
        seed=0,
        out_dir=tmp_path,
        n_envs=2,
    )
    result = EvalRunner().run(cfg)
    assert isinstance(result, RunResult)
    assert result.n_episodes == 4


def test_run_with_n_envs_2_jsonl_filename_includes_nenvs(tmp_path):
    """W9: vector-path JSONL filename has the nenvs suffix."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path,
        n_envs=2,
    )
    EvalRunner().run(cfg)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1
    assert "nenvs2" in jsonl_files[0].name


def test_run_with_n_envs_2_jsonl_lines_have_env_id(tmp_path):
    """W9: every JSONL line in the vector path carries an integer env_id."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path,
        n_envs=2,
    )
    EvalRunner().run(cfg)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1
    lines = jsonl_files[0].read_text().splitlines()
    assert len(lines) > 0
    seen_env_ids = set()
    for line in lines:
        rec = json.loads(line)
        assert "env_id" in rec, f"line missing env_id: {rec}"
        assert isinstance(rec["env_id"], int)
        assert rec["env_id"] in (0, 1)
        seen_env_ids.add(rec["env_id"])
    assert seen_env_ids == {0, 1}


def test_run_with_n_envs_1_jsonl_lines_have_no_env_id_key(tmp_path):
    """W9 back-compat: single-env path must NOT emit env_id (byte-identical)."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=tmp_path,
        n_envs=1,
    )
    EvalRunner().run(cfg)
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 1
    lines = jsonl_files[0].read_text().splitlines()
    assert len(lines) > 0
    for line in lines:
        rec = json.loads(line)
        assert "env_id" not in rec, f"single-env path leaked env_id: {rec}"


def test_run_with_n_envs_2_seed_reproducibility(tmp_path):
    """W9: two runs with the same seed yield the same per-env reward sequence."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cfg_a = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=42,
        out_dir=out_a,
        n_envs=2,
    )
    cfg_b = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=42,
        out_dir=out_b,
        n_envs=2,
    )
    EvalRunner().run(cfg_a)
    EvalRunner().run(cfg_b)
    jsonl_a = next(out_a.glob("*.jsonl"))
    jsonl_b = next(out_b.glob("*.jsonl"))

    def _per_env_rewards(path):
        per = {0: [], 1: []}
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            per[int(rec["env_id"])].append(float(rec["reward"]))
        return per

    pa = _per_env_rewards(jsonl_a)
    pb = _per_env_rewards(jsonl_b)
    assert pa[0] == pb[0]
    assert pa[1] == pb[1]
    assert len(pa[0]) > 0
    assert len(pa[1]) > 0
