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


# -----------------------------------------------------------------------------
# P9: --use-real-telemetry / TegrastatsDaemon wiring
# -----------------------------------------------------------------------------


def test_eval_config_default_use_real_telemetry_is_false():
    """P9: use_real_telemetry defaults to False for backward compatibility."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="mac_stub",
        n_episodes=1,
        seed=0,
        out_dir=Path("runs/eval"),
    )
    assert cfg.use_real_telemetry is False


def test_eval_config_round_trip_dict_preserves_use_real_telemetry():
    """P9: dict round-trip preserves use_real_telemetry=True."""
    cfg = EvalConfig(
        env_name="CartPole-v1",
        agent_type="random",
        ablation="none",
        platform="orin_nano",
        n_episodes=1,
        seed=0,
        out_dir=Path("runs/eval"),
        use_real_telemetry=True,
    )
    d = cfg.to_dict()
    assert d["use_real_telemetry"] is True
    cfg2 = EvalConfig.from_dict(d)
    assert cfg2.use_real_telemetry is True


def test_make_telemetry_orin_with_real_tele_no_tegrastats_falls_back_with_warning(monkeypatch):
    """P9: --use-real-telemetry on a host without tegrastats binary should
    fall back to Mac stub with a RuntimeWarning rather than crash."""
    import tetrarl.eval.runner as runner_mod

    # Force shutil.which to return None for "tegrastats" so we simulate Mac.
    import shutil as _shutil
    real_which = _shutil.which

    def fake_which(name, *a, **kw):
        if name == "tegrastats":
            return None
        return real_which(name, *a, **kw)

    monkeypatch.setattr(_shutil, "which", fake_which)

    with pytest.warns(RuntimeWarning) as record:
        source, adapter = runner_mod._make_telemetry("orin_nano", use_real_telemetry=True)
    assert any("tegrastats" in str(r.message).lower() for r in record)
    assert source.__class__.__name__ == "_MacStubTelemetry"


def test_make_telemetry_orin_with_real_tele_uses_real_when_daemon_starts(monkeypatch):
    """P9: when TegrastatsDaemon import + start succeed, _make_telemetry
    must return the real wrapper (not the Mac stub). Uses a fake daemon
    so the test runs on Mac without a real tegrastats binary."""
    import tetrarl.eval.runner as runner_mod
    import tetrarl.sys.tegra_daemon as daemon_mod

    # Pretend tegrastats is on PATH so the binary check passes.
    import shutil as _shutil

    def fake_which(name, *a, **kw):
        if name == "tegrastats":
            return "/fake/tegrastats"
        return _shutil.which(name, *a, **kw)

    monkeypatch.setattr(_shutil, "which", fake_which)

    started = {"v": False}

    class _FakeReading:
        ram_used_mb = 4096
        ram_total_mb = 8192

    class _FakeDaemon:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            started["v"] = True

        def stop(self):
            pass

        def latest(self):
            return _FakeReading()

    monkeypatch.setattr(daemon_mod, "TegrastatsDaemon", _FakeDaemon)

    source, adapter = runner_mod._make_telemetry("orin_nano", use_real_telemetry=True)
    assert source.__class__.__name__ != "_MacStubTelemetry", \
        f"expected real wrapper, got {source.__class__.__name__}"
    assert started["v"] is True

    # The adapter should produce a HardwareTelemetry with memory_util ~ 0.5.
    source.update(latency_ms=1.0, energy_remaining_j=999.0, memory_util=0.0)
    reading = source.latest()
    hw = adapter(reading)
    assert hw.memory_util is not None
    assert abs(hw.memory_util - 0.5) < 1e-6


def test_make_telemetry_legacy_no_flag_still_warns_and_stubs():
    """P9 back-compat: existing W9 behaviour preserved when
    use_real_telemetry is not passed."""
    with pytest.warns(RuntimeWarning) as record:
        source, _adapter = _make_telemetry("orin_nano")
    assert source.__class__.__name__ == "_MacStubTelemetry"
    assert any("orin" in str(r.message).lower() for r in record)


def test_make_telemetry_mac_stub_with_real_tele_falls_back_with_warning():
    """P9: --use-real-telemetry on platform=mac_stub is nonsensical;
    must fall back to stub with a RuntimeWarning."""
    with pytest.warns(RuntimeWarning) as record:
        source, _adapter = _make_telemetry("mac_stub", use_real_telemetry=True)
    assert source.__class__.__name__ == "_MacStubTelemetry"
    assert len(record) >= 1


def test_real_jetson_telemetry_warmup_waits_for_first_reading(monkeypatch):
    """P9 fix: _RealJetsonTelemetry must wait for the first daemon reading
    so the eval loop sees real memory_util, not the None-fallback 0.0."""
    import tetrarl.eval.runner as runner_mod
    import tetrarl.sys.tegra_daemon as daemon_mod
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name, *a, **kw: "/fake/tegrastats" if name == "tegrastats" else None)

    class _Reading:
        ram_used_mb = 4096
        ram_total_mb = 8192

    class _SlowDaemon:
        """Returns None for the first 3 latest() calls, then a real reading."""

        def __init__(self, *a, **kw):
            self._calls = 0

        def start(self):
            pass

        def stop(self):
            pass

        def latest(self):
            self._calls += 1
            if self._calls <= 3:
                return None
            return _Reading()

    monkeypatch.setattr(daemon_mod, "TegrastatsDaemon", _SlowDaemon)

    source, adapter = runner_mod._make_telemetry("orin_nano", use_real_telemetry=True)
    # The wrapper should have polled long enough to see the real reading.
    reading = source.latest()
    hw = adapter(reading)
    assert hw.memory_util is not None
    assert abs(hw.memory_util - 0.5) < 1e-6


def test_real_jetson_telemetry_warmup_timeout_warns(monkeypatch):
    """P9 fix: if the daemon never produces a reading, the wrapper warns
    rather than blocking forever."""
    import tetrarl.eval.runner as runner_mod
    import tetrarl.sys.tegra_daemon as daemon_mod
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name, *a, **kw: "/fake/tegrastats" if name == "tegrastats" else None)

    class _DeadDaemon:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def latest(self):
            return None

    monkeypatch.setattr(daemon_mod, "TegrastatsDaemon", _DeadDaemon)

    # Patch the timeout to keep the test fast. The class likely has a constant
    # like _WARMUP_TIMEOUT_S; if it doesn't, monkeypatch the time.sleep call
    # in the runner module and have it advance a fake clock — whichever is
    # cleanest. Easiest: temporarily monkeypatch the timeout class constant
    # if you put it on the class.
    monkeypatch.setattr(runner_mod._RealJetsonTelemetry, "_WARMUP_TIMEOUT_S", 0.05, raising=False)
    monkeypatch.setattr(runner_mod._RealJetsonTelemetry, "_WARMUP_POLL_S", 0.01, raising=False)

    import warnings
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        source, _adapter = runner_mod._make_telemetry("orin_nano", use_real_telemetry=True)
    msgs = [str(r.message) for r in record if issubclass(r.category, RuntimeWarning)]
    assert any("no reading" in m.lower() or "warmup" in m.lower() or "did not produce" in m.lower() for m in msgs), \
        f"expected a warmup/no-reading RuntimeWarning, got: {msgs}"
