"""Smoke tests for C-MORL integration.

Validates that the vendored code is structurally intact and that
CMORLAgent can be instantiated without heavyweight dependencies.
"""

import importlib
from pathlib import Path

import pytest

from tetrarl.morl.c_morl_agent import CMORLAgent, CMORLConfig, _CMORL_DIR


class TestVendoredStructure:
    """Verify that all expected vendored files are in place."""

    EXPECTED_MORL_FILES = [
        "morl.py",
        "mopg.py",
        "ep.py",
        "hypervolume.py",
        "sample.py",
        "scalarization_methods.py",
        "arguments.py",
        "run.py",
        "task.py",
        "utils.py",
        "warm_up.py",
        "__init__.py",
        "NOTICE",
    ]

    EXPECTED_DIRS = [
        "environments",
        "environments/building",
        "environments/building/data",
        "environments/assets",
        "externals",
        "externals/baselines",
        "externals/pytorch-a2c-ppo-acktr-gail",
    ]

    def test_morl_files_exist(self):
        for fname in self.EXPECTED_MORL_FILES:
            path = _CMORL_DIR / fname
            assert path.exists(), f"Missing vendored file: {fname}"

    def test_directory_structure(self):
        for dname in self.EXPECTED_DIRS:
            path = _CMORL_DIR / dname
            assert path.is_dir(), f"Missing vendored directory: {dname}"

    def test_notice_contains_attribution(self):
        notice = (_CMORL_DIR / "NOTICE").read_text()
        assert "C-MORL" in notice
        assert "ICLR 2025" in notice
        assert "Ruohong Liu" in notice

    def test_environments_init_exists(self):
        init = _CMORL_DIR / "environments" / "__init__.py"
        assert init.exists()


class TestCMORLConfig:
    """Verify CMORLConfig dataclass and conversion."""

    def test_default_config(self):
        cfg = CMORLConfig()
        assert cfg.obj_num == 2
        assert cfg.algo == "ppo"
        assert cfg.ref_point == [0.0, 0.0]

    def test_custom_config(self):
        cfg = CMORLConfig(
            env_name="building_3d",
            obj_num=3,
            ref_point=[0.0, 0.0, 0.0],
            num_time_steps=100_000,
        )
        assert cfg.env_name == "building_3d"
        assert cfg.obj_num == 3
        assert cfg.num_time_steps == 100_000

    def test_to_namespace(self):
        cfg = CMORLConfig(env_name="test_env", seed=42)
        ns = cfg.to_namespace()
        assert ns.env_name == "test_env"
        assert ns.seed == 42
        assert ns.algo == "ppo"


class TestCMORLAgent:
    """Verify CMORLAgent instantiation (no training)."""

    def test_instantiation(self):
        agent = CMORLAgent(
            env_name="MO-Hopper-v2",
            obj_num=2,
            ref_point=[0.0, 0.0],
        )
        assert agent.config.env_name == "MO-Hopper-v2"
        assert agent.config.obj_num == 2
        assert not agent._trained

    def test_custom_kwargs(self):
        agent = CMORLAgent(
            env_name="building_3d",
            obj_num=3,
            ref_point=[0.0, 0.0, 0.0],
            num_time_steps=500_000,
            num_init_steps=300_000,
            beta=0.95,
        )
        assert agent.config.num_time_steps == 500_000
        assert agent.config.beta == 0.95

    def test_vendored_path(self):
        p = CMORLAgent.vendored_path()
        assert p.is_dir()
        assert (p / "morl.py").exists()

    def test_get_pareto_front_before_training(self):
        agent = CMORLAgent(
            env_name="test",
            obj_num=2,
            ref_point=[0.0, 0.0],
            save_dir="/tmp/nonexistent_cmorl_test",
        )
        with pytest.raises(FileNotFoundError, match="No Pareto front results"):
            agent.get_pareto_front()
