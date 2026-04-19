"""Unified evaluation runner / sweep harness for TetraRL (Week 8).

Provides a single entry point that builds a :class:`TetraRLFramework`
from an :class:`EvalConfig`, runs N episodes against a Gymnasium env on
the requested platform (Mac stub or Jetson), writes per-step JSONL
records, computes summary metrics (HV, tail-p99, mean energy, etc.),
and returns a :class:`RunResult`.

The harness also supports ablations by swapping individual TetraRL
components for null/placeholder variants:

  - ``preference_plane``  -> ``_NullPreferencePlane`` (uniform omega)
  - ``resource_manager``  -> ``_NullResourceManager`` (always max idx)
  - ``rl_arbiter``        -> ``_RandomArbiter`` (uniform action)
  - ``override_layer``    -> ``_NullOverrideLayer`` (never fires)

A YAML-driven sweep mode (``run_sweep`` / ``load_sweep_yaml``) iterates
configs, writes one JSONL per run, and aggregates a CSV summary so the
analysis layer can ingest a sweep as a single artefact.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import yaml

from tetrarl.core.framework import (
    ResourceManager,
    StaticPreferencePlane,
    TetraRLFramework,
)
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)

# Non-uniform default so the real preference plane and the null variant
# (which always returns uniform omega) produce distinguishable arbiter
# behaviour in ablation studies.
DEFAULT_OMEGA: np.ndarray = np.array([0.7, 0.3], dtype=np.float32)

# -----------------------------------------------------------------------------
# Config / Result dataclasses
# -----------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Single eval run specification (one row in a sweep)."""

    env_name: str
    agent_type: str
    ablation: str
    platform: str
    n_episodes: int
    seed: int
    out_dir: Path
    extra: dict = field(default_factory=dict)
    # W9: number of parallel envs in a SyncVectorEnv. n_envs=1 keeps the
    # legacy single-env path byte-for-byte identical; n_envs > 1 routes
    # through the vector branch in EvalRunner.run().
    n_envs: int = 1

    def to_dict(self) -> dict:
        return {
            "env_name": self.env_name,
            "agent_type": self.agent_type,
            "ablation": self.ablation,
            "platform": self.platform,
            "n_episodes": int(self.n_episodes),
            "seed": int(self.seed),
            "out_dir": str(self.out_dir),
            "extra": dict(self.extra),
            "n_envs": int(self.n_envs),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        return cls(
            env_name=str(d["env_name"]),
            agent_type=str(d["agent_type"]),
            ablation=str(d["ablation"]),
            platform=str(d["platform"]),
            n_episodes=int(d["n_episodes"]),
            seed=int(d["seed"]),
            out_dir=Path(d["out_dir"]),
            extra=dict(d.get("extra") or {}),
            n_envs=int(d.get("n_envs", 1)),
        )

    def to_yaml(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


@dataclass
class RunResult:
    """Aggregated metrics + the originating config for a single run."""

    config: dict
    n_steps: int
    n_episodes: int
    hv: Optional[float]
    tail_p99_ms: float
    mean_energy_j: float
    mean_memory_util: float
    mean_reward: float
    override_fire_count: int
    oom_events: int
    wall_time_s: float

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# Telemetry stub (Mac / no-tegrastats path)
# -----------------------------------------------------------------------------


@dataclass
class _StubReading:
    latency_ema_ms: Optional[float] = None
    energy_remaining_j: Optional[float] = None
    memory_util: Optional[float] = None


class _MacStubTelemetry:
    """In-memory telemetry stub fed by the eval loop each step.

    Mirrors :class:`scripts.week6_e2e_smoke.StubTelemetrySource` so the
    same synthetic 4-D record pattern (latency / energy / memory) is
    reused without dragging the Week 6 script into the eval API surface.
    """

    def __init__(self, initial_energy_j: float = 1000.0):
        self._reading = _StubReading(
            latency_ema_ms=0.0,
            energy_remaining_j=float(initial_energy_j),
            memory_util=0.1,
        )

    def update(
        self,
        latency_ms: float,
        energy_remaining_j: float,
        memory_util: float,
    ) -> None:
        self._reading = _StubReading(
            latency_ema_ms=float(latency_ms),
            energy_remaining_j=float(energy_remaining_j),
            memory_util=float(memory_util),
        )

    def latest(self) -> _StubReading:
        return self._reading


def _telemetry_to_hw(reading: _StubReading) -> HardwareTelemetry:
    return HardwareTelemetry(
        latency_ema_ms=reading.latency_ema_ms,
        energy_remaining_j=reading.energy_remaining_j,
        memory_util=reading.memory_util,
    )


# -----------------------------------------------------------------------------
# Null / placeholder components for ablations
# -----------------------------------------------------------------------------


class _NullPreferencePlane:
    """Uniform omega regardless of step. Sums to 1.0."""

    def __init__(self, n_objectives: int = 2):
        self._n = int(n_objectives)

    def get(self) -> np.ndarray:
        return np.full(self._n, 1.0 / self._n, dtype=np.float32)


class _NullResourceManager:
    """Always selects the highest DVFS index (no throttling)."""

    def decide_dvfs(self, telemetry: HardwareTelemetry, n_levels: int) -> int:
        if n_levels <= 0:
            raise ValueError("n_levels must be positive")
        return int(n_levels - 1)


class _NullOverrideLayer:
    """Never fires; provides the same surface as ``OverrideLayer``."""

    def __init__(self) -> None:
        self.fire_count = 0
        self.last_reasons: list[str] = []

    def step(self, telemetry: HardwareTelemetry) -> tuple[bool, Any | None]:
        return False, None

    def reset(self) -> None:
        self.fire_count = 0
        self.last_reasons = []


# -----------------------------------------------------------------------------
# Arbiters
# -----------------------------------------------------------------------------


class _RandomArbiter:
    """Uniform-random discrete-action arbiter (no training)."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = int(n_actions)
        self._rng = random.Random(int(seed))

    def act(self, state: Any, omega: np.ndarray) -> int:
        return self._rng.randint(0, self.n_actions - 1)


class _FixedActionArbiter:
    """Always emits the same discrete action (debug / sanity baseline)."""

    def __init__(self, action: int = 0):
        self.action = int(action)

    def act(self, state: Any, omega: np.ndarray) -> int:
        return self.action


class _PreferencePPOArbiter:
    """Placeholder for the trained Preference-PPO arbiter.

    On Mac we lack the trained checkpoint so this falls back to a
    seeded numpy RNG over the discrete action space. The class name is
    distinct from :class:`_RandomArbiter` so the rl_arbiter ablation
    test (``test_ablation_rl_arbiter_substitutes_random``) can tell
    them apart.
    """

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = int(n_actions)
        self._rng = np.random.default_rng(int(seed))

    def act(self, state: Any, omega: np.ndarray) -> int:
        # Stochastic policy biased by omega: P(a=0) = omega[0] / sum(omega).
        # Deterministic given (seed, omega, state) since rng is seeded once.
        p_zero = float(omega[0]) / float(np.sum(omega) + 1e-12)
        return 0 if self._rng.random() < p_zero else (self.n_actions - 1)


# -----------------------------------------------------------------------------
# Component factories (one per TetraRL component, ablation-aware)
# -----------------------------------------------------------------------------


def _make_preference_plane(ablation: str, omega: list | None = None):
    """Build the preference plane for a single eval run.

    When ``omega`` is provided (typically from ``EvalConfig.extra["omega"]``),
    a :class:`StaticPreferencePlane` is built from that vector regardless of
    its dimensionality (2-D legacy, 4-D Week 10 matrix, etc.). When ``omega``
    is ``None`` the legacy 2-D ``DEFAULT_OMEGA`` path is preserved so existing
    runs behave bit-identically.
    """
    if omega is not None:
        return StaticPreferencePlane(np.asarray(omega, dtype=np.float32))
    if ablation == "preference_plane":
        return _NullPreferencePlane(n_objectives=2)
    return StaticPreferencePlane(DEFAULT_OMEGA.copy())


def _make_resource_manager(ablation: str):
    if ablation == "resource_manager":
        return _NullResourceManager()
    return ResourceManager()


def _make_rl_arbiter(agent_type: str, ablation: str, n_actions: int, seed: int):
    if ablation == "rl_arbiter":
        return _RandomArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "random":
        return _RandomArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "fixed":
        return _FixedActionArbiter(action=0)
    if agent_type == "preference_ppo":
        return _PreferencePPOArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "dvfs_drl_multitask":
        from tetrarl.morl.baselines.dvfs_drl_multitask import DVFSDRLMultitaskArbiter
        return DVFSDRLMultitaskArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "envelope_morl":
        from tetrarl.morl.baselines.envelope_morl import EnvelopeMORLArbiter
        return EnvelopeMORLArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "ppo_lagrangian":
        from tetrarl.morl.baselines.ppo_lagrangian_arbiter import PPOLagrangianArbiter
        return PPOLagrangianArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "focops":
        from tetrarl.morl.baselines.focops import FOCOPSArbiter
        return FOCOPSArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "duojoule":
        from tetrarl.morl.baselines.duojoule import DuoJouleArbiter
        return DuoJouleArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "max_a":
        from tetrarl.morl.baselines.max_action import MaxActionArbiter
        return MaxActionArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "max_p":
        from tetrarl.morl.baselines.max_performance import MaxPerformanceArbiter
        return MaxPerformanceArbiter(n_actions=n_actions, seed=seed)
    if agent_type == "pcn":
        from tetrarl.morl.baselines.pcn import PCNArbiter
        return PCNArbiter(n_actions=n_actions, seed=seed)
    # Unknown agent_type -> safe fallback
    return _RandomArbiter(n_actions=n_actions, seed=seed)


def _make_override_layer(ablation: str, fallback_action: int = 0):
    if ablation == "override_layer":
        return _NullOverrideLayer()
    return OverrideLayer(
        thresholds=OverrideThresholds(
            max_latency_ms=2.0,       # tight: fires when env+framework step is slow
            min_energy_j=0.5,
            max_memory_util=0.13,     # synthetic memory ramps 0.1 + 0.001*step;
                                      # fires from step 31 onward
        ),
        fallback_action=fallback_action,
        cooldown_steps=0,
    )


def _make_telemetry(platform: str) -> tuple[Any, Callable[[Any], HardwareTelemetry]]:
    """Return ``(telemetry_source, telemetry_adapter)`` for the platform.

    Only the Mac stub path is exercised by the unit tests; Jetson
    builds out a real :class:`TegrastatsDaemon`-backed source in the
    physical eval scripts.

    Note: when ``platform`` starts with ``"orin_"`` we still return the
    Mac stub here (the real tegrastats daemon lives in the
    platform-specific scripts under ``scripts/`` because it requires the
    ``tegrastats`` binary present only on Jetson). A
    :class:`RuntimeWarning` is emitted so the eval harness does not
    silently lie about its telemetry source — this closes the W8 hidden
    bug where Orin runs were quietly using synthetic data.
    """
    if platform.startswith("orin_"):
        warnings.warn(
            f"_make_telemetry() called with platform={platform!r} but the "
            "real tegrastats daemon is not wired up here; falling back to "
            "the Mac stub. Real-Orin runs should use the platform-specific "
            "scripts that build a TegrastatsDaemon directly.",
            RuntimeWarning,
            stacklevel=2,
        )
    if platform == "mac_stub":
        return _MacStubTelemetry(initial_energy_j=1000.0), _telemetry_to_hw
    # Default to the stub for any unrecognised platform — the physical
    # platforms live behind sudo / sysfs and are wired up by the
    # platform-specific scripts, not by the unit-test path.
    return _MacStubTelemetry(initial_energy_j=1000.0), _telemetry_to_hw


# -----------------------------------------------------------------------------
# YAML sweep loader
# -----------------------------------------------------------------------------


def load_sweep_yaml(path: Path) -> list[EvalConfig]:
    """Load a sweep YAML into a list of :class:`EvalConfig`."""
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not doc or "configs" not in doc:
        return []
    out: list[EvalConfig] = []
    for entry in doc["configs"]:
        out.append(EvalConfig.from_dict(entry))
    return out


# -----------------------------------------------------------------------------
# EvalRunner
# -----------------------------------------------------------------------------


def _aggregate_metrics(
    latencies: list[float],
    energies: list[float],
    memory_utils: list[float],
    rewards: list[float],
) -> tuple[float, float, float, float]:
    """Aggregate per-step samples into (tail_p99_ms, mean_energy_j,
    mean_memory_util, mean_reward). Empty lists collapse to 0.0.

    Shared by both single-env and vector-env paths in :class:`EvalRunner`.
    """
    tail_p99_ms = float(np.percentile(latencies, 99)) if latencies else 0.0
    mean_energy_j = float(np.mean(energies)) if energies else 0.0
    mean_memory_util = float(np.mean(memory_utils)) if memory_utils else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    return tail_p99_ms, mean_energy_j, mean_memory_util, mean_reward


class EvalRunner:
    """Executes one or more :class:`EvalConfig` runs against a Gym env."""

    def _build_framework(self, cfg: EvalConfig, n_actions: int = 2) -> TetraRLFramework:
        """Wire a TetraRLFramework from ``cfg`` (no env interaction).

        ``n_actions`` defaults to 2 (CartPole's discrete count) so unit
        tests can build the framework without instantiating Gymnasium;
        :meth:`run` rebuilds with the real ``env.action_space.n``.
        """
        omega_extra = cfg.extra.get("omega") if cfg.extra else None
        pref = _make_preference_plane(cfg.ablation, omega=omega_extra)
        rm = _make_resource_manager(cfg.ablation)
        arbiter = _make_rl_arbiter(cfg.agent_type, cfg.ablation, n_actions, cfg.seed)
        override = _make_override_layer(cfg.ablation, fallback_action=0)
        telemetry_source, telemetry_adapter = _make_telemetry(cfg.platform)
        # Mac stub path: no DVFSController (writes would target unsupported
        # sysfs nodes). Jetson path will inject one in a separate runner.
        return TetraRLFramework(
            preference_plane=pref,
            rl_arbiter=arbiter,
            resource_manager=rm,
            override_layer=override,
            telemetry_source=telemetry_source,
            telemetry_adapter=telemetry_adapter,
            dvfs_controller=None,
        )

    def run(self, cfg: EvalConfig) -> RunResult:
        """Execute ``cfg.n_episodes`` and return aggregated metrics.

        For ``cfg.n_envs == 1`` (the default), uses the original single-env
        path with ``gym.make(cfg.env_name)``. For ``cfg.n_envs > 1``, routes
        through :meth:`_run_vec_env` which wraps a ``gym.vector.SyncVectorEnv``
        and emits one JSONL line per ``(env_id, episode, step)`` tuple.
        """
        import gymnasium as gym  # lazy: keeps test collection cheap

        # Reset RNGs INSIDE run() so two separate EvalRunner instances
        # produce identical reward sequences for the same seed.
        np.random.seed(int(cfg.seed))
        random.seed(int(cfg.seed))

        if int(cfg.n_envs) > 1:
            return self._run_vec_env(cfg)

        env = gym.make(cfg.env_name)
        # Wrap multi-objective DAG env so the scalar-reward eval loop can
        # consume its 4-vector reward via omega @ r_vec. Defaults to a
        # uniform 4-D omega when none is provided in cfg.extra.
        if cfg.env_name.startswith("dag_scheduler_mo"):
            from tetrarl.envs.wrappers import MOAggregateWrapper
            omega_vec = cfg.extra.get("omega") if cfg.extra else None
            if omega_vec is None:
                omega_vec = [0.25, 0.25, 0.25, 0.25]
            env = MOAggregateWrapper(env, omega=np.asarray(omega_vec, dtype=np.float32))
        try:
            n_actions = int(env.action_space.n)  # type: ignore[attr-defined]
        except AttributeError:
            n_actions = 2

        framework = self._build_framework(cfg, n_actions=n_actions)
        # Hand the real telemetry stub back so we can update() it per step.
        telemetry: _MacStubTelemetry = framework.telemetry_source

        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_name = cfg.extra.get("jsonl_name") or (
            f"{cfg.ablation}__{cfg.agent_type}__seed{cfg.seed}.jsonl"
        )
        out_path = out_dir / jsonl_name

        latencies: list[float] = []
        energies: list[float] = []
        memory_utils: list[float] = []
        rewards: list[float] = []
        oom_events = 0
        n_steps = 0
        energy_remaining = 1000.0

        t0 = time.perf_counter()
        with open(out_path, "w", encoding="utf-8") as out_file:
            try:
                for ep in range(int(cfg.n_episodes)):
                    obs, _info = env.reset(seed=int(cfg.seed) + ep)
                    episode_step = 0
                    done = False
                    while not done:
                        memory_util = 0.1 + 0.001 * episode_step

                        t_fw0 = time.perf_counter()
                        record = framework.step(obs)
                        t_fw1 = time.perf_counter()
                        fw_dt_ms = (t_fw1 - t_fw0) * 1000.0

                        action = int(record["action"])

                        t_env0 = time.perf_counter()
                        obs, reward, terminated, truncated, _info = env.step(action)
                        t_env1 = time.perf_counter()
                        env_dt_ms = (t_env1 - t_env0) * 1000.0

                        latency_ms = env_dt_ms + fw_dt_ms
                        energy_j = 1e-3 * (action + 1)
                        energy_remaining = max(0.0, energy_remaining - energy_j)

                        record["latency_ms"] = float(latency_ms)
                        record["energy_j"] = float(energy_j)
                        record["memory_util"] = float(memory_util)
                        framework.observe_reward(float(reward))

                        telemetry.update(
                            latency_ms=latency_ms,
                            energy_remaining_j=energy_remaining,
                            memory_util=memory_util,
                        )

                        if memory_util >= 1.0:
                            oom_events += 1

                        line = {
                            "episode": int(ep),
                            "step": int(episode_step),
                            "action": int(action),
                            "reward": float(reward),
                            "latency_ms": float(latency_ms),
                            "energy_j": float(energy_j),
                            "memory_util": float(memory_util),
                            "omega": [float(x) for x in record["omega"]],
                        }
                        out_file.write(json.dumps(line) + "\n")

                        latencies.append(float(latency_ms))
                        energies.append(float(energy_j))
                        memory_utils.append(float(memory_util))
                        rewards.append(float(reward))
                        episode_step += 1
                        n_steps += 1
                        done = bool(terminated or truncated)
            finally:
                env.close()
        wall_time_s = time.perf_counter() - t0

        tail_p99_ms, mean_energy_j, mean_memory_util, mean_reward = _aggregate_metrics(
            latencies, energies, memory_utils, rewards
        )
        override_fire_count = int(getattr(framework.override_layer, "fire_count", 0))

        return RunResult(
            config=cfg.to_dict(),
            n_steps=int(n_steps),
            n_episodes=int(cfg.n_episodes),
            hv=None,
            tail_p99_ms=tail_p99_ms,
            mean_energy_j=mean_energy_j,
            mean_memory_util=mean_memory_util,
            mean_reward=mean_reward,
            override_fire_count=override_fire_count,
            oom_events=int(oom_events),
            wall_time_s=float(wall_time_s),
        )

    def _run_vec_env(self, cfg: EvalConfig) -> RunResult:
        """W9: vector-env rollout for ``cfg.n_envs > 1``.

        Builds a :class:`gymnasium.vector.SyncVectorEnv` of size ``cfg.n_envs``,
        steps all envs in lockstep (one ``framework.step()`` per env per
        rollout step), and writes one JSONL line per ``(env_id, episode,
        step)`` tuple including an integer ``env_id`` field. ``cfg.n_episodes``
        becomes the per-env episode cap, so the returned ``RunResult.n_episodes``
        reports ``cfg.n_envs * cfg.n_episodes``.
        """
        import gymnasium as gym  # lazy: keeps test collection cheap

        n_envs = int(cfg.n_envs)
        n_episodes_per_env = int(cfg.n_episodes)
        env_name = cfg.env_name

        # Resolve per-env wrapping for MO envs once, outside the factory.
        wrap_mo = env_name.startswith("dag_scheduler_mo")
        omega_vec = cfg.extra.get("omega") if cfg.extra else None
        if wrap_mo and omega_vec is None:
            omega_vec = [0.25, 0.25, 0.25, 0.25]

        def _make_one(env_name: str = env_name):
            e = gym.make(env_name)
            if wrap_mo:
                from tetrarl.envs.wrappers import MOAggregateWrapper
                e = MOAggregateWrapper(e, omega=np.asarray(omega_vec, dtype=np.float32))
            return e

        vec_env = gym.vector.SyncVectorEnv([_make_one for _ in range(n_envs)])
        try:
            n_actions = int(vec_env.single_action_space.n)  # type: ignore[attr-defined]
        except AttributeError:
            n_actions = 2

        framework = self._build_framework(cfg, n_actions=n_actions)
        telemetry: _MacStubTelemetry = framework.telemetry_source

        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_name = cfg.extra.get("jsonl_name") or (
            f"{cfg.ablation}__{cfg.agent_type}__seed{cfg.seed}__nenvs{n_envs}.jsonl"
        )
        out_path = out_dir / jsonl_name

        latencies: list[float] = []
        energies: list[float] = []
        memory_utils: list[float] = []
        rewards: list[float] = []
        oom_events = 0
        n_steps = 0
        energy_remaining = 1000.0

        # Per-env episode + within-episode-step counters.
        per_env_episode = [0] * n_envs
        per_env_step = [0] * n_envs
        # Mark envs that have hit their per-env episode cap so we stop writing
        # JSONL lines for them once they're done. We continue to step them in
        # the vector-env (cheap) until all are done.
        per_env_done = [False] * n_envs

        t0 = time.perf_counter()
        obs, _info = vec_env.reset(seed=int(cfg.seed))
        with open(out_path, "w", encoding="utf-8") as out_file:
            try:
                while not all(per_env_done):
                    actions = np.zeros(n_envs, dtype=np.int64)
                    fw_dts_ms = [0.0] * n_envs
                    omegas: list[list[float]] = [[]] * n_envs

                    # Per-env framework.step (one action per env per rollout step).
                    for i in range(n_envs):
                        memory_util_i = 0.1 + 0.001 * per_env_step[i]
                        # Update telemetry to mirror the single-env contract
                        # (the framework.step reads telemetry latched from
                        # the previous env-i step).
                        telemetry.update(
                            latency_ms=latencies[-1] if latencies else 0.0,
                            energy_remaining_j=energy_remaining,
                            memory_util=memory_util_i,
                        )

                        t_fw0 = time.perf_counter()
                        record = framework.step(obs[i])
                        t_fw1 = time.perf_counter()
                        fw_dts_ms[i] = (t_fw1 - t_fw0) * 1000.0
                        actions[i] = int(record["action"])
                        omegas[i] = [float(x) for x in record["omega"]]

                    # Single vector step submits all actions at once.
                    t_env0 = time.perf_counter()
                    obs, vec_rewards, vec_term, vec_trunc, _vinfo = vec_env.step(
                        actions
                    )
                    t_env1 = time.perf_counter()
                    env_dt_ms_total = (t_env1 - t_env0) * 1000.0
                    # Spread the vector-step latency evenly across envs (a
                    # SyncVectorEnv steps them sequentially, so per-env
                    # contribution is amortised).
                    env_dt_ms_per = env_dt_ms_total / max(1, n_envs)

                    for i in range(n_envs):
                        if per_env_done[i]:
                            # Still increment counters in case the vector-env
                            # auto-resets, but skip JSONL emission.
                            done_i = bool(vec_term[i] or vec_trunc[i])
                            if done_i:
                                per_env_step[i] = 0
                            else:
                                per_env_step[i] += 1
                            continue

                        memory_util_i = 0.1 + 0.001 * per_env_step[i]
                        action_i = int(actions[i])
                        reward_i = float(vec_rewards[i])
                        latency_ms_i = env_dt_ms_per + fw_dts_ms[i]
                        energy_j_i = 1e-3 * (action_i + 1)
                        energy_remaining = max(0.0, energy_remaining - energy_j_i)

                        framework.observe_reward(reward_i)

                        if memory_util_i >= 1.0:
                            oom_events += 1

                        line = {
                            "env_id": int(i),
                            "episode": int(per_env_episode[i]),
                            "step": int(per_env_step[i]),
                            "action": int(action_i),
                            "reward": float(reward_i),
                            "latency_ms": float(latency_ms_i),
                            "energy_j": float(energy_j_i),
                            "memory_util": float(memory_util_i),
                            "omega": omegas[i],
                        }
                        out_file.write(json.dumps(line) + "\n")

                        latencies.append(float(latency_ms_i))
                        energies.append(float(energy_j_i))
                        memory_utils.append(float(memory_util_i))
                        rewards.append(float(reward_i))
                        n_steps += 1

                        done_i = bool(vec_term[i] or vec_trunc[i])
                        if done_i:
                            per_env_episode[i] += 1
                            per_env_step[i] = 0
                            if per_env_episode[i] >= n_episodes_per_env:
                                per_env_done[i] = True
                        else:
                            per_env_step[i] += 1
            finally:
                vec_env.close()
        wall_time_s = time.perf_counter() - t0

        tail_p99_ms, mean_energy_j, mean_memory_util, mean_reward = _aggregate_metrics(
            latencies, energies, memory_utils, rewards
        )
        override_fire_count = int(getattr(framework.override_layer, "fire_count", 0))

        return RunResult(
            config=cfg.to_dict(),
            n_steps=int(n_steps),
            n_episodes=int(n_envs * n_episodes_per_env),
            hv=None,
            tail_p99_ms=tail_p99_ms,
            mean_energy_j=mean_energy_j,
            mean_memory_util=mean_memory_util,
            mean_reward=mean_reward,
            override_fire_count=override_fire_count,
            oom_events=int(oom_events),
            wall_time_s=float(wall_time_s),
        )

    def run_sweep(self, configs: list[EvalConfig]) -> list[RunResult]:
        """Run a list of configs sequentially; aggregate to summary CSV."""
        if not configs:
            return []

        results: list[RunResult] = []
        for cfg in configs:
            results.append(self.run(cfg))

        common_dir = Path(configs[0].out_dir)
        common_dir.mkdir(parents=True, exist_ok=True)
        csv_path = common_dir / "summary.csv"
        cols = [
            "env_name",
            "agent_type",
            "ablation",
            "platform",
            "seed",
            "n_episodes",
            "n_steps",
            "mean_reward",
            "override_fire_count",
            "tail_p99_ms",
            "mean_energy_j",
            "mean_memory_util",
            "wall_time_s",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for r in results:
                c = r.config
                writer.writerow(
                    [
                        c.get("env_name", ""),
                        c.get("agent_type", ""),
                        c.get("ablation", ""),
                        c.get("platform", ""),
                        c.get("seed", ""),
                        c.get("n_episodes", ""),
                        r.n_steps,
                        f"{r.mean_reward:.6f}",
                        r.override_fire_count,
                        f"{r.tail_p99_ms:.6f}",
                        f"{r.mean_energy_j:.6f}",
                        f"{r.mean_memory_util:.6f}",
                        f"{r.wall_time_s:.6f}",
                    ]
                )
        return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tetrarl.eval.runner", description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=None,
        help="Path to sweep YAML config",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override out_dir for all configs",
    )
    args = parser.parse_args(argv)
    if args.config is None:
        parser.print_help()
        return 0
    configs = load_sweep_yaml(Path(args.config))
    if args.out_dir:
        for c in configs:
            c.out_dir = Path(args.out_dir)
    runner = EvalRunner()
    results = runner.run_sweep(configs)
    print(f"Completed {len(results)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
