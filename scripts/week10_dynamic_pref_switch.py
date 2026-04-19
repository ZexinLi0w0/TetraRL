"""Week 10 Task 4: dynamic preference-switching demo.

Drives a :class:`tetrarl.core.framework.TetraRLFramework` over a Gym env
where the preference plane's omega vector is *flipped* mid-run at a
configurable switch-episode. Records per-episode reward + omega, plots
a 2-panel figure, and writes a markdown summary that validates the
"smooth transition (< 10 episode adjustment) without reward collapse"
acceptance criterion from the W10 spec.

The preference plane normally emits a static omega (see
:class:`tetrarl.core.framework.StaticPreferencePlane`); to support a
runtime switch without touching the framework, we wrap it locally in a
``MutableStaticPreferencePlane`` whose ``set(omega)`` swaps the vector
in place. The framework reads ``preference_plane.get()`` once per step
so the next episode picks up the new vector immediately.

Outputs in ``--out-dir``:

  - dynamic_pref_switch.csv          (one row per episode)
  - dynamic_pref_switch.png/.svg     (top: reward; bottom: 4 omega lines)
  - dynamic_pref_switch_summary.md   (pre/post-switch means + collapse flag)

Example::

    python3 scripts/week10_dynamic_pref_switch.py \\
        --out-dir runs/w10_orin_full \\
        --n-episodes 100 \\
        --switch-episode 50 \\
        --seed 0 \\
        --agent envelope_morl
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make the repo root importable so ``python scripts/week10_*`` works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tetrarl.core.framework import (  # noqa: E402
    ResourceManager,
    TetraRLFramework,
)
from tetrarl.eval.runner import (  # noqa: E402
    _MacStubTelemetry,
    _make_rl_arbiter,
    _telemetry_to_hw,
)
from tetrarl.morl.native.override import (  # noqa: E402
    OverrideLayer,
    OverrideThresholds,
)


class MutableStaticPreferencePlane:
    """Static preference plane whose omega can be swapped at runtime.

    Mirrors :class:`tetrarl.core.framework.StaticPreferencePlane` but
    exposes a :meth:`set` mutator so the dynamic-switch driver can flip
    omega between episodes without rebuilding the framework.
    """

    def __init__(self, omega: np.ndarray):
        self._omega = np.asarray(omega, dtype=np.float32).copy()

    def get(self) -> np.ndarray:
        return self._omega.copy()

    def set(self, omega: np.ndarray) -> None:
        self._omega = np.asarray(omega, dtype=np.float32).copy()


def _parse_omega_csv(s: str) -> np.ndarray:
    """Parse a CSV omega string like '0.5,0.5,0.0,0.0' into a 4-D ndarray."""
    parts = [float(tok.strip()) for tok in s.split(",") if tok.strip()]
    return np.asarray(parts, dtype=np.float32)


def _build_framework(
    *,
    plane: MutableStaticPreferencePlane,
    agent: str,
    n_actions: int,
    seed: int,
) -> TetraRLFramework:
    """Wire a TetraRLFramework around the mutable preference plane."""
    arbiter = _make_rl_arbiter(
        agent_type=agent, ablation="none", n_actions=n_actions, seed=seed,
    )
    rm = ResourceManager()
    override = OverrideLayer(
        thresholds=OverrideThresholds(
            max_latency_ms=2.0,
            min_energy_j=0.5,
            max_memory_util=0.13,
        ),
        fallback_action=0,
        cooldown_steps=0,
    )
    telem = _MacStubTelemetry(initial_energy_j=1000.0)
    return TetraRLFramework(
        preference_plane=plane,
        rl_arbiter=arbiter,
        resource_manager=rm,
        override_layer=override,
        telemetry_source=telem,
        telemetry_adapter=_telemetry_to_hw,
        dvfs_controller=None,
    )


def _run_episodes(
    *,
    env_name: str,
    plane: MutableStaticPreferencePlane,
    framework: TetraRLFramework,
    n_episodes: int,
    switch_episode: int,
    pre_omega: np.ndarray,
    post_omega: np.ndarray,
    seed: int,
) -> tuple[list[float], list[np.ndarray]]:
    """Execute ``n_episodes`` and return (per-ep reward, per-ep omega)."""
    import gymnasium as gym  # lazy: keeps test collection cheap

    rewards: list[float] = []
    omegas: list[np.ndarray] = []
    telemetry: _MacStubTelemetry = framework.telemetry_source
    energy_remaining = 1000.0

    env = gym.make(env_name)
    try:
        for ep in range(int(n_episodes)):
            if ep == int(switch_episode):
                plane.set(post_omega)

            obs, _info = env.reset(seed=int(seed) + ep)
            ep_reward = 0.0
            ep_omega = plane.get()
            done = False
            episode_step = 0
            while not done:
                memory_util = 0.1 + 0.001 * episode_step
                record = framework.step(obs)
                action = int(record["action"])
                obs, reward, terminated, truncated, _info = env.step(action)
                framework.observe_reward(float(reward))
                ep_reward += float(reward)
                energy_j = 1e-3 * (action + 1)
                energy_remaining = max(0.0, energy_remaining - energy_j)
                telemetry.update(
                    latency_ms=0.0,
                    energy_remaining_j=energy_remaining,
                    memory_util=memory_util,
                )
                episode_step += 1
                done = bool(terminated or truncated)

            rewards.append(float(ep_reward))
            omegas.append(np.asarray(ep_omega, dtype=np.float32))
    finally:
        env.close()
    return rewards, omegas


def _write_csv(
    out_path: Path,
    rewards: list[float],
    omegas: list[np.ndarray],
) -> None:
    """Write per-episode CSV: episode, reward, omega_0..omega_3."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "omega_0", "omega_1", "omega_2", "omega_3"])
        for i, (r, w) in enumerate(zip(rewards, omegas)):
            w4 = list(w[:4]) + [0.0] * max(0, 4 - len(w))
            writer.writerow([i, f"{r:.6f}"] + [f"{float(x):.6f}" for x in w4[:4]])


def _compute_window_means(
    rewards: list[float],
    switch_episode: int,
    window: int = 10,
    adjustment: int = 10,
) -> tuple[float, float, int, int]:
    """Return (pre_mean, post_mean, pre_n, post_n) using <=window episodes.

    Pre-switch window: the ``window`` episodes immediately before the
    switch (clipped to [0, switch_episode)).
    Post-switch window: the ``window`` episodes starting ``adjustment``
    episodes after the switch (clipped to total length).
    """
    n = len(rewards)
    sw = int(switch_episode)
    pre_lo = max(0, sw - window)
    pre_hi = max(0, min(sw, n))
    post_lo = max(0, min(sw + adjustment, n))
    post_hi = min(n, post_lo + window)
    pre = rewards[pre_lo:pre_hi]
    post = rewards[post_lo:post_hi]
    pre_mean = float(np.mean(pre)) if pre else 0.0
    post_mean = float(np.mean(post)) if post else 0.0
    return pre_mean, post_mean, len(pre), len(post)


def _write_summary_md(
    out_path: Path,
    *,
    n_episodes: int,
    switch_episode: int,
    pre_omega: np.ndarray,
    post_omega: np.ndarray,
    pre_mean: float,
    post_mean: float,
    pre_n: int,
    post_n: int,
    collapse: bool,
    adjustment: int,
) -> None:
    """Write the dynamic-switch acceptance summary as markdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pre_str = ",".join(f"{float(x):.3f}" for x in pre_omega[:4])
    post_str = ",".join(f"{float(x):.3f}" for x in post_omega[:4])
    body = (
        f"# W10 dynamic preference-switching summary\n\n"
        f"- n_episodes: {n_episodes}\n"
        f"- switch_episode: {switch_episode}\n"
        f"- pre-switch omega: [{pre_str}]\n"
        f"- post-switch omega: [{post_str}]\n"
        f"- adjustment window (skipped after switch): {adjustment} episodes\n\n"
        f"## Reward windows\n\n"
        f"- pre-switch mean reward (last {pre_n} eps before switch): "
        f"{pre_mean:.4f}\n"
        f"- post-switch mean reward (eps "
        f"{switch_episode + adjustment}..{switch_episode + adjustment + post_n - 1}): "
        f"{post_mean:.4f}\n\n"
        f"## Acceptance\n\n"
        f"- collapse criterion: post < 0.5 * pre -> "
        f"{0.5 * pre_mean:.4f}\n"
        f"- reward_collapse: {str(bool(collapse)).lower()}\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)


def _plot(
    out_dir: Path,
    rewards: list[float],
    omegas: list[np.ndarray],
    switch_episode: int,
) -> None:
    """Write the 2-panel PNG + SVG figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = list(range(len(rewards)))
    omega_arr = np.zeros((len(omegas), 4), dtype=np.float32)
    for i, w in enumerate(omegas):
        for j in range(min(4, len(w))):
            omega_arr[i, j] = float(w[j])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    ax_r, ax_w = axes

    ax_r.plot(eps, rewards, color="tab:blue", label="reward")
    ax_r.axvline(int(switch_episode), color="k", linestyle="--",
                 label=f"switch @ ep {switch_episode}")
    ax_r.set_ylabel("episode reward")
    ax_r.set_title("Reward over episodes")
    ax_r.legend(loc="best", fontsize=8)
    ax_r.grid(True, alpha=0.3)

    colors = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
    for j in range(4):
        ax_w.plot(eps, omega_arr[:, j], color=colors[j], label=f"omega_{j}")
    ax_w.axvline(int(switch_episode), color="k", linestyle="--", alpha=0.5)
    ax_w.set_xlabel("episode")
    ax_w.set_ylabel("omega component")
    ax_w.set_title("Preference plane omega over episodes")
    ax_w.legend(loc="best", fontsize=8, ncol=4)
    ax_w.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "dynamic_pref_switch.png", dpi=150)
    fig.savefig(out_dir / "dynamic_pref_switch.svg")
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Directory to write png/svg/md/csv into.")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Total episodes to run.")
    parser.add_argument("--switch-episode", type=int, default=50,
                        help="Episode index at which omega flips.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed shared by env reset and arbiter.")
    parser.add_argument("--agent", type=str, default="envelope_morl",
                        help="Arbiter agent_type (e.g. envelope_morl, "
                             "preference_ppo).")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gymnasium env id.")
    parser.add_argument("--pre-omega", type=str, default="0.5,0.5,0.0,0.0",
                        help="CSV pre-switch 4-D omega.")
    parser.add_argument("--post-omega", type=str, default="0.0,0.0,0.2,0.8",
                        help="CSV post-switch 4-D omega.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_omega = _parse_omega_csv(args.pre_omega)
    post_omega = _parse_omega_csv(args.post_omega)

    # Resolve action-space dimensionality without instantiating the env
    # twice (we still need it for the rollout below).
    import gymnasium as gym  # lazy
    probe = gym.make(args.env)
    try:
        n_actions = int(probe.action_space.n)  # type: ignore[attr-defined]
    except AttributeError:
        n_actions = 2
    finally:
        probe.close()

    plane = MutableStaticPreferencePlane(pre_omega)
    framework = _build_framework(
        plane=plane, agent=args.agent, n_actions=n_actions, seed=int(args.seed),
    )

    rewards, omegas = _run_episodes(
        env_name=str(args.env),
        plane=plane,
        framework=framework,
        n_episodes=int(args.n_episodes),
        switch_episode=int(args.switch_episode),
        pre_omega=pre_omega,
        post_omega=post_omega,
        seed=int(args.seed),
    )

    _write_csv(out_dir / "dynamic_pref_switch.csv", rewards, omegas)

    pre_mean, post_mean, pre_n, post_n = _compute_window_means(
        rewards, switch_episode=int(args.switch_episode),
        window=10, adjustment=10,
    )
    collapse = bool(post_mean < 0.5 * pre_mean) if pre_mean > 0 else False
    _write_summary_md(
        out_dir / "dynamic_pref_switch_summary.md",
        n_episodes=int(args.n_episodes),
        switch_episode=int(args.switch_episode),
        pre_omega=pre_omega,
        post_omega=post_omega,
        pre_mean=pre_mean,
        post_mean=post_mean,
        pre_n=pre_n,
        post_n=post_n,
        collapse=collapse,
        adjustment=10,
    )

    _plot(out_dir, rewards, omegas, switch_episode=int(args.switch_episode))

    print(
        f"wrote {len(rewards)} episodes to {out_dir}; "
        f"pre_mean={pre_mean:.3f} post_mean={post_mean:.3f} "
        f"collapse={str(collapse).lower()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
