"""DAG-env GCN smoke test.

Validates the graph-aware code path end-to-end: builds a
``TetraRLNativeAgent`` on the synthetic ``DAGSchedulerEnv`` with the GCN
feature extractor and the DAG ready-mask both active, trains for a small
budget, and asserts that PPO actually moves the policy off the all-zeros
baseline.

This is the integration counterpart to ``tests/test_gnn_extractor.py``
(which only exercises the GCN as a standalone module). Until Stage 3 of
the Week 4 bonus, the GCN had no integration test against a real
graph-structured MO env -- this script closes that gap by:

  * driving the (node_features, edge_index, batch) path through the
    rollout buffer, the per-minibatch merged-graph PPO update, and the
    deterministic eval loop;
  * asserting HV > 0 (training learned something non-trivial); and
  * asserting |PF| > 1 (policy spans multiple trade-offs across the
    3-objective ``[throughput, -energy, -peak_memory]`` space).

Exits 0 on success, 1 on failure. Run as::

    python3 scripts/dag_env_smoke.py
"""

from __future__ import annotations

import sys
import time

from tetrarl.envs.dag_scheduler import DAGReadyMask
from tetrarl.morl.native.agent import TetraRLNativeAgent


def main() -> int:
    n_tasks = 8
    density = 0.3
    hidden_dim = 32

    agent = TetraRLNativeAgent(
        env_name="dag",
        obj_num=3,
        ref_point=[0.0, -50.0, -50.0],
        total_timesteps=30_000,
        num_steps=256,
        hidden_dim=hidden_dim,
        eval_interval=10,
        eval_episodes=2,
        n_eval_interior=5,
        seed=0,
        device="cpu",
        use_gnn=True,
        use_masking=True,
        action_mask=DAGReadyMask(),
        n_tasks=n_tasks,
        density=density,
    )

    t0 = time.time()
    results = agent.train(verbose=False)
    runtime = time.time() - t0
    front = agent.get_pareto_front()

    hv = float(front["hv"])
    pf_size = int(len(front["objectives"]))
    hv_history = results["hv_history"]

    gnn_params = sum(
        p.numel() for p in agent._gnn_extractor.parameters()
    )

    print(f"n_tasks           : {n_tasks}")
    print(f"density           : {density}")
    print(f"total timesteps   : {results['global_step']}")
    print(f"final HV          : {hv:.3f}")
    print(f"|PF|              : {pf_size}")
    print(f"runtime           : {runtime:.2f} s")
    print(f"GNN params        : {gnn_params}")
    print("hv_history (step, HV, |PF|):")
    for step, h, n in hv_history:
        print(f"  step={step:6d}  HV={h:8.3f}  |PF|={n}")

    ok = hv > 0.0 and pf_size > 1
    if ok:
        print("SMOKE TEST PASSED")
        return 0
    print("SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
