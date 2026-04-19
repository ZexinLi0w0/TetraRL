"""Week 4 DST smoke test.

Builds a TetraRLNativeAgent on Deep Sea Treasure with the Week 4
masking and hardware-override paths active (but neutral), trains for a
small budget, and asserts:

  * training produces a non-trivial Pareto front (HV > 0)
  * the override layer is wired but never fires under benign telemetry

DST has no deadline structure, so we use NoOpMask -- this exercises the
masking integration plumbing (rollout buffer, logit rewrite, deterministic
eval path) without changing the policy distribution. The override
threshold is set 1e6 ms so it can never fire; we still pass a telemetry_fn
to confirm the call path is live.

Exits 0 on success, 1 on failure. Run as::

    python3 scripts/week4_dst_smoke.py
"""

from __future__ import annotations

import sys

from tetrarl.morl.native.agent import TetraRLNativeAgent
from tetrarl.morl.native.masking import NoOpMask
from tetrarl.morl.native.override import (
    HardwareTelemetry,
    OverrideThresholds,
)


def main() -> int:
    agent = TetraRLNativeAgent(
        env_name="dst",
        obj_num=2,
        ref_point=[0.0, -25.0],
        total_timesteps=10_000,
        num_steps=128,
        hidden_dim=32,
        eval_interval=10,
        eval_episodes=2,
        n_eval_interior=5,
        seed=42,
        device="cpu",
        use_masking=True,
        action_mask=NoOpMask(),
        use_override=True,
        override_thresholds=OverrideThresholds(max_latency_ms=1e6),
        override_fallback=0,
        telemetry_fn=lambda: HardwareTelemetry(latency_ema_ms=1.0),
    )

    results = agent.train(verbose=False)
    front = agent.get_pareto_front()

    hv = float(front["hv"])
    pf_size = int(len(front["objectives"]))
    fires = int(results["override_fire_count"])

    print(f"final HV          : {hv:.3f}")
    print(f"|PF|              : {pf_size}")
    print(f"override_fire_cnt : {fires}")

    ok = hv > 0.0 and fires == 0
    if ok:
        print("SMOKE TEST PASSED")
        return 0
    print("SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
