"""Hardware-emergency override layer.

Acts as a hard safety net beneath the learned policy.  When any resource
dimension exceeds a critical threshold, the override bypasses the RL
policy and forces a conservative fallback action.  Inspired by the CPO
recovery step (Achiam et al., 2017).
"""

# TODO: Week 4 — implement threshold checks for memory (>95% util),
#       energy (budget exhausted), and deadline cascade (>3 consecutive
#       misses); unit-test override triggering logic.


class OverrideLayer:
    """Hardware-emergency override that preempts the RL policy."""

    def __init__(
        self,
        mem_util_threshold: float = 0.95,
        energy_budget_j: float = float("inf"),
        deadline_miss_cascade: int = 3,
    ):
        self.mem_util_threshold = mem_util_threshold
        self.energy_budget_j = energy_budget_j
        self.deadline_miss_cascade = deadline_miss_cascade
        self._consecutive_misses = 0

    def should_override(
        self, mem_util: float, energy_used_j: float, deadline_missed: bool
    ) -> bool:
        """Check whether any resource threshold is breached.

        @param mem_util        Current memory utilization in [0, 1].
        @param energy_used_j   Cumulative energy consumed (Joules).
        @param deadline_missed Whether the most recent step missed its deadline.
        @return                True if the override should activate.
        """
        raise NotImplementedError

    def fallback_action(self) -> dict:
        """Return a conservative fallback action configuration.

        @return  Dict with keys: batch_size, replay_buffer_size, cpu_freq, gpu_freq.
        """
        raise NotImplementedError
