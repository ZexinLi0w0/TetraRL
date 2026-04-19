# TetraRL / R⁴ TC Paper — 12-Week Action Plan

**Document version**: 2026-04-18 (revised; original 2026-04-17)
**Calendar span**: Week 1 (2026-04-20) through Week 12 (ending 2026-07-12)
**Target venue**: IEEE Transactions on Computers (TC)

---

## External Dependencies & Known Blockers

| Dependency | Status | Affected Weeks | Mitigation |
|---|---|---|---|
| **DonkeyCar simulator family** (gym-donkeycar + Unity DonkeySim binary) — includes **all DonkeyCar-derived experiments** (e.g. DonkeyCar-SAC, DonkeyCar-PPO, C-MORL/DonkeyCar) | ⚠️ **Blocked** — requires external x86 PC with discrete GPU to host the Unity sim; not available now (and it is uncertain whether SAC-on-DonkeyCar specifically can also be made to run even once the PC is provisioned) | W7, W9, W10 (originally) | Substitute with **DAG-scheduler-MO env** (W4-bonus) + **PyBullet-HalfCheetah** as the closed-loop targets. Any DonkeyCar variant is added retroactively only if/when both the external PC is provisioned **and** the SAC-on-DonkeyCar pipeline is verified to run end-to-end. Not on critical path. |
| **Xavier NX hardware** | ❌ **Removed** from plan (hardware unavailable) | W7, W9, W10 (originally) | Multi-platform scope reduced to **Orin AGX + Jetson Nano** only. |
| **Jetson Nano** | ✅ Available | W7 (Track B), W9, W10 | Standard porting workflow; sysfs paths differ from Orin (allow ~1 day buffer). |
| **Orin AGX** | ✅ Available | W5–12 | Primary target platform. |

**Project conventions** (added 2026-04-18):
- All production code work goes through **Claude Code CLI with the `superpowers` plugin** enabled. Do not write production code from the OpenClaw main session directly.
- Subagent handoffs use `~/.openclaw/workspace/AGENTS_HANDOFF/active/<task-id>/spec.md`.

---

## Phase I — Infrastructure & Algorithm Baselines on Simulator (Weeks 1–4)

### Week 1 (2026-04-20 to 2026-04-26)
**Theme**: Repository scaffolding and PD-MORL baseline reproduction

**Concrete tasks**:
1. Create the TetraRL repository with the following top-level modules: `tetrarl/core/`, `tetrarl/envs/`, `tetrarl/morl/`, `tetrarl/sys/`, `tetrarl/eval/`, `tetrarl/utils/`.
2. Implement `tetrarl/morl/pd_morl.py` — the PD-MORL MO-DDQN-HER variant (discrete) following Basaklar 2023 Eq. 6. Verify on Deep Sea Treasure (DST, 2-D objectives) that the Hypervolume (HV) metric reaches ≥95% of the value reported in PD-MORL Table 1 (HV ≥ 229).
3. Implement the cosine-similarity envelope operator `Sc(ω_p, Q) · (ω^T Q)` in `tetrarl/morl/operators.py`.
4. Write unit tests for the envelope operator and preference-sampling utilities (`tests/test_operators.py`, `tests/test_preference_sampling.py`).
5. Set up CI with `pytest` and `mypy` type-checking on the repository.

**Deliverables**:
- `tetrarl/morl/pd_morl.py` — functional MO-DDQN-HER agent
- `tetrarl/morl/operators.py` — cosine-similarity envelope operator
- DST HV convergence curve (training curve, PNG) reproducible via `scripts/run_dst.sh`

**Validation criteria**:
- `pytest tests/` passes with ≥90% line coverage on `morl/` modules.
- DST 2-D HV reaches ≥229 within 200k environment steps on a single desktop GPU.

**Risk / Pitfall to watch**:
- The homotopy λ annealing schedule from Envelope MORL (Yang 2019) is tuned for same-scale game scores; if ported verbatim to heterogeneous-scale objectives later, convergence will degrade. Validate only on same-scale DST this week; scale-mismatch handling is deferred to Week 3.

---

### Week 2 (2026-04-27 to 2026-05-03)
**Theme**: C-MORL (ICLR 2025) Integration as Building Block

**Concrete tasks**:
1. Incorporate C-MORL (Constrained MORL) repository as the foundational algorithm block (`tetrarl/morl/c_morl_agent.py`), replacing the previous C-MORL / PD-MORL continuous action direction.
2. Run C-MORL on the `Building-3d` benchmark locally to validate baseline performance and understand the CPO/IPO constraint mechanism (as it strongly aligns with our constraint-based setting).
3. Refactor C-MORL's PPO+GAE implementation to cleanly interface with our `tetrarl/core/` framework.
4. Run initial sanity checks to ensure the two-stage Pareto discovery (Init Stage + Constraint Stage) operates correctly.

**Deliverables**:
- `tetrarl/morl/c_morl_agent.py` — functional C-MORL agent adapted for our repo
- Baseline reproduction curves for `Building-3d` (Init Stage vs Constraint Stage)

**Validation criteria**:
- Successfully reproduce C-MORL Pareto front on `Building-3d` within 10% of the paper's reported hypervolume.
- No NaN or divergence in the CPO/IPO constraint stages.

**Risk / Pitfall to watch**:
- C-MORL relies on older versions of `mujoco-py` and Python. Ensure dependencies are cleanly isolated or updated to be compatible with our modern stack (Python 3.10+).

---

### Week 3 (2026-05-04 to 2026-05-10)
**Theme**: 4-D Objective Vector & Mo-Gymnasium Environment Integration

**Concrete tasks**:
1. Define the 4-D objective vector `r = [performance, -energy, -memory, -latency]` in `tetrarl/core/reward.py`.
2. Wrap the Jetson DAG workload scheduling problem into a strict `mo-gymnasium` compatible environment API (`tetrarl/envs/dag_scheduler.py`), as C-MORL expects this interface.
3. Setup C-MORL constraint thresholds: map the latency deadline and memory budget directly to the CPO/IPO constraint thresholds in the second stage of C-MORL.
4. Run C-MORL on a simulated 4-D DAG scheduling environment to verify that the constraint stage correctly handles memory and latency bounds.

**Deliverables**:
- `tetrarl/envs/dag_scheduler.py` — `mo-gymnasium` compatible scheduling env
- 4-D constraint mapping logic in the C-MORL agent

**Validation criteria**:
- C-MORL constraint stage successfully identifies and optimizes policies that respect the simulated memory budget and deadline.

**Risk / Pitfall to watch**:
- C-MORL's scalarization in the Init Stage might ignore constraints initially. Ensure the Constraint Stage is triggered correctly to fix boundary violations.

---

### Week 4 (2026-05-11 to 2026-05-17)
**Theme**: Action Masking & GNN Feature Extractor

**Concrete tasks**:
1. Implement Action Masking for hard real-time constraints: at state $S_t$, mask out DVFS actions that mathematically guarantee a deadline miss. Integrate this mask into C-MORL's PPO actor network.
2. Implement the hardware-emergency override layer in `tetrarl/sys/override.py` as a fallback safety net.
3. Integrate Graph Neural Networks (GNNs) as the feature extractor for the C-MORL actor-critic networks to ensure generalization across different DAG topologies (learning topology, not just static task IDs).
4. Run end-to-end simulated validation of C-MORL + Masking + GNN.

**Deliverables**:
- `tetrarl/core/masking.py` — Action masking logic
- `tetrarl/gnn/extractor.py` — GNN feature extractor
- `tetrarl/sys/override.py` — Hardware-emergency override layer

**Validation criteria**:
- Action masking reduces early-training deadline violations by >80%.
- GNN extractor enables zero-shot transfer to unseen DAG topologies with <15% performance drop.

**Risk / Pitfall to watch**:
- Action masking changes the valid action space dynamically, which can interfere with PPO's probability ratio calculations. Ensure masked actions are correctly assigned -inf logits.

---

## Phase II — Systems Heavy-Lifting & Orin AGX Closed Loop (Weeks 5–8)

### Week 5 (2026-05-18 to 2026-05-24)
**Theme**: Tegrastats sensor daemon and DVFS control on Orin AGX

**Concrete tasks**:
1. Implement the tegrastats async sensor daemon in `tetrarl/sys/tegra_daemon.py` with 100 Hz sampling and 10 Hz dispatch to the RL agent (kernel/user split pattern borrowed from DVFS-DRL-Multitask 2024). Apply EMA filtering with α = 0.1 on raw sensor readings.
2. Implement the DVFS controller in `tetrarl/sys/dvfs.py` — enumerate available CPU/GPU/EMC frequency points on Orin AGX, expose a `set_freq(cpu_freq, gpu_freq)` API, and measure transition latencies for all frequency pairs.
3. Integrate tegrastats daemon outputs into the 4-D state augmentation: `S_sys = [s_t, L_prev, E_rem, M_util, ω]` in `tetrarl/core/state.py`.
4. Profile DVFS transition overhead (ms per switch) across all frequency levels on Orin AGX. Adopt super-block decision granularity (SparseDVFS 2025): perform DVFS adjustment only every N = 10 training steps.

**Deliverables**:
- `tetrarl/sys/tegra_daemon.py` — async 100 Hz sensor daemon with EMA filter
- `tetrarl/sys/dvfs.py` — DVFS frequency controller for Orin AGX
- DVFS transition latency table (all frequency pairs, ms per switch) — paper-figure-ready

**Validation criteria**:
- Tegrastats daemon runs for 1 hour without memory leaks or dropped samples (verified via `/proc/meminfo` delta < 1 MB).
- DVFS frequency changes are confirmed via `/sys/devices/` readback within 50 ms of the API call.

**Risk / Pitfall to watch**:
- DVFS transition overhead on Jetson platforms is non-negligible (identified in §7 limitation 1). If transition latency exceeds 5 ms, the super-block granularity N must be increased to amortize overhead. Measure first, then set N.

---

### Week 6 (2026-05-25 to 2026-05-31)
**Theme**: Four-component framework integration and replay-buffer memory management

**Concrete tasks**:
1. Implement the four-component framework shell in `tetrarl/core/framework.py`: (i) Preference Plane, (ii) Resource Manager, (iii) RL Arbiter, (iv) Hardware Override Layer. Wire the data flow: Preference Plane → RL Arbiter → Resource Manager → Override → DVFS/Memory actuators.
2. Implement pre-allocated max replay buffer with index-mask soft truncation in `tetrarl/sys/buffer.py` (avoiding physical deallocation/reallocation per §9.7 pitfall 2; reusing R³ memory layout Li 2023). Pre-allocate at initialization, use index masking for runtime "soft truncation."
3. Integrate the state cache + soft-link replay buffer from R³ (Li 2023) for stacked-frame environments.
4. Port the full C-MORL agent to run on Orin AGX with the tegrastats daemon providing real energy/memory readings. Replace simulated reward dimensions with real hardware metrics.
5. Run a first end-to-end loop: CartPole (Classic Control) with C-MORL + DVFS + tegrastats on Orin AGX. Log all 4 dimensions over 100 episodes.

**Deliverables**:
- `tetrarl/core/framework.py` — four-component framework orchestrator
- `tetrarl/sys/buffer.py` — pre-allocated soft-truncation replay buffer
- First end-to-end 4-D training curve on Orin AGX (CartPole, 100 episodes) — paper-figure-ready time-series of [Reward, Latency, Energy, Memory]

**Validation criteria**:
- No OOM on Orin AGX over 100 episodes (memory utilization stays below 90% of available unified memory).
- All 4 reward dimensions are logged with valid (non-NaN, non-zero) values at every step.
- Framework overhead (per-step wall-clock cost of the framework minus the bare RL step) is < 2 ms.

**Risk / Pitfall to watch**:
- Frequent adjustments to the memory bound cause PyTorch to generate fragmentation on Orin Unified Memory (§9.7 pitfall 2). The soft-truncation buffer must be validated to prevent this. Monitor `torch.cuda.memory_stats()` fragmentation counters.

---

### Week 7 (2026-06-01 to 2026-06-07) — REVISED 2026-04-18
**Theme**: Full closed-loop validation on Orin AGX **+** parallel Jetson Nano porting

> **Plan revision (2026-04-18)**: Per user direction, multi-platform scope is reduced to **Orin AGX + Jetson Nano** only (Xavier NX dropped — hardware not available). Nano porting tasks originally scheduled in Week 9 are pulled forward into Week 7 to run in parallel with the Orin closed-loop validation. Week 9 is correspondingly re-scoped (see Week 9 below).

**Concrete tasks**:

*Track A — Orin AGX closed-loop (primary):*
1. **⚠️ BLOCKED ON EXTERNAL PC** — ~~Run C-MORL / TetraRL-Native on DonkeyCar simulator with DVFS + tegrastats on Orin AGX for 500 episodes, sweeping 5 representative preference vectors ω (one-hot corners + center).~~ DonkeyCar simulator (gym-donkeycar + Unity DonkeySim binary) cannot run on the available Orin AGX directly; it requires a separate x86 PC with a discrete GPU to host the simulator and stream observations to Orin over the network. **Fallback for Week 7**: substitute with the existing **synthetic DAG scheduling MO env** (Week 4-bonus, `tetrarl/envs/dag_scheduler.py`) and **PyBullet-HalfCheetah** (Track A.2) as the primary closed-loop targets. DonkeyCar runs are deferred until the external PC is provisioned (tracked separately, not on critical path).
2. Run PPO-Lagrangian (on-policy) on PyBullet-HalfCheetah with the unified knob mapper (`n_steps`, `n_epochs`, `mini_batch_size` as knobs per §9.2) on Orin AGX. Validate that the Lagrangian dual variables converge and constraints are respected (with override layer as safety net).
3. Implement the "thinking-while-moving" concurrent decision trick from DVFO (Zhang TMC 2023) in `tetrarl/sys/concurrent.py` — overlap DVFS decision computation with RL forward pass to mask decision-loop overhead.
4. Generate Pareto front visualization: 2-D projections of the 4-D Pareto front (T vs. A, E vs. A, M vs. A) on Orin. **Substitute env**: DAG scheduler MO env + PyBullet-HalfCheetah (DonkeyCar deferred). Compute HV indicator.
5. Run the co-runner FFmpeg interference test (R³ protocol, Fig. 15) on Orin: 720p / 1080p / 2K background video decoding while training. Log tail-latency shifts.

*Track B — Jetson Nano porting (parallel, pulled from Week 9):*
6. Port `tetrarl/sys/tegra_daemon.py` and `tetrarl/sys/dvfs.py` to Jetson Nano. Adjust frequency tables (Nano CPU/GPU sysfs paths differ from Orin AGX) and EMA filter parameters per platform. Re-profile DVFS transition latencies on Nano and produce a transition-latency table.
7. Run a reduced-scope experiment on Jetson Nano (CartPole + Classic Control only, due to 4 GB memory constraint). Validate that the override layer correctly prevents OOM on Nano under stress (override fire_count ≥ 1 over the run).
8. Collect tail-latency CDFs on **Orin and Nano** for the same preference vector. Generate a 2-panel CDF figure (Orin / Nano) — paper Figure candidate.

**Deliverables**:
- `tetrarl/sys/concurrent.py` — decision-loop overhead masking
- 4-D Pareto front scatter plots (3 × 2-D projections) for **DAG-scheduler-MO + PyBullet-HalfCheetah** on Orin — paper-figure-ready (DonkeyCar deferred until external PC available)
- FFmpeg co-runner interference table (tail-latency 99th percentile at 720p/1080p/2K) — paper-figure-ready
- Jetson Nano DVFS transition latency table — paper Table candidate
- 2-platform tail-latency CDF figure (Orin / Nano) — paper Figure candidate
- Nano-CartPole training log proving override-driven OOM prevention

**Validation criteria**:
- Substitute env (DAG-scheduler-MO or PyBullet-HalfCheetah) achieves ≥70% of MAX-A reward while maintaining zero deadline misses under the center preference ω = [0.25, 0.25, 0.25, 0.25] on Orin. (DonkeyCar criterion deferred.)
- PPO-Lagrangian constraint violation rate is < 20% of episodes (with override layer active) on Orin.
- FFmpeg co-runner at 1080p does not increase 99th-percentile latency by more than 2× relative to the no-interference case on Orin.
- Jetson Nano completes CartPole training without OOM (override layer triggers ≥ 1 time during the run, confirming its activation on constrained hardware).

**Risk / Pitfall to watch**:
- If PPO uses multi-processing via `n_envs`, lowering the CPU frequency will cause env process progress to desynchronize, leading to straggler tail latencies (§9.7 pitfall 3). Use `n_envs = 1` initially; multi-env scaling is deferred to Week 9.
- Jetson Nano sysfs DVFS interfaces differ from Orin AGX (different cpufreq/devfreq layout). Allow ~1 day of buffer for Nano-specific path discovery; gate sysfs writes behind `--allow-real-dvfs` flag during porting.
- Code work on this and later weeks must go through Claude Code CLI with the **superpowers** plugin enabled (per project convention 2026-04-18). Do not write production code from the OpenClaw main session directly.
- **External-PC dependency for DonkeyCar (deferred)**: DonkeyCar runs require a separate x86 PC with discrete GPU to host the Unity-based simulator. This blocker is tracked outside the critical path; do not block Week 7 progress on DonkeyCar availability. When the external PC is provisioned, DonkeyCar runs can be retroactively added to Week 9/10/11 supplementary results.

---

### Week 8 (2026-06-08 to 2026-06-14)
**Theme**: Ablation study design and per-component overhead measurement

**Concrete tasks**:
1. Design and execute the ablation study: remove each of the four components individually (Preference Plane, Resource Manager, RL Arbiter, Override Layer) and measure the impact on HV, constraint violation rate, and tail latency. Record results in `eval/ablation_results.csv`.
2. Measure per-component runtime overhead and memory overhead: produce a table itemizing the cost of each subsystem (tegrastats daemon, DVFS controller, Lagrangian updater, override checker, preference embedding, interpolator). Apply NeuOS (Bateni 2020) LAG-metric encoding as an additional state feature for multi-DNN co-running scenarios and measure its overhead.
3. Run the latency-per-mJ visualization metric from DVFO (Zhang TMC 2023) across all preference vectors. Generate a Pareto curve with this metric on the x-axis.
4. Prepare a unified evaluation harness (`tetrarl/eval/runner.py`) that automates all experiments with configurable ablation flags, platform selection, and seed sweeps.
5. Identify and document any show-stopping bugs or performance regressions from the Week 5–7 integration. Reserve 1.5 days as buffer for critical fixes.

**Deliverables**:
- Ablation study results table (7 configurations × 4 metrics × 3 seeds) — paper Table candidate
- Per-component overhead breakdown table (runtime ms + memory MB per component) — paper Table candidate
- `tetrarl/eval/runner.py` — automated evaluation harness

**Validation criteria**:
- Each ablated component shows a statistically significant (p < 0.05, Welch t-test across 3 seeds) degradation in at least one metric, confirming that every subsystem is individually necessary.
- Total framework overhead is < 5% of the bare RL training step wall-clock time.

**Risk / Pitfall to watch**:
- If the override layer ablation does not show significant degradation under normal conditions, the ablation must be run under stress conditions (high co-runner load, low-battery simulation) where the override actually triggers. Design the ablation scenarios to exercise each component's critical path.

---

## Phase III — Multi-Platform Consolidation & Paper Draft (Weeks 9–12, REVISED 2026-04-18)

### Week 9 (2026-06-15 to 2026-06-21) — REVISED 2026-04-18
**Theme**: Nano deep-validation + DVFS-DRL-Multitask baseline + cross-platform consolidation

> **Plan revision (2026-04-18)**: Original Week 9 (Xavier NX + Nano porting) is replaced. Xavier NX is dropped (hardware unavailable). Nano porting moved to Week 7 Track B. This week now consolidates Nano deep-validation, the DVFS-DRL-Multitask baseline, and the multi-env scaling deferred from Week 7.

**Concrete tasks**:
1. **⚠️ BLOCKED ON EXTERNAL PC** — ~~Run the full DonkeyCar pipeline on Jetson Nano (200 episodes, 3 preference vectors).~~ DonkeyCar requires external x86 PC + discrete GPU (see Week 7 note). **Fallback**: run the substitute envs (DAG-scheduler-MO + CartPole) on Nano in their place; DonkeyCar runs are deferred to whenever the external PC is provisioned.
2. Re-fit the RBF interpolator on Nano if HV drops below 50% of Orin's on the **substitute env** (§3 Idea 3 failure mode — bottleneck shifts between CPU/GPU-bound regimes are non-linear across architectures). Re-fit on DonkeyCar later if the external PC becomes available.
3. Incorporate the DVFS-DRL-Multitask (2024) soft-deadline reward shaping (Algorithm 3) as an additional baseline. Run on **both Orin and Nano**.
4. Re-enable PPO multi-env (`n_envs > 1`) on Orin, the multi-env scaling deferred from Week 7. Document the `n_envs` × DVFS-frequency interaction (per §9.7 pitfall 3 straggler tail latencies).
5. Cross-platform tail-latency CDF expansion: extend the Week 7 Orin/Nano 2-panel figure with per-preference-vector breakdowns.

**Deliverables**:
- Nano substitute-env Pareto-front data (paper-figure-ready). DonkeyCar Pareto data deferred.
- DVFS-DRL-Multitask baseline results on Orin and Nano
- Multi-env scaling results table (`n_envs` ∈ {1, 2, 4} × {fixed-freq, DVFS-on})
- Expanded tail-latency CDF figure with preference-vector breakdowns

**Validation criteria**:
- Nano substitute env (DAG-scheduler-MO or CartPole) completes 200 episodes without OOM (override active). DonkeyCar criterion deferred.
- DVFS-DRL-Multitask baseline runs to completion on both Orin and Nano without crashes.
- Multi-env scaling does not cause > 3× tail-latency degradation when DVFS is active (otherwise document as a known limitation).

**Risk / Pitfall to watch**:
- The bottleneck shift between CPU-bound and GPU-bound tasks across different architectures is non-linear (§3 Idea 3 failure mode). The RBF interpolator trained on Orin data may produce mis-projections on Nano. Re-fit the interpolator per platform if HV drops below 50% of Orin's.

---

### Week 10 (2026-06-22 to 2026-06-28) — REVISED 2026-04-18
**Theme**: Full metric collection and Pareto-front analysis (Orin + Nano, no Xavier NX)

**Concrete tasks**:
1. Run the complete evaluation matrix: {SAC, PPO} × {Orin, Nano} × {5 preference vectors} × {3 seeds} on the available envs (PyBullet-HalfCheetah + DAG-scheduler-MO; DonkeyCar deferred until external PC provisioned). Collect all metrics: HV, tail-latency 99th, energy per step (J), memory peak (MB), reward. (Nano reduced scope: CartPole + DAG-scheduler-MO-reduced as feasible.)
2. Compute and plot Hypervolume (HV) graphs with ablation lines: PD-MORL (ours) vs. Envelope MORL (Yang 2019) vs. PPO-Lagrangian vs. FOCOPS (Zhang 2020) vs. DuoJoule vs. MAX-A vs. MAX-P. Apply PCN (Reymond 2022) as a discrete-action baseline if applicable.
3. Generate the Reward vs. Wall-clock Time and Reward vs. Cumulative Energy (Joules) plots (per §9.6: not Reward vs. Steps, which is unfair to on-policy algorithms).
4. Produce the dynamic preference-switching demonstration: a time-series showing smooth transitions as ω changes mid-episode (e.g., simulating "low battery → increase w_E").
5. Measure and document the Lagrangian constraint violation rate with and without the override layer. Present as a table showing that Lagrangian alone violates by 20–30% (confirming Spoor 2025 findings) but override + Lagrangian stays below 5%.

**Deliverables**:
- Complete evaluation results spreadsheet (`eval/full_results.csv`)
- HV comparison bar chart (7 methods × 2 platforms) — paper Figure candidate
- Reward vs. Wall-clock / Reward vs. Energy plots — paper Figure candidate
- Dynamic preference-switching time-series — paper Figure candidate
- Lagrangian + Override constraint satisfaction table — paper Table candidate

**Validation criteria**:
- TetraRL achieves the highest HV on Orin across all baselines (statistical significance via Welch t-test, p < 0.05).
- Dynamic preference switching shows smooth (< 10 episode) transition without reward collapse.
- Override layer reduces constraint violation from ~25% (Lagrangian only) to < 5%.

**Risk / Pitfall to watch**:
- Frequent and drastic preference switching (thrashing) can fill the replay buffer with contradictory transitions, causing training divergence (§3 Idea 1 honest failure mode). If observed, implement a preference-change cooldown period and document it.

---

### Week 11 (2026-06-29 to 2026-07-05) — REVISED 2026-04-18
**Theme**: Final figures, buffer, supplementary experiments

> **Plan revision (2026-04-18)**: Paper draft pushed to Week 12. This week is used for finalizing all paper-ready figures and tables, running any gap-filling experiments, and preparing the supplementary material. Also includes the Nano-GRPO micro-benchmark appendix.

**Concrete tasks**:
1. Produce all final figures: (a) Architecture diagram showing the four-component framework with kernel/user space interactions, (b) 4-D Pareto front projections, (c) tail-latency CDFs (Orin/Nano), (d) HV comparison, (e) ablation table, (f) overhead table, (g) dynamic preference switching, (h) training curves.
2. Prepare the Nano-GRPO micro-benchmark appendix (§9.3): a minimalist RLHF task for TinyLlama/Phi-3 on Orin AGX measuring system-level response curves, proving generality without requiring perfect alignment. Include critic-elimination as a discrete action knob (GRPO Shao 2024).
3. Re-run any experiments where gaps are identified (e.g., additional seeds, missing baselines, Nano-specific edge cases). Reserve 2 days as buffer.
4. Prepare supplementary material: (a) full hyperparameter tables, (b) per-seed raw data CSV, (c) code repository README with reproduction instructions.
5. Final figure polish: ensure all figures are vector PDFs where possible, legible at 300 DPI print resolution, and labeled consistently.

**Deliverables**:
- All figures finalized as vector PDFs in `paper/figures/`
- Nano-GRPO appendix data + figure
- Supplementary material package (hyperparameters, raw data, code)
- Gap-filling experiment results (if any)

**Validation criteria**:
- All figures are legible at print resolution (300 DPI minimum for raster elements).
- Supplementary material covers every experimental claim (each claim traceable to a specific CSV/figure).
- Nano-GRPO appendix demonstrates measurable system-level response curves on Orin.

**Risk / Pitfall to watch**:
- If Week 10 experiments are delayed, this week's buffer must absorb the overflow. Prioritize figure generation over supplementary polish.

---

### Week 12 (2026-07-06 to 2026-07-12) — REVISED 2026-04-18
**Theme**: Paper draft, internal review, and submission preparation

> **Plan revision (2026-04-18)**: Per user direction, the full paper draft is written in Week 12 (not earlier). All experimental data, figures, and supplementary material should be complete by Week 11.

**Concrete tasks**:
1. Write the full paper in LaTeX: Title, Abstract, Introduction (with "Differences from Prior Conference Papers R³ and DuoJoule" subsection per §6), System Model, Architecture, Algorithm, Evaluation, Related Work, Limitations, Conclusion.
2. Write the Related Work section covering: MORL (PD-MORL, Envelope, PG-MORL, PCN), Edge ML (DVFO, DVFS-DRL-Multitask, SparseDVFS), Constrained RL (CPO, FOCOPS, OmniSafe), and prior group work (R³, DuoJoule, NeuOS, RED, RT-LM, BOXR, MIMONet). Reference GRPO/DAPO in the Discussion as future work per §9.3.
3. Write the Limitations paragraph (§7 items 1–7): DVFS transition overhead, non-stationary Pareto fronts, thermal throttling, OS/ROS interference, replay buffer fragmentation, Lagrangian constraint violation, and interpolator quality at discontinuities.
4. Draft the System and Resource Model section early in the paper: GPU unified-memory contention model and DVFS state-machine transition delay model (per §6 TC editorial preferences).
5. Conduct internal review: circulate draft, address comments, refine writing.
6. Write the cover letter for TC submission, emphasizing the ≥ 50% new material over R³/DuoJoule and the 2-platform (Orin + Nano) evaluation.
7. Final proofreading pass: verify all figure references, table numbering, equation formatting, and bibliography completeness. Run a plagiarism self-check.

**Deliverables**:
- Complete LaTeX draft with all sections populated (≥ 12 pages, double-column TC format)
- Complete bibliography (≥ 50 references)
- Cover letter for TC submission
- Final camera-ready LaTeX PDF

**Validation criteria**:
- The "Differences from Prior Conference Papers" subsection explicitly enumerates ≥ 5 architectural differences from R³ and ≥ 3 from DuoJoule, demonstrating > 50% new material.
- The Limitations paragraph covers all 7 items from the brainstorm document.
- The paper compiles without LaTeX warnings.
- All experimental claims in the paper are backed by data in the supplementary material.
- The paper narrative centers on "embedded runtime system that integrates MORL" rather than "new MORL algorithm applied to edge." The architecture diagram and system model appear before the algorithm section (per §6 TC scope match).

**Risk / Pitfall to watch**:
- The ≥ 30% new material requirement (§6) is the single most common rejection reason for journal extensions. The cover letter must explicitly map each new section/experiment to a percentage of the total paper. Allocate time for a careful accounting.
- Writing the full draft in one week is aggressive. Mitigate by having the LaTeX skeleton, all figures, and related work references prepared in Week 11. Week 12 should focus on prose, not figure generation.

---

## Cross-Week Dependencies & Critical Path

```
Week 1 ──→ Week 2 ──→ Week 3 ──→ Week 4  (sequential: algorithm builds on prior)
                                    │
                                    ▼
              Week 5 ──→ Week 6 ──→ Week 7 ──→ Week 8  (sequential: systems integration)
                                     │ (Track B)    │
                                     └─ Nano port ──┘
                                                 │
                                                 ▼
                          Week 9 ──→ Week 10 ──→ Week 11 ──→ Week 12  (eval + figures + paper)
```

**Critical path**: Weeks 1 → 2 → 3 → 5 → 6 → 7 → 10 → 11 → 12

- **Week 2 blocks Week 3**: C-MORL must work before 4-D reward vector can be tested.
- **Week 3 blocks Week 5**: The z-score normalization and interpolator must be validated on simulator before integration with real hardware sensors.
- **Week 4 is semi-independent**: PPO-Lagrangian baseline and override layer can be developed in parallel with Week 3, but both must be ready before Week 6 integration.
- **Week 5 blocks Week 6**: Tegrastats daemon and DVFS controller must function before the four-component framework can be wired.
- **Week 6 blocks Week 7**: The framework must run end-to-end before full closed-loop validation.
- **Week 7 Track B (Nano port)** runs in parallel with Track A (Orin closed-loop). Nano porting (originally Week 9) is pulled forward.
- **Week 7 blocks Week 10**: Orin AGX Pareto front data is needed for the full evaluation matrix.
- **Week 8 blocks Week 11**: Ablation results and overhead tables are required for final figures.
- **Week 9 consolidates Nano deep-validation** and DVFS-DRL-Multitask baseline on both platforms.
- **Week 10 blocks Week 11**: All figures and tables must be generated before Week 11 figure finalization.
- **Week 11 blocks Week 12**: All figures, supplementary material, and Nano-GRPO appendix must be ready before paper writing begins.
- **Week 12 = paper draft + submit** (no earlier paper writing). Aggressive but feasible if Weeks 10-11 deliver complete data/figures.

---

## Buffer / Contingency Strategy

| Week | Built-in Slack | Contingency if Delayed |
|------|---------------|----------------------|
| Week 1 | 0.5 days | DST reproduction is straightforward; if CI setup takes longer, defer to Week 2 |
| Week 2 | 0 days (tight) | If C-MORL diverges, fall back to MO-TD3-HER verbatim and revisit SAC port in Week 4 buffer |
| Week 3 | 1 day | Interpolator pre-training is lightweight; risk is in the synthetic 4-D benchmark setup |
| Week 4 | 1.5 days | PPO-Lagrangian is well-documented in OmniSafe; reserve buffer for integration debugging |
| Week 5 | 1 day | DVFS profiling is mechanical; risk is in tegrastats daemon stability under long runs |
| Week 6 | 0 days (tight) | This is the highest-risk week (full integration). If blocked, delay Week 7 experiments and use Week 8 buffer |
| Week 7 | 0.5 days | DonkeyCar deferred (external PC dependency); substitute envs (DAG-scheduler-MO + PyBullet-HalfCheetah) used in its place |
| Week 8 | 1.5 days | Explicitly reserved as integration buffer for Weeks 5–7 spillover |
| Week 9 | 1 day | Nano deep-validation + DVFS-DRL-Multitask baseline; memory-specific workarounds may be needed |
| Week 10 | 0.5 days | Evaluation runs are parallelizable across platforms; bottleneck is Nano's slow training |
| Week 11 | 1 day | Final figures + supplementary; buffer for Week 10 overflow |
| Week 12 | 0 days (tight) | Full paper draft + submit — no buffer; all data/figures must be done by Week 11 |

**Global contingency**: If Weeks 6–7 encounter show-stopping hardware issues (e.g., Orin kernel bugs, DVFS driver incompatibilities), the fallback plan is to demonstrate the full 4-D system on the desktop GPU with simulated DVFS/energy and port to Orin for a reduced evaluation (Orin-only, no Nano). This sacrifices the multi-platform contribution but preserves the algorithmic and systems architecture contributions. Xavier NX has been removed from the plan (hardware unavailable).

**Method Borrowing Map reference**: The following papers are actively applied across the 12 weeks:

| Paper | Weeks Applied | Component |
|-------|--------------|-----------|
| PD-MORL (Basaklar 2023) | 1, 2, 3, 10 | Core MORL algorithm, cosine-similarity operator, interpolator |
| Envelope MORL (Yang 2019) | 1, 2, 10 | HER-style preference replay |
| R³ (Li 2023) | 6, 7, 8 | State cache, soft-link buffer, MAX-A/MAX-P baselines, FFmpeg protocol |
| Lagrangian-Empirical (Spoor 2025) | 4, 7, 10 | PI gains, anti-windup, constraint violation analysis |
| CPO (Achiam 2017) | 4, 8 | Recovery step for hardware override |
| FOCOPS (Zhang 2020) | 4, 10 | First-order constrained RL ablation arm |
| OmniSafe (Ji 2023) | 4 | Starting codebase for PPO-Lagrangian |
| DVFS-DRL-Multitask (2024) | 5, 7, 9 | Kernel/user tegrastats split, soft-deadline reward shaping |
| DVFO (Zhang 2023) | 7, 8 | Concurrent decision trick, latency-per-mJ metric |
| SparseDVFS (2025) | 5 | Super-block DVFS granularity |
| NeuOS (Bateni 2020) | 8 | LAG metric for co-running state |
| PCN (Reymond 2022) | 10 | Discrete-action baseline |
| GRPO (Shao 2024) | 12 | Critic-elimination knob for Nano-GRPO appendix |
