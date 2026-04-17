# TetraRL / R⁴ TC Paper — 12-Week Action Plan

**Document version**: 2026-04-17
**Calendar span**: Week 1 (2026-04-20) through Week 12 (ending 2026-07-12)
**Target venue**: IEEE Transactions on Computers (TC)

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
**Theme**: Continuous-action PD-MORL (MO-TD3-HER → MO-SAC-HER) on MuJoCo

**Concrete tasks**:
1. Implement `tetrarl/morl/mo_sac_her.py` — port PD-MORL's MO-TD3-HER (Basaklar 2023 Eq. 11–12) to SAC by adding the directional regularization term `g(ω_p, Q) = arccos(Sc(ω_p, Q))` to both the SAC critic and actor losses. Apply PD-MORL cosine-similarity from Basaklar 2023 → `loss.py`.
2. Implement HER-style preference replay: each transition is re-labeled with N_w = 32 resampled ω′ values (borrowing from Yang 2019 but adapted to the PD-MORL cosine operator).
3. Configure MO-MuJoCo benchmarks (Walker2d, HalfCheetah) following PD-MORL's experimental setup (1M steps each).
4. Run MO-SAC-HER on MO-Walker2d and MO-HalfCheetah; log HV, Sparsity, and per-objective return curves.

**Deliverables**:
- `tetrarl/morl/mo_sac_her.py` — continuous-action MO-SAC-HER agent
- `tetrarl/morl/loss.py` — modular loss functions (cosine-similarity critic loss, directional actor loss)
- HV comparison table (MO-SAC-HER vs. PD-MORL MO-TD3-HER vs. Envelope MOQL on Walker2d/HalfCheetah)

**Validation criteria**:
- MO-Walker2d HV reaches ≥5.0e6 (PD-MORL reports 5.41e6 with TD3; ≥92% is acceptable for the SAC port).
- No NaN or divergence in any training run across 3 random seeds.

**Risk / Pitfall to watch**:
- PD-MORL's original paper uses TD3; the SAC migration adds entropy regularization that may interact adversely with the directional loss term. Monitor the α (entropy coefficient) auto-tuning and clamp if critic loss spikes.

---

### Week 3 (2026-05-04 to 2026-05-10)
**Theme**: 4-D reward vector, z-score normalization, and RBF interpolator

**Concrete tasks**:
1. Define the 4-D reward vector `r = [r_env, −c_L(L_prev), −c_E(E_step), −c_M(M_util)]` in `tetrarl/core/reward.py`. Implement per-dimension running z-score normalization with a 1k-step warmup window.
2. Implement the RBF interpolator I(ω) in `tetrarl/morl/interpolator.py`. Pre-train on 11 anchor preferences: 4 one-hot corners + 6 edge midpoints + 1 center (per PD-MORL Basaklar 2023 §4.2). Validate projection quality by checking that one-hot ω inputs recover the corresponding extremal Q-vector directions.
3. Implement structured preference sampling in `tetrarl/morl/preference.py`: corner sampling (one-hot) + edge + interior, replacing uniform Dirichlet (following the reusable insight from Envelope MORL Yang 2019).
4. Run MO-SAC-HER on a synthetic 4-D MuJoCo variant (HalfCheetah with two artificial cost dimensions appended) to stress-test the z-score normalization and interpolator under heterogeneous scales.

**Deliverables**:
- `tetrarl/core/reward.py` — 4-D reward vector with z-score normalization
- `tetrarl/morl/interpolator.py` — RBF interpolator with 11-anchor pre-training
- Training curve showing stable 4-D HV convergence on synthetic HalfCheetah-4D (no divergence over 500k steps)

**Validation criteria**:
- Z-score normalization ensures all 4 reward dimensions have mean ≈ 0 and std ≈ 1 after the warmup window (verified via logged statistics).
- RBF interpolator reconstruction error ≤ 5% on held-out anchor preferences.

**Risk / Pitfall to watch**:
- The RBF interpolator may produce mis-projections at DVFS step-jump boundaries because the hardware cost surface is discontinuous at frequency jumps (identified in §7 limitation 7 of the brainstorm document). On simulator this week, verify that the interpolator handles step-function cost surfaces by testing with a piecewise-constant synthetic cost dimension.

---

### Week 4 (2026-05-11 to 2026-05-17)
**Theme**: Lagrangian dual-variable infrastructure and PPO-Lagrangian baseline

**Concrete tasks**:
1. Implement the Lagrangian dual-variable updater in `tetrarl/sys/lagrangian.py` using a PI controller with initial gains K_P = K_I = 10⁻⁴, K_D = 0, and anti-windup cap I_max (directly borrowing from Lagrangian-Empirical Spoor 2025).
2. Implement the hardware-emergency override layer in `tetrarl/sys/override.py`, inspired by the CPO recovery step (Achiam 2017 Eq. 14): when energy or memory limits are exceeded, the override ignores policy gradients and forces conservative action.
3. Integrate PPO-Lagrangian baseline using OmniSafe (Ji 2023) as the starting codebase. Configure the CMDP wrapper with memory and energy constraints.
4. Run PPO-Lagrangian on PyBullet (HalfCheetah, Ant) with simulated energy/memory costs to validate mathematical correctness of dual updates. Apply FOCOPS (Zhang NeurIPS 2020) first-order primal-dual update as an ablation arm.
5. Implement the unified knob taxonomy (§9.4): map `replay_buffer_size` (off-policy) and `n_steps` (on-policy) to a common `M_store` primitive in `tetrarl/sys/knob_mapper.py`.

**Deliverables**:
- `tetrarl/sys/lagrangian.py` — PI-based dual-variable updater with anti-windup
- `tetrarl/sys/override.py` — hardware-emergency override layer
- `tetrarl/sys/knob_mapper.py` — unified PPO/SAC primitive mapper
- PPO-Lagrangian convergence plots on PyBullet with constraint satisfaction curves (cost vs. limit over training)

**Validation criteria**:
- PPO-Lagrangian on PyBullet-HalfCheetah converges to ≥80% of unconstrained PPO reward while maintaining constraint violations below the specified limit for ≥90% of evaluation episodes.
- The override layer triggers correctly when simulated energy exceeds threshold (unit-tested).

**Risk / Pitfall to watch**:
- Lagrangian methods frequently violate constraints even in simulations (Spoor 2025 reports cost of 30.84 vs. limit 25 in PointGoal1). The override layer must be validated as the true safety net this week, not the Lagrangian itself.

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
4. Port the full MO-SAC-HER agent to run on Orin AGX with the tegrastats daemon providing real energy/memory readings. Replace simulated reward dimensions with real hardware metrics.
5. Run a first end-to-end loop: CartPole (Classic Control) with MO-SAC-HER + DVFS + tegrastats on Orin AGX. Log all 4 dimensions over 100 episodes.

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

### Week 7 (2026-06-01 to 2026-06-07)
**Theme**: Full closed-loop validation with SAC + PPO on Orin AGX

**Concrete tasks**:
1. Run MO-SAC-HER (off-policy) on DonkeyCar simulator with DVFS + tegrastats on Orin AGX for 500 episodes. Sweep 5 representative preference vectors ω (one-hot corners + center).
2. Run PPO-Lagrangian (on-policy) on PyBullet-HalfCheetah with the unified knob mapper (`n_steps`, `n_epochs`, `mini_batch_size` as knobs per §9.2) on Orin AGX. Validate that the Lagrangian dual variables converge and constraints are respected (with override layer as safety net).
3. Implement the "thinking-while-moving" concurrent decision trick from DVFO (Zhang TMC 2023) in `tetrarl/sys/concurrent.py` — overlap DVFS decision computation with RL forward pass to mask decision-loop overhead.
4. Generate Pareto front visualization: 2-D projections of the 4-D Pareto front (T vs. A, E vs. A, M vs. A) for the DonkeyCar-SAC runs. Compute HV indicator.
5. Run the co-runner FFmpeg interference test (R³ protocol, Fig. 15): 720p / 1080p / 2K background video decoding while training. Log tail-latency shifts.

**Deliverables**:
- `tetrarl/sys/concurrent.py` — decision-loop overhead masking
- 4-D Pareto front scatter plots (3 × 2-D projections) for DonkeyCar-SAC on Orin — paper-figure-ready
- FFmpeg co-runner interference table (tail-latency 99th percentile at 720p/1080p/2K) — paper-figure-ready

**Validation criteria**:
- DonkeyCar-SAC achieves ≥70% of MAX-A reward while maintaining zero deadline misses under the center preference ω = [0.25, 0.25, 0.25, 0.25].
- PPO-Lagrangian constraint violation rate is < 20% of episodes (with override layer active).
- FFmpeg co-runner at 1080p does not increase 99th-percentile latency by more than 2× relative to the no-interference case.

**Risk / Pitfall to watch**:
- If PPO uses multi-processing via `n_envs`, lowering the CPU frequency will cause env process progress to desynchronize, leading to straggler tail latencies (§9.7 pitfall 3). Use `n_envs = 1` initially; multi-env scaling is deferred to Week 9.

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

## Phase III — Multi-Platform Scaling & Paper Draft (Weeks 9–12)

### Week 9 (2026-06-15 to 2026-06-21)
**Theme**: Porting to Xavier NX and Jetson Nano

**Concrete tasks**:
1. Port the tegrastats daemon and DVFS controller to Xavier NX and Jetson Nano. Adjust frequency tables and EMA filter parameters per platform. Re-profile DVFS transition latencies.
2. Run the full MO-SAC-HER pipeline on Xavier NX (DonkeyCar, 500 episodes, 5 preference vectors). Adjust super-block granularity N if DVFS overhead differs from Orin.
3. Run a reduced-scope experiment on Jetson Nano (CartPole + Classic Control only, due to memory constraints). Validate that the override layer correctly prevents OOM on Nano's 4 GB memory.
4. Incorporate the DVFS-DRL-Multitask (2024) soft-deadline reward shaping (Algorithm 3) as an additional baseline. Run on all 3 platforms.
5. Collect tail-latency CDFs across all 3 platforms for the same preference vector. Generate a 3-panel CDF figure (one per platform).

**Deliverables**:
- Platform-specific DVFS transition latency tables for Xavier NX and Jetson Nano
- 3-platform tail-latency CDF figure (Orin / Xavier NX / Nano) — paper Figure candidate
- DVFS-DRL-Multitask baseline results on all 3 platforms

**Validation criteria**:
- Xavier NX achieves ≥60% of Orin's HV on the same DonkeyCar task (accounting for reduced compute).
- Jetson Nano completes CartPole training without OOM (override layer triggers ≥ 1 time during the run, confirming its activation on constrained hardware).
- DVFS-DRL-Multitask baseline runs to completion on all 3 platforms without crashes.

**Risk / Pitfall to watch**:
- The bottleneck shift between CPU-bound and GPU-bound tasks across different architectures is non-linear (§3 Idea 3 failure mode). The RBF interpolator trained on Orin data may produce mis-projections on Nano. Re-fit the interpolator per platform if HV drops below 50% of Orin's.

---

### Week 10 (2026-06-22 to 2026-06-28)
**Theme**: Full metric collection and Pareto-front analysis

**Concrete tasks**:
1. Run the complete evaluation matrix: {SAC, PPO} × {Orin, Xavier NX, Nano} × {5 preference vectors} × {3 seeds}. Collect all metrics: HV, tail-latency 99th, energy per step (J), memory peak (MB), reward.
2. Compute and plot Hypervolume (HV) graphs with ablation lines: PD-MORL (ours) vs. Envelope MORL (Yang 2019) vs. PPO-Lagrangian vs. FOCOPS (Zhang 2020) vs. DuoJoule vs. MAX-A vs. MAX-P. Apply PCN (Reymond 2022) as a discrete-action baseline if applicable.
3. Generate the Reward vs. Wall-clock Time and Reward vs. Cumulative Energy (Joules) plots (per §9.6: not Reward vs. Steps, which is unfair to on-policy algorithms).
4. Produce the dynamic preference-switching demonstration: a time-series showing smooth transitions as ω changes mid-episode (e.g., simulating "low battery → increase w_E").
5. Measure and document the Lagrangian constraint violation rate with and without the override layer. Present as a table showing that Lagrangian alone violates by 20–30% (confirming Spoor 2025 findings) but override + Lagrangian stays below 5%.

**Deliverables**:
- Complete evaluation results spreadsheet (`eval/full_results.csv`)
- HV comparison bar chart (7 methods × 3 platforms) — paper Figure candidate
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

### Week 11 (2026-06-29 to 2026-07-05)
**Theme**: Paper draft — structure, figures, and Related Work

**Concrete tasks**:
1. Write the paper skeleton in LaTeX: Title, Abstract, Introduction (with "Differences from Prior Conference Papers R³ and DuoJoule" subsection per §6), System Model, Architecture, Algorithm, Evaluation, Related Work, Limitations, Conclusion.
2. Produce all final figures: (a) Architecture diagram showing the four-component framework with kernel/user space interactions, (b) 4-D Pareto front projections, (c) tail-latency CDFs, (d) HV comparison, (e) ablation table, (f) overhead table, (g) dynamic preference switching, (h) training curves.
3. Write the Related Work section covering: MORL (PD-MORL, Envelope, PG-MORL, PCN), Edge ML (DVFO, DVFS-DRL-Multitask, SparseDVFS), Constrained RL (CPO, FOCOPS, OmniSafe), and prior group work (R³, DuoJoule, NeuOS, RED, RT-LM, BOXR, MIMONet). Reference GRPO/DAPO in the Discussion as future work per §9.3.
4. Write the Limitations paragraph (§7 items 1–7): DVFS transition overhead, non-stationary Pareto fronts, thermal throttling, OS/ROS interference, replay buffer fragmentation, Lagrangian constraint violation, and interpolator quality at discontinuities.
5. Draft the System and Resource Model section early in the paper: GPU unified-memory contention model and DVFS state-machine transition delay model (per §6 TC editorial preferences).

**Deliverables**:
- LaTeX draft with all sections populated (≥ 12 pages, double-column TC format)
- All figures finalized as vector PDFs in `paper/figures/`
- Complete bibliography (≥ 50 references)

**Validation criteria**:
- The "Differences from Prior Conference Papers" subsection explicitly enumerates ≥ 5 architectural differences from R³ and ≥ 3 from DuoJoule, demonstrating > 50% new material.
- All figures are legible at print resolution (300 DPI minimum for raster elements).
- The Limitations paragraph covers all 7 items from the brainstorm document.

**Risk / Pitfall to watch**:
- The paper must not read as a pure RL algorithm derivation (§6 TC scope match). Ensure the narrative centers on "embedded runtime system that integrates MORL" rather than "new MORL algorithm applied to edge." The architecture diagram and system model must appear before the algorithm section.

---

### Week 12 (2026-07-06 to 2026-07-12)
**Theme**: Paper polishing, internal review, and submission preparation

**Concrete tasks**:
1. Conduct an internal review cycle: circulate the draft for feedback, address all comments, and refine the writing.
2. Re-run any experiments where reviewer feedback identifies gaps (e.g., additional seeds, missing baselines, clarification experiments).
3. Prepare the supplementary material: (a) full hyperparameter tables, (b) per-seed raw data, (c) code repository README with reproduction instructions.
4. Write the cover letter for TC submission, emphasizing the ≥ 50% new material over R³/DuoJoule and the multi-platform evaluation.
5. Final proofreading pass: verify all figure references, table numbering, equation formatting, and bibliography completeness. Run a plagiarism self-check.
6. Prepare the Nano-GRPO micro-benchmark appendix (§9.3): a minimalist RLHF task for TinyLlama/Phi-3 on Orin AGX measuring system-level response curves, proving generality without requiring perfect alignment. Include critic-elimination as a discrete action knob (GRPO Shao 2024).

**Deliverables**:
- Final camera-ready LaTeX PDF
- Supplementary material package (hyperparameters, raw data, code)
- Cover letter for TC submission
- Nano-GRPO appendix demonstrating LLM generality

**Validation criteria**:
- All experimental claims in the paper are backed by data in the supplementary material (each claim traceable to a specific CSV/figure).
- The paper compiles without LaTeX warnings.
- Internal reviewers confirm that all 4 stated contributions (§6) are substantiated by the evaluation.

**Risk / Pitfall to watch**:
- The ≥ 30% new material requirement (§6) is the single most common rejection reason for journal extensions. The cover letter must explicitly map each new section/experiment to a percentage of the total paper. Allocate time for a careful accounting.

---

## Cross-Week Dependencies & Critical Path

```
Week 1 ──→ Week 2 ──→ Week 3 ──→ Week 4  (sequential: algorithm builds on prior)
                                    │
                                    ▼
              Week 5 ──→ Week 6 ──→ Week 7 ──→ Week 8  (sequential: systems integration)
                                                 │
                                                 ▼
                          Week 9 ──→ Week 10 ──→ Week 11 ──→ Week 12  (sequential: eval + writing)
```

**Critical path**: Weeks 1 → 2 → 3 → 5 → 6 → 7 → 10 → 11 → 12

- **Week 2 blocks Week 3**: MO-SAC-HER must work before 4-D reward vector can be tested.
- **Week 3 blocks Week 5**: The z-score normalization and interpolator must be validated on simulator before integration with real hardware sensors.
- **Week 4 is semi-independent**: PPO-Lagrangian baseline and override layer can be developed in parallel with Week 3, but both must be ready before Week 6 integration.
- **Week 5 blocks Week 6**: Tegrastats daemon and DVFS controller must function before the four-component framework can be wired.
- **Week 6 blocks Week 7**: The framework must run end-to-end before full closed-loop validation.
- **Week 7 blocks Week 10**: Orin AGX Pareto front data is needed for the full evaluation matrix.
- **Week 8 blocks Week 11**: Ablation results and overhead tables are required for the paper draft.
- **Week 9 is partially independent**: Xavier NX and Nano porting can begin as soon as Week 6 is complete (code runs on Orin); full data collection requires the evaluation harness from Week 8.
- **Week 10 blocks Week 11**: All figures and tables must be generated before the paper draft.

---

## Buffer / Contingency Strategy

| Week | Built-in Slack | Contingency if Delayed |
|------|---------------|----------------------|
| Week 1 | 0.5 days | DST reproduction is straightforward; if CI setup takes longer, defer to Week 2 |
| Week 2 | 0 days (tight) | If MO-SAC-HER diverges, fall back to MO-TD3-HER verbatim and revisit SAC port in Week 4 buffer |
| Week 3 | 1 day | Interpolator pre-training is lightweight; risk is in the synthetic 4-D benchmark setup |
| Week 4 | 1.5 days | PPO-Lagrangian is well-documented in OmniSafe; reserve buffer for integration debugging |
| Week 5 | 1 day | DVFS profiling is mechanical; risk is in tegrastats daemon stability under long runs |
| Week 6 | 0 days (tight) | This is the highest-risk week (full integration). If blocked, delay Week 7 experiments and use Week 8 buffer |
| Week 7 | 0.5 days | DonkeyCar experiments are long-running; parallelizable with PPO runs |
| Week 8 | 1.5 days | Explicitly reserved as integration buffer for Weeks 5–7 spillover |
| Week 9 | 1 day | Xavier NX port is a subset of Orin code; Nano may require memory-specific workarounds |
| Week 10 | 0.5 days | Evaluation runs are parallelizable across platforms; bottleneck is Nano's slow training |
| Week 11 | 0 days (tight) | Paper writing is on the critical path; no buffer — start figure production in Week 10 if possible |
| Week 12 | 2 days | Explicitly reserved as paper-polishing buffer; re-run experiments only if gaps are identified |

**Global contingency**: If Weeks 6–7 encounter show-stopping hardware issues (e.g., Orin kernel bugs, DVFS driver incompatibilities), the fallback plan is to demonstrate the full 4-D system on the desktop GPU with simulated DVFS/energy and port to Orin for a reduced evaluation (Orin-only, no Xavier NX/Nano). This sacrifices the multi-platform contribution but preserves the algorithmic and systems architecture contributions.

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
| DVFS-DRL-Multitask (2024) | 5, 9 | Kernel/user tegrastats split, soft-deadline reward shaping |
| DVFO (Zhang 2023) | 7, 8 | Concurrent decision trick, latency-per-mJ metric |
| SparseDVFS (2025) | 5 | Super-block DVFS granularity |
| NeuOS (Bateni 2020) | 8 | LAG metric for co-running state |
| PCN (Reymond 2022) | 10 | Discrete-action baseline |
| GRPO (Shao 2024) | 12 | Critic-elimination knob for Nano-GRPO appendix |
