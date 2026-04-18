# TetraRL Native: Preference-Conditioned PPO Design

## Motivation

Weeks 1-2 revealed four architectural barriers preventing C-MORL (Liu, ICLR 2025) from running on NVIDIA Jetson Orin AGX (see `docs/c_morl_systems_analysis.md`):

| Layer | Barrier | Root Cause |
|-------|---------|------------|
| 1 | Multi-process architecture | PGMORL legacy `mp.Queue` + fork pattern |
| 2 | CUDA fork safety | `fork` after CUDA init = undefined behavior |
| 3 | EnergyPlus file descriptors | Inherited across fork, collide |
| 4 | L4T kernel quirks | JetPack `/proc/` layout differences |

**Path D**: rather than patching 2018-era OpenAI Baselines code through four compatibility layers, build a modern single-process implementation from scratch.

## Architecture

```
tetrarl/morl/native/
├── __init__.py            # Exports TetraRLNativeAgent
├── NOTICE                 # cleanrl attribution (MIT)
├── ppo_base.py            # Vendored cleanrl PPO (reference only)
├── preference_ppo.py      # Core: preference-conditioned PPO
└── agent.py               # High-level TetraRLNativeAgent wrapper
```

### Core Idea: Preference-Conditioned Policy

Instead of training N separate policies (C-MORL approach), train **one** policy conditioned on a preference vector ω:

```
π(a | s, ω)    where ω ∈ Δ^{k-1} (k-simplex)
```

Three modifications to standard PPO:

1. **Observation augmentation**: `obs_aug = concat(obs, ω)`
2. **Reward scalarization**: `r_scalar = ω · r_vec` (linear scalarization)
3. **Preference sampling**: new ω ~ Dirichlet(1) sampled per episode

### Network Architecture

```
PreferenceNetwork(obs_dim + pref_dim → hidden → hidden → act_dim / 1)
├── critic:       (obs_aug) → scalar value
├── actor_mean:   (obs_aug) → action mean        [continuous]
├── actor_logstd: learnable log-std               [continuous]
└── actor_logits: (obs_aug) → action logits       [discrete]
```

Supports both discrete (DST) and continuous (MO-Hopper) action spaces.

### Training Loop (Single Process)

```
for iteration in 1..N:
    for step in 1..num_steps:           # Collect rollout
        obs_aug = concat(obs, ω)
        action ~ π(· | obs_aug)
        r_scalar = ω · env.step(action).reward_vec
        if episode_done: ω ~ Dirichlet(1)
    GAE(γ, λ)                           # Compute advantages
    PPO_update(clip, entropy, value)    # Standard PPO
    if eval_time:                       # Periodic evaluation
        for ω_eval in corners + interior:
            evaluate(π, ω_eval) → obj_return
        pareto_filter + hypervolume
```

### Pareto Front Discovery

During periodic evaluation:
1. Generate preference vectors: k corners + n interior (Dirichlet samples)
2. For each ω, run policy deterministically for m episodes
3. Record mean multi-objective return
4. Apply `pareto_filter()` and compute `hypervolume()`
5. Track best-HV network snapshot

## 4-Component Framework Integration

| Component | TetraRL Native Role |
|-----------|-------------------|
| **Preference Plane** | ω vector input to policy; Dirichlet sampling |
| **Resource Manager** | Single-process = no mp overhead; DVFS-compatible |
| **RL Arbiter** | PPO with preference conditioning |
| **Override Layer** | Compatible — single thread, no fork hazards |

## Comparison: C-MORL vs TetraRL Native

| Aspect | C-MORL (Path B) | TetraRL Native (Path D) |
|--------|-----------------|------------------------|
| Architecture | Multi-process (mp.Queue) | Single-process |
| Policy count | N separate policies | 1 preference-conditioned |
| Base code | OpenAI Baselines (2018) | cleanrl (2022) |
| CUDA safety | Fork hazard | No fork |
| Edge-ready | Requires patches | Native |
| Pareto discovery | PGMORL evolutionary | Preference sweep |
| Dependencies | baselines, mpi4py | torch, gymnasium |

## API

```python
from tetrarl.morl.native import TetraRLNativeAgent

agent = TetraRLNativeAgent(
    env_name="dst",           # or "mo-hopper-v4"
    obj_num=2,
    ref_point=[0.0, -25.0],
    total_timesteps=100_000,
    device="cpu",             # or "cuda"
)

# Train
results = agent.train(verbose=True)

# Inspect Pareto front
front = agent.get_pareto_front()
print(f"HV={front['hv']:.2f}, |PF|={len(front['objectives'])}")

# Evaluate at specific preference
obj = agent.evaluate(np.array([0.7, 0.3]))

# Save / load
agent.save("checkpoints/dst_native")
agent.load("checkpoints/dst_native")
```

## Deviations from cleanrl Reference

1. **No SyncVectorEnv**: uses single env directly (edge constraint)
2. **No observation normalization**: omitted for simplicity; add via env wrapper if needed
3. **No TensorBoard writer**: metrics returned in result dict instead
4. **Added discrete action support**: cleanrl's continuous PPO extended with `Categorical`
5. **Added preference conditioning**: observation augmentation + reward scalarization
6. **Added Pareto evaluation**: periodic preference sweep + hypervolume tracking
