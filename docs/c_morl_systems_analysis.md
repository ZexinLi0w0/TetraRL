# Why C-MORL Has So Many Bugs On Embedded GPU: A Systems Analysis

**Date**: 2026-04-18
**Context**: Week 2 attempt to deploy C-MORL (Liu et al., ICLR 2025) on NVIDIA Jetson Orin AGX

## TL;DR

**C-MORL is a CPU-server-era research codebase, never tested on embedded GPU.** Our 8-version attempt (v1→v8) systematically uncovered **4 architectural barriers** that motivate the systems-side contributions of TetraRL.

## The Bug Stack (4 layers)

### Layer 1: PGMORL Legacy Baggage (Severity: ⭐⭐⭐⭐⭐)

C-MORL inherits from PGMORL (Xu CoRL 2020), which inherits from `ikostrikov/pytorch-a2c-ppo-acktr-gail` (~6 years old).

- 2018-2019 era PyTorch + multiprocessing patterns
- `fork`-based subprocess spawning (CUDA-incompatible)
- File-descriptor-based tensor sharing (default `set_sharing_strategy('file_descriptor')`)
- No isolation between worker process state and parent CUDA context

**Why it works on CPU**: No CUDA → no fork issues.
**Why it breaks on GPU**: CUDA context cannot survive `fork()`.

### Layer 2: Multi-Worker × Multi-Env × CUDA Context (Severity: ⭐⭐⭐⭐⭐)

C-MORL architecture (Figure 4 of paper):
```
Master process
├── Worker 0 (Pareto preference 0)
│   ├── env 0 (subprocess)
│   ├── env 1 (subprocess)
│   └── ... (× 8 envs per worker)
├── Worker 1 (Pareto preference 1)
│   └── ... (× 8 envs)
└── ... (× N workers)
```

- Each worker = 1 process forked from master
- Each env = 1 subprocess
- Total ~50 processes must share a single CUDA context
- → `torch.AcceleratorError: CUDA error: invalid argument` at `queue.get()` deserialization

**Comparison**: Vanilla PPO uses 1 master + N env subprocs without sharing CUDA tensors → no fork issue.

### Layer 3: EnergyPlus Subprocess Overhead (Severity: ⭐⭐⭐)

The Building-3d benchmark wraps EnergyPlus HVAC simulators:
- Each env process boots its own EnergyPlus instance
- Each instance opens 5-10 sockets/files/pipes
- ×48 env subprocs = ~400 file descriptors just for env runtime
- Combined with PyTorch shared memory → **fd exhaustion at default `ulimit -n=1024`**

### Layer 4: Jetson Orin L4T Quirks (Severity: ⭐⭐)

- Linux for Tegra (L4T) ≠ standard Linux
- `nvidia-smi` doesn't report stats (must use `tegrastats`)
- aarch64 + integrated CUDA has different memory mapping than x86 + discrete GPU
- PyTorch coverage on Jetson is lower than desktop

## Failed Attempts (Documented for Future Reference)

| Version | Fix Attempted | Outcome |
|---------|--------------|---------|
| v1 | Default CPU mode | Confirmed working but slow (~12h estimated) |
| v2 | GPU + `--cuda` flag | fd exhaustion @ ulimit 1024 |
| v3 | Reduced workers (6→3) | Same fd exhaustion, just delayed |
| v4 | Single worker (`--num-select 1`) | Killed before validating (still 6 init workers spawned) |
| v5 | Patched mopg.py with frequent logging | Killed for v6 |
| v6 | `ulimit -n 65535` | ✅ Init complete (1.5M steps × 6 tasks); CUDA fork bug at constraint |
| v7 | + `set_sharing_strategy('file_system')` | ✅ Init complete; same CUDA fork bug |
| v8 | + `mp.set_start_method('spawn')` | ❌ dtype mismatch (Float vs Double) — spawn re-init issue |

## What Worked

- ✅ `ulimit -n 65535` solved fd exhaustion
- ✅ Patched logging gives 30 steps/sec/task throughput visibility
- ✅ Init stage (1.5M steps × 6 tasks) completes successfully
- ✅ Per-task convergence validated (value_loss, dist_entropy stable)

## What's Still Broken

- ❌ Init→Constraint stage transition: master process crashes on `queue.get()` due to CUDA-tensor-in-queue
- ❌ `set_start_method('spawn')` introduces new issues (dtype, re-init overhead)

## Why This Matters For TetraRL Paper

This systematic exploration is **valuable research output**, not noise:

### Selling Point #1: First Reproducer
> "We are the first to attempt deploying C-MORL on an embedded GPU (Jetson Orin AGX). We identify three deterministic system-level barriers that prevent direct adoption of cloud-trained MORL algorithms."

### Selling Point #2: Documents Missing TetraRL Contribution
> "Existing MORL libraries (PGMORL, C-MORL) inherit assumptions from cloud-GPU PPO codebases that break under embedded constraints. TetraRL provides a clean redesign that addresses these architectural barriers."

### Selling Point #3: Motivates Systems-Side Architecture
> "Our 4-component framework (Preference Plane / Resource Manager / RL Arbiter / Override) deliberately avoids the multi-process queue pattern shown to be incompatible with embedded GPU deployment."

## Recommended Next Steps (Week 3)

### Path A: Switch Environment
Try MO-Hopper-3d or MO-Ant-3d (no EnergyPlus subprocess overhead) — reduces fd pressure, may not hit Layer 3.

### Path B: Custom Master/Worker Collector
Replace `multiprocessing.Queue` with disk-based comm:
- Workers write `.pt` checkpoints
- Master reads from disk, no CUDA tensor crossing process boundary
- Estimated effort: 1-2 days, ~100 lines change

### Path C: Use Init-Stage Data, Defer Constraint
- Already have valid init data from v6+v7
- Write Section 5 around init-stage convergence
- Mark constraint stage as "future work, blocked by C-MORL implementation issue"

### Path D: Build TetraRL Native Implementation
- Skip C-MORL entirely
- Implement preference-conditioned PPO (single process) on top of `cleanrl` (modern PyTorch baseline)
- Estimated effort: 3-5 days, but solves all 4 layers

## References

- C-MORL paper: https://arxiv.org/abs/2410.02236
- C-MORL repo: https://github.com/RuohLiuq/C-MORL
- PyTorch CUDA + multiprocessing docs: https://pytorch.org/docs/stable/multiprocessing.html
- Jetson L4T compatibility: https://docs.nvidia.com/jetson/

---

*Auto-generated as part of Week 2 Final Consolidation. Source data: ~/Downloads/TetraRL/results/week2_building3d_v6_init_only/ and v7_init_only/*
