# Week 2: C-MORL Building-3d On-Device Training Study

**Status:** Init Stage Complete; Constraint Stage Deferred Due to Systems Bug  
**Date:** 2026-04-18  
**Platform:** NVIDIA Jetson Orin AGX 64GB (single GPU)

---

## 1. Experimental Setup

**Hardware.** All experiments were conducted on a single NVIDIA Jetson Orin AGX developer kit equipped with 64 GB of unified memory, an Ampere-architecture GPU, and an ARM Cortex-A78AE CPU complex. The device operates under a unified memory architecture in which the CPU and GPU share the same physical DRAM, a constraint that distinguishes it from discrete-GPU server environments and introduces additional fragility in CUDA multiprocessing semantics.

**Software.** The software stack comprises Python 3.10, PyTorch compiled against CUDA 12.6, MuJoCo 3.x as the physics backend, and mo-gymnasium 1.1.0 for multi-objective environment wrappers.

**Algorithm.** We employ C-MORL (Liu et al., ICLR 2025), a constrained multi-objective reinforcement learning algorithm. The implementation is vendored locally at `tetrarl/morl/c_morl/` to allow source-level modifications for on-device deployment.

**Environment.** The target environment is Building-3d from the mo-gymnasium suite. The init stage trains 6 independent tasks, each corresponding to one warmup preference vector. Each task runs for 488 iterations with 512 environment steps per iteration, yielding 249,856 steps per task.

---

## 2. Pipeline Modifications

The following modifications were applied to the upstream C-MORL codebase in order to run on the Orin AGX platform. We document both successful and unsuccessful interventions for reproducibility.

### 2.1 File Descriptor Limit Adjustment

The default Linux `ulimit -n` of 1024 open file descriptors proved insufficient. During multiprocessing rollout collection, the process exhausted available file descriptors, causing silent failures in worker spawning. The fix was applied at the shell level prior to training:

```bash
ulimit -n 65535
```

This resolved file descriptor exhaustion but did not address the CUDA multiprocessing crash described in Section 5.

### 2.2 Step-Level Logging and Progress Dumps (v6/v7)

The upstream `mopg.py` module was patched to emit frequent step-level diagnostic logs and to serialize a `progress.json` snapshot every 10 iterations. This modification enabled fine-grained convergence tracking and provided crash-recovery checkpoints.

### 2.3 File-System Sharing Strategy (v7, Ineffective)

In v7, the following call was inserted before worker process creation:

```python
torch.multiprocessing.set_sharing_strategy('file_system')
```

The intent was to bypass POSIX shared-memory limits by routing inter-process tensor communication through the filesystem. This change had no measurable effect on the CUDA crash at the init-to-constraint stage transition, nor did it alter training dynamics (see Section 4).

### 2.4 Spawn Start Method (v8, Regressed)

In v8, the multiprocessing start method was switched from `fork` to `spawn`:

```python
mp.set_start_method('spawn')
```

While `spawn` avoids inheriting the parent's CUDA context, it caused a dtype regression: environment observation and action spaces were re-initialized under `spawn` semantics with `torch.float32` (Float) rather than the expected `torch.float64` (Double). This dtype mismatch propagated through the policy network and caused runtime errors, rendering v8 non-functional.

---

## 3. Results

All 6 init-stage tasks completed the full 488/488 iterations, producing 249,856 steps per task and 1,499,136 total environment steps across all tasks.

### 3.1 v7 Final Metrics (Primary)

| Task | Iterations | Steps   | Wall Time (s) | Value Loss | Action Loss | Dist Entropy |
|------|-----------|---------|---------------|------------|-------------|-------------|
| 0    | 488       | 249,856 | 7,409         | 0.0473     | -0.0006     | 33.60       |
| 1    | 488       | 249,856 | 7,409         | 0.0572     | -0.0005     | 29.82       |
| 2    | 488       | 249,856 | 7,440         | 0.0680     | -0.0008     | 25.17       |
| 3    | 488       | 249,856 | 7,444         | 0.0492     | -0.0006     | 33.07       |
| 4    | 488       | 249,856 | 7,491         | 0.0659     | -0.0007     | 29.06       |
| 5    | 488       | 249,856 | 7,477         | 0.0532     | -0.0006     | 32.02       |

### 3.2 v6 Final Metrics (Comparison Baseline)

| Task | Iterations | Steps   | Wall Time (s) | Value Loss | Action Loss | Dist Entropy |
|------|-----------|---------|---------------|------------|-------------|-------------|
| 0    | 488       | 249,856 | 7,403         | 0.0414     | -0.0007     | 33.26       |
| 1    | 488       | 249,856 | 7,403         | 0.0638     | -0.0006     | 30.39       |
| 2    | 488       | 249,856 | 7,485         | 0.0640     | -0.0006     | 24.95       |
| 3    | 488       | 249,856 | 7,470         | 0.0489     | -0.0006     | 33.03       |
| 4    | 488       | 249,856 | 7,460         | 0.0632     | -0.0006     | 29.48       |
| 5    | 488       | 249,856 | 7,520         | 0.0520     | -0.0006     | 32.29       |

### 3.3 Throughput

The average training throughput across all tasks and both versions is approximately **33.6 steps/sec per task**. Wall-clock times are consistent across tasks (7,400--7,520 seconds), indicating stable GPU utilization throughout the init stage.

---

## 4. Convergence Analysis

The following observations are drawn from the training curves recorded across v6 and v7.

**Value loss.** Value loss decreases steadily across all 6 tasks, converging to the 0.04--0.07 range by the final iteration. Tasks 0 and 3 achieve the lowest terminal value loss (approximately 0.04--0.05), while tasks 2 and 4 exhibit slightly higher terminal values (approximately 0.06--0.07), consistent with the higher difficulty of their respective preference vectors.

**Action loss.** Action loss stabilizes near zero across all tasks, with terminal values clustered around -0.0006. This indicates that the policy gradient updates have reached a near-stationary point with respect to the advantage function.

**Distribution entropy.** Dist entropy stabilizes at task-specific plateaus ranging from approximately 25 to 34. The variation across tasks reflects meaningful policy differentiation: each preference vector induces a distinct action distribution, with lower-entropy policies (Task 2, approximately 25) concentrating probability mass on fewer actions and higher-entropy policies (Task 0, approximately 34) maintaining broader exploration. This is expected behavior for multi-objective warm-start training.

**Cross-version consistency.** The v6 and v7 training curves are nearly identical across all metrics and all tasks. This confirms that the `file_system` sharing strategy change introduced in v7 had no effect on training dynamics, isolating its (null) impact to the IPC layer only.

**Reference figures:**
- `docs/figures/week2_v6v7_convergence.png` -- Convergence curves for all metrics across tasks
- `docs/figures/week2_v6v7_completion_table.png` -- Completion bar chart
- `docs/figures/week2_v6v7_overhead.png` -- Throughput analysis

---

## 5. The Systems Bug: CUDA Fork Multiprocessing Failure

This section documents a deterministic systems-level failure that prevents the C-MORL pipeline from transitioning beyond the init stage on the Orin AGX platform. We consider this a key research finding for the TetraRL project, as it exposes a real-world deployment barrier for on-device multi-objective RL.

### 5.1 Symptom

The crash occurs deterministically at the init-to-constraint stage transition. After all 6 init-stage tasks complete their full 488 iterations, the master process attempts to retrieve results from a multiprocessing queue. The call to `results_queue.get()` raises a `torch.AcceleratorError` with the message `CUDA error: invalid argument`.

### 5.2 Root Cause

The failure originates from the interaction between PyTorch's CUDA runtime and Python's `multiprocessing` module operating in `fork` mode. When a process is forked, the child inherits the parent's CUDA context. Upon completion, the child serializes CUDA tensors into the multiprocessing queue. When the parent subsequently deserializes these tensors via `_new_shared_cuda`, the CUDA runtime rejects the operation because the shared CUDA memory handles from the child process are no longer valid in the parent's context.

This is a known class of bugs in the PyTorch ecosystem. The PyTorch documentation explicitly warns against using `fork` with CUDA, but the C-MORL codebase relies on `fork`-mode multiprocessing for its worker-based rollout collector.

### 5.3 Exact Traceback

The following traceback is reproduced identically in both v6 and v7:

```
File ".../morl.py", line 91, in run
    rl_results = results_queue.get()
File ".../multiprocessing/queues.py", line 122, in get
    return _ForkingPickler.loads(res)
File ".../torch/multiprocessing/reductions.py", line 181, in rebuild_cuda_tensor
    storage = storage_cls._new_shared_cuda(
torch.AcceleratorError: CUDA error: invalid argument
```

### 5.4 Attempted Fixes

Four interventions were attempted, none of which resolved the CUDA crash:

| Attempt | Intervention | Outcome |
|---------|-------------|---------|
| 1 | `ulimit -n 65535` | Resolved file descriptor exhaustion; CUDA crash persists |
| 2 | `torch.multiprocessing.set_sharing_strategy('file_system')` (v7) | No effect on crash or training dynamics |
| 3 | `mp.set_start_method('spawn')` (v8) | Caused dtype regression (Float vs Double) due to spawn re-initialization of environment observation/action spaces |
| 4 | Single-worker configuration | Still crashes; the issue is in CUDA context sharing, not worker count |

### 5.5 Impact

The constraint stage cannot execute, which prevents C-MORL from computing constraint-aware policies. However, the init-stage data -- comprising pre-trained policies for each of the 6 preference vectors -- is complete and independently usable.

### 5.6 Proposed Resolution

The recommended fix, to be pursued in subsequent work, is to redesign the master/worker collector architecture to use **disk-based serialization**. Under this design, worker processes write their results as `.pt` checkpoint files to a shared filesystem path, and the master process reads and loads them after worker completion. This eliminates the need to pass CUDA tensors through multiprocessing queues entirely, circumventing the shared-memory CUDA context invalidation at its source.

---

## 6. Significance and Discussion

**On-device MORL feasibility.** The successful completion of 1,499,136 total environment steps across 6 preference-vector tasks demonstrates that on-device multi-objective reinforcement learning training is feasible at meaningful scale on edge GPU hardware. The Orin AGX sustained approximately 33.6 steps/sec per task with stable GPU utilization over multi-hour training runs.

**Edge-specific deployment challenges.** The CUDA fork multiprocessing failure documented in Section 5 represents a class of systems-level issues that arise specifically in on-device deployment and are not typically observed in desktop or server environments. The Orin AGX's unified memory architecture and single-GPU constraint make CUDA fork semantics particularly fragile, as there is no discrete GPU memory boundary to naturally isolate parent and child CUDA contexts.

**Motivation for systems contributions.** This finding directly motivates the systems-side contributions of the TetraRL project. Resource-aware scheduling and IPC redesign for edge GPU platforms are not merely performance optimizations but correctness requirements: without addressing the CUDA multiprocessing barrier, constrained MORL algorithms cannot complete their full training pipeline on edge devices.

**Value of init-stage data.** The init-stage data itself retains practical value. The 6 pre-trained policy initializations, one per preference vector, can serve as warm-start checkpoints for constraint optimization once the IPC architecture is redesigned. This decoupled workflow (init on device, constraint with fixed IPC) may itself prove to be a viable deployment pattern for edge MORL.

---

## 7. Data Artifacts

The following artifacts are archived in the repository:

| Artifact | Path | Description |
|----------|------|-------------|
| v6 training data | `results/week2_building3d_v6_init_only/` | Progress JSONs and full training log for v6 |
| v7 training data | `results/week2_building3d_v7_init_only/` | Progress JSONs and full training log for v7 |
| Convergence figure | `docs/figures/week2_v6v7_convergence.png` | Value loss, action loss, and entropy curves for all tasks across v6 and v7 |
| Completion figure | `docs/figures/week2_v6v7_completion_table.png` | Bar chart showing iteration completion status for all tasks |
| Throughput figure | `docs/figures/week2_v6v7_overhead.png` | Wall-clock time and throughput analysis across tasks and versions |
