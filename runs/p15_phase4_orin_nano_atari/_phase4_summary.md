# P15 Phase 4 — Atari Breakout × Orin Nano (25 cells, status snapshot)

**Phase**: 4 of 6 (Atari Breakout × Orin Nano, reduced budget)
**Spec**: `spec-phase4.md`
**Runner**: `scripts/p15_unified_runner.py` at commit 0222374
**Hardware**: nano2 (Orin Nano 8 GB shared CPU+GPU)
**Date**: 2026-04-26

## ⚠️ BLOCKER — read this first

The 21 "active" cells in this matrix have **`status = "DEFERRED"`**, not `"COMPLETED"`.

The unified runner contains a hardcoded DEFERRED branch for `--env breakout`
(`scripts/p15_unified_runner.py` lines 163–172):

> "Atari training requires CNN backbone — wired in Phase 1"

…but the CNN backbone was never wired. `tetrarl/morl/algos.py` is MLP-only:
the DRL classes (`DQNAlgo`, `DDQNAlgo`, `C51Algo`, `A2CAlgo`, `PPOAlgo`) flatten
their input via `np.prod(obs_shape)` into a 64-hidden MLP — fine for CartPole's
4-dim observation, but cannot consume Atari's 210×160×3 frames (which would
require a Nature-DQN style Conv2d encoder + frame-skip + frame-stack +
84×84 grayscale + uint8 replay packing).

Phase 2's own result report explicitly anticipated this:

> "Real GPU peak will appear in Phase 4 (Atari × Nano) once we resolve the
>  driver mismatch (or use a pinned-cuda torch wheel) and the CNN backbone
>  actually allocates CUDA tensors."  — `result-phase2.md` line 131

Neither prerequisite was done before Phase 4 launched. Result: every "active"
cell exits inside the runner's DEFERRED branch in <1s, writing a stub
summary.json with `status="DEFERRED"`, `wall_time_s=0.0`, and no metric fields.

**Phase 3 (Atari × Orin AGX, on orin1) is running concurrently with the same
unified runner and will hit the identical blocker — its 63 "active" cells will
also produce DEFERRED stubs.**

## 25-cell status

|         | DQN      | DDQN     | C51      | A2C      | PPO      |
|---------|----------|----------|----------|----------|----------|
| MAX-A   | DEFERRED | DEFERRED | DEFERRED | DEFERRED | DEFERRED |
| MAX-P   | DEFERRED | DEFERRED | DEFERRED | DEFERRED | DEFERRED |
| R³      | DEFERRED | DEFERRED | DEFERRED | SKIPPED  | SKIPPED  |
| DuoJoule| DEFERRED | DEFERRED | DEFERRED | SKIPPED  | SKIPPED  |
| TetraRL | DEFERRED | DEFERRED | DEFERRED | DEFERRED | DEFERRED |

= 21 DEFERRED (active-but-stubbed) + 4 SKIPPED (compat-incompatible) = 25 total.

## Wall

Total wall: **145 s** (~2.4 min) for all 25 cells, serial. Each cell exits
inside the DEFERRED branch in 5–6 s. The spec's "~32 h" estimate assumed real
training; that estimate is irrelevant to this run.

## Validator

- `--expected-cells 21` (the value the spec author probably meant — active
   count): **FAILED** (`expected 21 COMPLETED cells, found 0`).
- `--expected-cells 0` (the only value satisfiable by the current runner):
   **ALL OK** (`COMPLETED=0  SKIPPED/DEFERRED=25  other=0  total cells=25`).

The DEFERRED cells are correctly schema-formed (status + reason), so the
matrix-mode validator would also accept them as legal SKIPPED equivalents
per `_check_skipped` (validator lines 102–109, 209–214).

## What's needed to unblock

To produce real per-cell stats on Atari × Orin Nano:
1. Wire a Nature-DQN style CNN backbone in `tetrarl/morl/algos.py` for the
   off-policy algos (DQN/DDQN/C51) and a shared CNN trunk for the on-policy
   actor-critics (A2C/PPO).
2. Add ALE preprocessing wrappers (frame-skip 4, frame-stack 4, 84×84
   grayscale, uint8 replay packing) — Phase 0 probe already validated 50 000
   capacity fits in 8 GB.
3. Resolve the CUDA driver / torch mismatch (`torch.cuda.is_available()=False`
   on nano2 due to NVIDIA driver 11040 vs torch 2.11.0+cu130) — or live with
   CPU-only training (each Atari cell will then take many hours, not the
   ~1.5 h the spec assumed).
4. Remove the runner's hardcoded `if args.env == "breakout": return DEFERRED`
   branch.

This is at least one full prep phase of work; it was not in scope for the
"adapt scripts/p15_run_phase2.sh" instruction in spec-phase4.md.
