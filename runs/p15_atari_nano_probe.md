# P15 Atari Replay-Buffer Memory Headroom Probe — Orin Nano (8 GB)

**Run date**: 2026-04-25
**Host**: nano2 (Orin Nano 8 GB shared CPU+GPU)
**Reproduce**: `python scripts/p15_probe_atari_nano_memory.py --capacities 1000,5000,10000,50000 --device cpu --out-md <path>`

## Caveats

- `torch.cuda.is_available()` returned **False** on nano2 because the installed NVIDIA driver (11040) is older than what torch 2.11.0+cu130 expects. The probe ran with `--device cpu`. For the Phase 4 Atari Nano runs we will need to either downgrade torch to a CUDA-11-compatible build or upgrade the driver; **deferred to Phase 4 hardware setup**.
- Because Orin Nano uses unified memory (GPU and CPU share the same DRAM), the replay-buffer headroom measurement is dominated by the numpy `uint8` buffer footprint regardless of device. The Nature-DQN CNN itself is ~5 MB so the device choice does not materially change the measurement.
- An earlier version of the probe used `np.zeros` + a single-byte touch, which on Linux maps every page to a single shared zero page (calloc) and grossly under-counts physical RAM. The probe was fixed (commit pending in this branch) to call `obs.fill(1)` / `next_obs.fill(1)` so every page is committed. The numbers below are post-fix.

## Results

Config: frame_stack=4, frame_h=84, frame_w=84, device=cpu

| capacity | replay_alloc_mb_estimate | mem_avail_before_mb | mem_avail_after_mb | delta_mb | fits_in_8gb |
| --- | --- | --- | --- | --- | --- |
| 1000  | 53.8   | 6330.9 | 6256.7 | 74.2   | True |
| 5000  | 269.2  | 6319.7 | 6034.1 | 285.6  | True |
| 10000 | 538.3  | 6310.8 | 5764.7 | 546.1  | True |
| 50000 | 2691.7 | 6310.6 | 3607.3 | 2703.3 | True |

Replay-buffer footprint scales linearly with capacity as expected. The 50 000-capacity buffer consumes ~2.7 GB and leaves ~3.6 GB headroom on Nano — well over the 1 GB threshold the probe uses to declare "comfortable fit".

## Recommendation

Adopt **`replay_capacity = 50_000`** for Phase 4 Atari Nano cells. This matches the standard small-Atari replay size, fits comfortably in 8 GB, and matches the budget specified in `spec-phase0.md` §1 (CartPole/Atari Nano budget: 50k frames × 1 seed × 21 cells = ~32 h).

Capacities above 50 000 were not probed in this run because:
- `100 000` would consume ~5.4 GB (still under 8 GB but eats the headroom that DQN training itself needs for its target net + optimizer state + Atari env emulator);
- `200 000` would need ~11 GB and is guaranteed to OOM on an 8 GB Nano.

If a future Phase 4 wants to push capacity higher, re-run the probe with capacities including the actual DQN training resident set (target net + optimizer state) for a more honest headroom measurement.
