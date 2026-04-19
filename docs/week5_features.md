# Week 5 Features: Tegrastats Daemon + DVFS Controller

**Status**: implemented on branch `week5/tegra-dvfs`.
**Theme** (per `docs/action-plan-weekly.md`): Tegrastats sensor daemon and DVFS control on Orin AGX.

## Modules

### `tetrarl/sys/tegra_daemon.py`

Async sensor daemon that samples tegrastats output at 100 Hz and dispatches
EMA-filtered readings (alpha=0.1) to a callback at 10 Hz. Implements the
kernel/user split pattern from DVFS-DRL-Multitask (2024).

Public API:
- `TegrastatsReading` dataclass — RAM, per-core CPU util/freq, GR3D %,
  GPU/EMC freq, GPU temp, VDD_GPU_SOC mW, VDD_CPU_CV mW.
- `parse_tegrastats_line(line) -> TegrastatsReading | None` — pure parser
  (regex-based) usable from offline log analysis.
- `TegrastatsDaemon(sample_hz, dispatch_hz, ema_alpha, source, on_dispatch)` —
  background thread.
  - `source="auto"`: prefer `tegrastats` binary on PATH, else no-op.
  - `source="binary"`: spawn `tegrastats --interval <ms>` subprocess.
  - `source="file:<path>"`: cycle through a captured fixture (Mac dev / CI).
- `start()` / `stop()` (idempotent) / `latest()` (most recent EMA reading).

Sample tick layout at 100 Hz / 10 Hz:
```
tick:    0  1  2  3  4  5  6  7  8  9  10 11 ...
sample:  R  R  R  R  R  R  R  R  R  R  R  R  ...
dispatch:D                       D                ...
```

### `tetrarl/sys/dvfs.py`

DVFS controller for Orin AGX (cpufreq + devfreq). Mac dev defaults to stub
mode automatically when sysfs nodes are absent.

Public API:
- `DVFSConfig` dataclass — `(cpu_freq_khz, gpu_freq_hz)`.
- `TransitionLatency` dataclass — `(domain, from_freq, to_freq, latency_ms)`.
- `DVFSController(platform, stub, cpu_paths, gpu_paths)`:
  - `available_frequencies() -> {"cpu": [...], "gpu": [...]}`.
  - `set_freq(cpu_idx=None, gpu_idx=None) -> DVFSConfig` — writes
    `scaling_setspeed` for CPU and `min_freq`/`max_freq` for GPU devfreq.
  - `current_state() -> DVFSConfig` — reads `scaling_cur_freq` / `cur_freq`.
  - `profile_transition_latency(domain, n_iters) -> [TransitionLatency]` —
    measures every (from, to) pair `n_iters` times, returns mean ms.

Stub frequency tables are representative Orin AGX values (12 CPU points,
14 GPU points) so the API can be exercised end-to-end on Mac.

## Profiling script

`scripts/profile_orin_dvfs.py` enumerates the frequency table on the host
and runs `profile_transition_latency` for both domains. Emits a CSV
(`docs/orin_dvfs_latency_table.csv`) and a paper-figure-ready markdown
table (`docs/orin_dvfs_latency_table.md`).

## Testing

- Unit tests run on Mac via stub mode: `pytest tests/test_tegra_daemon.py
  tests/test_dvfs.py`.
- Tegrastats daemon is exercised end-to-end via the captured fixture
  `tests/fixtures/tegra_sample.txt` (cycled by the file source).
- Real-mode DVFS is covered via a `tmp_path` test that injects synthetic
  sysfs files and asserts the correct bytes are written.

## Validation criteria (action plan)

| Criterion | Status |
|---|---|
| Tegrastats daemon runs 1 hour without leaks (Δ < 1 MB) | Deferred to Orin smoke |
| DVFS readback within 50 ms of API call | See `orin_dvfs_latency_table.md` |
| Super-block N tuned from measured transition overhead | Pending real-hardware data |

## Rationale notes

- EMA `alpha = 0.1`: matches DVFS-DRL-Multitask 2024 default; smooths
  100 Hz noise without lagging RL controller (10 Hz dispatch ≈ 100 ms
  effective horizon).
- `set_freq(cpu_idx, gpu_idx)`: index-based instead of MHz to keep the
  RL action space discrete and platform-portable (the controller resolves
  index → MHz via the sysfs frequency table).
- Stub mode auto-detect (sysfs presence check) makes the same code work
  unmodified across Mac dev and Orin without configuration flags in
  caller code.
