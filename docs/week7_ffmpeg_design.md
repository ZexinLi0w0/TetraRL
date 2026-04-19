# Week 7 Task 5 — FFmpeg Co-runner Interference Harness

## R³ protocol citation

The original protocol comes from Li et al., **"R³: Learning to Schedule
on Heterogeneous Edge Platforms with Reinforcement Learning"**, *RTSS '23*,
Fig. 15. They quantify training-step latency interference by running the
RL training workload (1) **alone**, then (2) **alongside an `ffmpeg`
process decoding an H.264 stream** at 720p / 1080p / 2K, and report the
99th-percentile training-step latency under each condition. The
multiplicative slowdown vs the alone baseline is the headline number
that motivates their hardware-aware scheduling decisions.

This harness reproduces that protocol: a TetraRL framework + Gym
training loop is the workload-under-test, and an `ffmpeg` subprocess
is the controlled co-runner. The harness records per-step wall time
via `time.perf_counter_ns()` (monotonic, nanosecond resolution),
serialises samples to JSONL, and renders a Markdown table with p50 /
p90 / p99 / p99.9 plus the `slowdown_p99` column relative to the
`none` baseline.

## Diff vs R³

* **Synthetic source by default.** R³ used a real H.264 file. We
  default to `ffmpeg -f lavfi -i testsrc=size=WxH:rate=30 -f null -`
  so the harness has no external video file dependency and is
  reproducible from any clean checkout. The synthetic source still
  exercises the encoder/null-mux path and produces measurable CPU
  pressure (we see ~3.7x p99 slowdown on Mac at 720p with 200 steps).
* **Optional real-video path on Orin.** Pass `--video PATH` to use
  `-stream_loop -1 -i PATH` instead, so the Orin run can compare
  apples-to-apples against R³'s measurements when a real H.264
  sample is available on the device.
* **Resolution ladder.** Same as R³: `720p` (1280x720), `1080p`
  (1920x1080), `2K` (2560x1440). The mapping is in
  `FFmpegInterference._resolution_to_wh`.
* **Workload.** R³ used their full RL training stack; we use the
  Week 6 TetraRLFramework with a `RandomArbiter` + `StubTelemetrySource`
  + stub `DVFSController`. This isolates the framework wiring overhead
  + Gym env step cost from any policy-network forward time, so the
  measured slowdown is a clean signal of OS / kernel / hardware
  contention from `ffmpeg`.

## Why hardware decode matters on Orin

Pass `--hw-decode` to request `-c:v h264_nvv4l2dec` (NVIDIA's V4L2
H.264 hardware decoder). This routes the decode workload through
**NVDEC** instead of the CPU, which:

1. **Frees the CPU** so the contention picked up by the workload's
   p99 latency is GPU/memory-bandwidth contention, not CPU contention.
   This decomposes the interference signal so we can attribute
   slowdown to the right resource.
2. **Stresses the GPU subsystem** that the RL Arbiter would also use
   in a full deployment, modelling the worst-case where both
   workloads contend for the same shared accelerator.

The flag is **silently dropped** when the running `ffmpeg` build
does not advertise `nvv4l2dec` / `nvdec` / `cuda` in its
`-hwaccels` output — for example on Mac, where only `videotoolbox`
is listed. This keeps the same CLI usable on both platforms;
`FFmpegInterference._hw_decode_available()` is the gate.

It is also dropped when `video_path` is `None` (synthetic `lavfi`
testsrc), because lavfi generates raw frames and there is no H.264
bitstream to decode.

## How to read the percentile table

```
| condition | n | p50_ms | p90_ms | p99_ms | p99.9_ms | slowdown_p99 |
|-----------|---|--------|--------|--------|----------|--------------|
| none      | 200 | 0.004 | 0.005 | 0.012 | 0.040 | 1.00x |
| 720p      | 200 | 0.017 | 0.024 | 0.044 | 0.147 | 3.70x |
```

* **`condition`** — the co-runner workload during the run
  (`none` is the alone baseline).
* **`n`** — number of recorded per-step samples.
* **`p50` / `p90` / `p99` / `p99.9`** — per-step wall time (ms),
  computed via type-7 linear interpolation over sorted samples
  (matches numpy's default `np.percentile` so cross-validation is
  trivial). `p99` and `p99.9` are the tail metrics that matter for
  real-time guarantees.
* **`slowdown_p99`** — `condition.p99 / baseline.p99`. A value of
  `2.50x` means the 99th-percentile latency is 2.5x worse under
  this co-runner than alone. The baseline is the recorder keyed
  by `'none'`; if it is absent the column reads `N/A`.

A higher `slowdown_p99` at higher resolutions is the signature of
real interference. A flat slowdown column means either the workload
is too small to detect interference or the co-runner is not actually
using the contended resource (check that `ffmpeg` actually spawned
with `ps`).

## What is NOT measured here

* **Energy under FFmpeg co-run.** This harness only measures
  per-step latency. Energy/power profiling under the same
  co-runner conditions is a separate Week 7 task and will be
  driven from the Orin `tegrastats` daemon on hardware.
* **Sustained throughput.** We measure tail latency, not steady-
  state throughput. A workload that meets its tail-latency budget
  but degrades throughput badly will not show up in this table.
* **Memory footprint.** `ffmpeg` allocates working buffers for the
  decoder; this is captured indirectly via `latency_ms` if the
  pressure pushes the workload into swap, but is not reported as
  a separate column.

## File layout

* `tetrarl/eval/ffmpeg_interference.py` — `LatencyRecorder`,
  `FFmpegInterference`, `run_workload`, `summarize`,
  `ffmpeg_available`.
* `scripts/week7_ffmpeg_corunner.py` — CLI driver.
* `tests/test_ffmpeg_interference.py` — 15 unit + 1 spawn test
  (the spawn test skips cleanly when `ffmpeg` is absent).
* `runs/week7_ffmpeg_<unix_ts>/` — per-condition `*.jsonl` plus
  `summary.md`. Sample run from Mac (200 steps, 720p):

  ```
  | condition | n | p50_ms | p90_ms | p99_ms | p99.9_ms | slowdown_p99 |
  |-----------|---|--------|--------|--------|----------|--------------|
  | none | 200 | 0.004 | 0.005 | 0.012 | 0.040 | 1.00x |
  | 720p | 200 | 0.017 | 0.024 | 0.044 | 0.147 | 3.70x |
  ```
