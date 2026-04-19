# Week 7 Task 6: Jetson Nano DVFS / Tegrastats Port

Status: implemented on branch `week7/nano-dvfs-port` (Mac stub layer only; physical Nano sysfs writes deferred).

Theme (per `docs/action-plan-weekly.md`, Week 7 Track B): extend the Week 5 tegrastats daemon and DVFS controller from Orin AGX to Jetson Nano so the multi-platform evaluation pipeline can run on both supported devices without per-platform code forks.

## Motivation

Week 5 shipped `tetrarl/sys/tegra_daemon.py` and `tetrarl/sys/dvfs.py` for Orin AGX with hardcoded CPU/GPU frequency tables, devfreq sysfs paths, and a single set of tegrastats power-rail regexes (`VDD_GPU_SOC` / `VDD_CPU_CV`). Extending the multi-platform scope to Jetson Nano (Track B in the action plan; Xavier NX has been removed) without copy-pasting two near-identical modules requires factoring the per-platform divergence out of the controllers. Task 6 introduces a `Platform` enum and a `PlatformProfile` registry in `tetrarl/sys/platforms.py`; the existing controllers now dispatch through that registry. Physical Nano hardware validation (real sysfs writes, real tegrastats subprocess, transition-latency table) is explicitly deferred — see Section 7.

## Platform Diff

Values below are read directly from `tetrarl/sys/platforms.py` (the registry) — do not edit this table without re-checking the source.

| Dimension | Orin AGX | Jetson Nano (4 GB) |
|---|---|---|
| CPU frequency table | 13 points, 115.2..2188.8 MHz | 15 points, 102.0..1479.0 MHz |
| GPU frequency table | 14 points, 114.75..1377.0 MHz | 12 points, 76.8..921.6 MHz |
| CPU sysfs setspeed template | `/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed` | `/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed` |
| GPU sysfs setspeed path | `/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/userspace/set_freq` | `/sys/devices/57000000.gpu/devfreq/57000000.gpu/userspace/set_freq` |
| Tegrastats field layout key | `"orin"` | `"nano"` |
| Default EMA alpha | `0.1` (DVFS-DRL-Multitask 2024 default) | `0.2` (older sensor cadence + 4-core CPU -> noisier samples) |
| `mem_total_mb` | 32768 (32 GiB) | 4096 (4 GiB) |
| Tegrastats power rails | `VDD_GPU_SOC` / `VDD_CPU_CV` (mW) | `POM_5V_GPU` / `POM_5V_CPU` (mW) |

Notes:
- The CPU setspeed template is identical because both Jetson families expose the standard cpufreq layout under `cpu{N}/cpufreq/`. Only the GPU devfreq node address (`17000000.gpu` vs `57000000.gpu`) and the absence of `/platform/` in the Nano path actually differ.
- The Orin AGX stub table here (13 CPU points, 14 GPU points) is a representative subset of the 29/11 points actually exposed on the measured orin1 unit (per the orin1 DVFS root memory). The stub keeps the table small for fast unit tests while preserving the boundary frequencies; real-mode reads come straight from `scaling_available_frequencies`.

## Why a registry pattern

Two existing call sites consume per-platform constants — `tetrarl/sys/dvfs.py` (frequency tables, sysfs paths) and `tetrarl/sys/tegra_daemon.py` (power-rail regex set, EMA default). The Week 9 hardware-emergency override layer (`tetrarl/sys/override.py` per the action plan) will be a third. Localizing per-platform divergence to a single file (`tetrarl/sys/platforms.py`) avoids the alternative of either branching on a string inside each controller or copy-pasting two near-identical modules per Jetson family.

Concretely:
- Adding a future Jetson SKU = one new `PlatformProfile` entry plus one `Platform` enum member. No changes to controller code, no changes to call sites, no changes to tests that did not assert about the new SKU.
- The `Platform` enum is `str`-valued, so callers can pass either `Platform.NANO` or the literal `"nano"` and `get_profile()` normalizes — useful for CLI integration (see `scripts/profile_orin_dvfs.py --platform nano`) without sacrificing type safety inside the library.
- `PlatformProfile` is a `@dataclass(frozen=True)` so the registry cannot be mutated at runtime by mistake.
- Backward compatible with Week 5 callers: `DVFSController()` with no args still defaults to `Platform.ORIN_AGX`, and `TegrastatsDaemon()` with no args still picks the Orin layout and `alpha=0.1`. See Section 8.

## Tegrastats field layout dispatch

The Week 5 `parse_tegrastats_line(line)` function gained a single new keyword argument:

```python
def parse_tegrastats_line(
    line: str, layout: Literal["orin", "nano"] = "orin"
) -> Optional[TegrastatsReading]: ...
```

When `layout == "nano"` the parser swaps in the `POM_5V_GPU` / `POM_5V_CPU` regexes; otherwise it uses the original `VDD_GPU_SOC` / `VDD_CPU_CV` regexes. The output `TegrastatsReading` dataclass is unchanged — both paths populate `vdd_gpu_soc_mw` and `vdd_cpu_cv_mw` so downstream consumers (the EMA blender, RL state encoders, paper plots) see uniform field names regardless of source rail. The other regexes (RAM, CPU per-core util/freq, `GR3D_FREQ`, `EMC_FREQ`, GPU temp) are common to both layouts and stay layout-agnostic.

The `layout` token is the `Literal["orin", "nano"]` value of `PlatformProfile.tegrastats_field_layout`; `TegrastatsDaemon._loop()` plumbs it via `self.profile.tegrastats_field_layout` so the per-tick parser call uses the right regex set without having to re-check the platform on every line.

## EMA alpha tuning

Week 5 (`docs/week5_features.md`) sets `alpha = 0.1` as the Orin default, citing DVFS-DRL-Multitask 2024 and the 100 Hz sample / 10 Hz dispatch tick layout (≈100 ms effective horizon). The Nano profile bumps the default to `alpha = 0.2`:

- Nano's tegrastats sampling cadence is closer to its native 5-10 Hz rate (vs Orin's higher rate); aggressive smoothing would lag the controller.
- Nano power-rail readings come from the `POM` I2C INA monitors and are coarser in mW resolution than Orin's per-core integrated readings, so the per-sample noise floor is higher and a slightly more reactive blend tracks transients better.

This is only a default. Callers can still pass `ema_alpha=...` explicitly to override; `test_explicit_ema_alpha_overrides_profile_default` pins that behavior.

## What is NOT validated yet (deferred)

- Real `set_freq` writes to `/sys/devices/57000000.gpu/devfreq/57000000.gpu/userspace/set_freq` on a physical Nano. Same constraints as Orin per the orin1 DVFS root memory: needs Nano hardware, sudo, and `governor=userspace` set on every cpu before `scaling_setspeed` writes are honored.
- Real `tegrastats` binary subprocess on Nano. Only fixture-driven parsing is exercised on Mac via `tests/fixtures/tegrastats_nano_sample.txt` (30 lines covering an idle -> ramp -> peak -> idle profile, with non-zero `POM_5V_*` rails throughout the active phase).
- DVFS transition-latency table for Nano. `scripts/profile_orin_dvfs.py --platform nano --allow-real-dvfs` will produce `docs/nano_dvfs_latency_table.{csv,md}` once a Nano is plugged in. The script defaults to stub mode and refuses to write sysfs unless `--allow-real-dvfs` is also passed, so accidentally running it on a Jetson without root will not repin the governor.
- End-to-end TetraRLFramework run on Nano. Per the action plan, this is a Week 9/10 task — the Week 7 deliverable is the abstraction, not the on-device benchmark.

## Migration note

Backward compatibility was an explicit goal so that no Week 5 caller needs to change:

- `DVFSController(stub=True)` (no `platform=` kwarg) still selects the Orin AGX defaults; the constructor signature is `platform: Union[Platform, str] = Platform.ORIN_AGX`.
- `STUB_CPU_FREQS_KHZ` and `STUB_GPU_FREQS_HZ` are still exported from `tetrarl/sys/dvfs.py`; they are now thin re-aliases of `PLATFORM_PROFILES[Platform.ORIN_AGX].cpu_freqs_hz` / `.gpu_freqs_hz` so any external code or test importing them keeps working byte-for-byte.
- `parse_tegrastats_line(line)` (single-argument call) still defaults to `layout="orin"`, so any offline log-analysis script written against the Week 5 API is unaffected.
- The `platform` kwarg accepts both `Platform` enum members and string values (`"orin_agx"`, `"nano"`) for ergonomic CLI integration; `get_profile()` raises `KeyError` on unknown strings (`test_get_profile_unknown_platform_raises` pins the contract using `"xavier_nx"` as the canonical removed platform).
- `DVFSController.platform` (the attribute) preserves whatever string form the caller passed in, so any pre-refactor code that introspected `ctrl.platform == "orin_agx"` still works.

## Test coverage summary

In-scope unit-test additions for this task:

| File | New tests | Total in file |
|---|---|---|
| `tests/test_platforms.py` | 18 (new file) | 18 |
| `tests/test_dvfs.py` | 7 | 20 |
| `tests/test_tegra_daemon.py` | 7 | 19 |

Aggregate: 209 passing in the repo (up from 177 before this task). The three scoped test modules together collect 57 tests and all 57 pass on Mac (`pytest tests/test_platforms.py tests/test_dvfs.py tests/test_tegra_daemon.py`).

No new dependencies were introduced; `pyproject.toml` is unchanged on this branch (`git diff main..week7/nano-dvfs-port -- pyproject.toml` is empty).

## Files touched

- `tetrarl/sys/platforms.py` — new module: `Platform` enum, `PlatformProfile` dataclass, `PLATFORM_PROFILES` registry, `get_profile()` resolver.
- `tetrarl/sys/dvfs.py` — refactored: `DVFSController.__init__` now takes `platform`, derives sysfs paths from the profile, retains `STUB_CPU_FREQS_KHZ` / `STUB_GPU_FREQS_HZ` aliases.
- `tetrarl/sys/tegra_daemon.py` — refactored: `parse_tegrastats_line(line, layout=...)` dispatches power-rail regexes, `TegrastatsDaemon.__init__` takes `platform` and reads `default_ema_alpha` from the profile.
- `tests/fixtures/tegrastats_nano_sample.txt` — new 30-line Nano tegrastats capture (POM_5V_* rails, 4-core CPU layout, idle -> peak -> idle).
- `tests/test_platforms.py` — new file, 18 tests pinning the registry contract.
- `tests/test_dvfs.py` — 7 new Nano-focused tests (incl. backward-compat default check, string-arg acceptance, distinct freq tables, Nano devfreq node assertion).
- `tests/test_tegra_daemon.py` — 7 new Nano-focused tests (Nano-layout parse, Orin-layout-on-Nano-fixture power=0 check, profile-driven alpha defaults, end-to-end fixture dispatch with non-zero power, explicit-alpha override, `platform_name` attribute).
- `scripts/week7_nano_smoke.py` — new Mac-runnable smoke that drives both `DVFSController(platform=Nano)` and `TegrastatsDaemon(platform=Nano, source="file:...")` against the captured fixture; emits a side-by-side top-frequency table.
- `scripts/profile_orin_dvfs.py` — gained `--platform {orin_agx,nano}` and `--allow-real-dvfs` flags; output filenames are now platform-prefixed (`<platform>_dvfs_latency_table.{csv,md}`); auto-falls-back to stub mode if real sysfs is detected without the explicit opt-in.
