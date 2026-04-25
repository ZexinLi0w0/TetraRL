#!/usr/bin/env python3
"""Multi-tenant Nano-GRPO benchmark harness for Orin AGX (paper §VIII.C / §IX).

Single-tenant Nano-GRPO (``scripts/week11_nano_grpo_orin.py``) shows only
~0.8% latency / 0.25% energy savings between with-critic and without-critic
because there is no contention. Under multi-tenant load (3 concurrent
workloads on a 32GB Orin AGX), the framework's R^4 arbiter (OverrideLayer)
becomes load-bearing.

This script dispatches three modes from a single file via ``--mode``:

- ``grpo`` (main): runs one GRPO/PPO pass (selected by ``--condition``) and
  spawns the perception and llm_inf tenants as subprocesses for the duration.
  Records per-step CSV with override-fire annotations and a summary.json.
- ``perception``: ResNet-18 inference loop at a target FPS.
- ``llm_inf``: TinyLlama generation loop at a target period.

Example invocations
-------------------
::

    # Main multi-tenant run (all three tenants).
    python scripts/p12_multitenant_nano_grpo_orin.py \\
        --mode grpo --condition with_critic --seed 0 --n-steps 200 \\
        --out-dir runs/p12_multitenant_grpo/with_critic_seed0/

    # Mac dev (no tegrastats binary).
    python scripts/p12_multitenant_nano_grpo_orin.py \\
        --mode grpo --condition without_critic --seed 1 --n-steps 50 \\
        --no-tegrastats --no-perception --no-llm-inf \\
        --out-dir /tmp/p12_dev/

    # Perception subprocess (normally spawned by --mode grpo).
    python scripts/p12_multitenant_nano_grpo_orin.py \\
        --mode perception --fps 30 --out-csv /tmp/perception.csv

    # LLM inference subprocess (normally spawned by --mode grpo).
    python scripts/p12_multitenant_nano_grpo_orin.py \\
        --mode llm_inf --period-s 2.0 --out-csv /tmp/llm_inf.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform as _pyplatform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import IO, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from tetrarl.morl.native.override import (  # noqa: E402
    HardwareTelemetry,
    OverrideLayer,
    OverrideThresholds,
)
from tetrarl.sys.tegra_daemon import (  # noqa: E402
    TegrastatsDaemon,
    TegrastatsReading,
)


# Copied from scripts/week11_nano_grpo_orin.py - kept self-contained for clarity.
class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512, dtype=dtype),
            nn.GELU(),
            nn.Linear(512, 1, dtype=dtype),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


# Copied from scripts/week11_nano_grpo_orin.py - kept self-contained for clarity.
def synthetic_reward(token_ids: torch.Tensor) -> torch.Tensor:
    return (token_ids % 2 == 0).float().mean(dim=-1)


# Copied from scripts/week11_nano_grpo_orin.py - kept self-contained for clarity.
class _SafeLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):  # type: ignore[override]
        scores = scores.masked_fill(~torch.isfinite(scores), -1e4)
        return scores.clamp(min=-1e4, max=1e4)


@dataclass
class StepRecord:
    step: int
    generate_ms: float
    forward_ms: float
    backward_ms: float
    optim_ms: float
    total_ms: float
    loss: float
    reward_mean: float
    reward_std: float
    gpu_util_pct: float
    gpu_freq_mhz: int
    ram_used_mb: int
    ram_total_mb: int
    vdd_gpu_mw: int
    vdd_cpu_mw: int
    override_fired: int
    override_reason: str


def spawn_tenant(args_list: list[str], log_path: Path) -> Tuple[subprocess.Popen, IO]:
    log_f = log_path.open("w")
    proc = subprocess.Popen(
        args_list, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True,
    )
    return proc, log_f


def kill_tenant(
    proc: subprocess.Popen, log_f: IO, name: str, timeout: float = 10.0
) -> None:
    try:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(
                    f"[main] {name} did not exit in {timeout}s, sending SIGKILL",
                    flush=True,
                )
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        try:
            log_f.close()
        except Exception:
            pass


def _install_stop_handler() -> dict:
    stop = {"v": False}

    def _stop(signum, frame):  # noqa: ARG001
        stop["v"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return stop


def _resolve_dtype(name: str, on_cuda: bool) -> torch.dtype:
    if not on_cuda:
        return torch.float32
    return torch.bfloat16 if name == "bf16" else torch.float16


# ---------------------------------------------------------------------------
# Mode 2: perception
# ---------------------------------------------------------------------------


def run_perception(args: argparse.Namespace) -> None:
    import torchvision.models as tvm

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device.type == "cuda")
    print(
        f"[perception] device={device} dtype={dtype} fps={args.fps} "
        f"batch={args.batch} input_size={args.input_size}",
        flush=True,
    )
    model = tvm.resnet18(weights=None).to(device=device, dtype=dtype)
    model.eval()
    inp = torch.zeros(
        (args.batch, 3, args.input_size, args.input_size),
        device=device, dtype=dtype,
    )
    f = out_csv.open("w", newline="")
    w = csv.writer(f)
    w.writerow(["frame_idx", "timestamp_s", "latency_ms"])
    stop = _install_stop_handler()
    period = 1.0 / max(1e-6, args.fps)
    frame_idx = 0
    try:
        while not stop["v"]:
            t0 = time.perf_counter()
            inp.normal_()
            with torch.no_grad():
                _ = model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            w.writerow([frame_idx, f"{t1:.6f}", f"{latency_ms:.3f}"])
            if frame_idx % 10 == 0:
                f.flush()
            frame_idx += 1
            sleep_s = period - (t1 - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            f.flush()
        finally:
            f.close()
        print(f"[perception] exited after {frame_idx} frames", flush=True)


# ---------------------------------------------------------------------------
# Mode 3: llm_inf
# ---------------------------------------------------------------------------


def run_llm_inf(args: argparse.Namespace) -> None:
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device.type == "cuda")
    print(
        f"[llm_inf] device={device} dtype={dtype} period_s={args.period_s} "
        f"gen_len={args.gen_len} model={args.model}",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    enc = tok(args.prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    f = out_csv.open("w", newline="")
    w = csv.writer(f)
    w.writerow(["prompt_idx", "timestamp_s", "latency_ms", "n_new_tokens"])
    stop = _install_stop_handler()
    prompt_idx = 0
    try:
        while not stop["v"]:
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=args.gen_len,
                    do_sample=True, top_p=0.9, temperature=1.0,
                    pad_token_id=tok.eos_token_id,
                    logits_processor=LogitsProcessorList([_SafeLogits()]),
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            n_new = int(out.shape[1] - input_ids.shape[1])
            w.writerow([prompt_idx, f"{t1:.6f}", f"{latency_ms:.3f}", n_new])
            f.flush()
            prompt_idx += 1
            sleep_s = args.period_s - (t1 - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            f.flush()
        finally:
            f.close()
        print(f"[llm_inf] exited after {prompt_idx} prompts", flush=True)


# ---------------------------------------------------------------------------
# Mode 1: grpo
# ---------------------------------------------------------------------------


def run_grpo_pass(
    *, condition: str, model: nn.Module, tok, fixed_prompt: torch.Tensor,
    hidden_size: int, device: torch.device, dtype: torch.dtype,
    args: argparse.Namespace, out_dir: Path, teg: TegrastatsDaemon,
    readings: List[TegrastatsReading], override_layer: OverrideLayer,
) -> Tuple[dict, bool]:
    with_critic = condition == "with_critic"
    print(
        f"[grpo] condition={condition} with_critic={with_critic} "
        f"n_steps={args.n_steps}",
        flush=True,
    )
    value_head = ValueHead(hidden_size, dtype).to(device)
    value_head.train()
    model.train()
    params: list = list(model.parameters())
    if with_critic:
        params += list(value_head.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    per_step: List[StepRecord] = []
    t_pass_start = time.perf_counter()
    oom = False

    try:
        for step in range(args.n_steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                input_ids = fixed_prompt.repeat(args.group_size, 1)
                attn_mask = torch.ones_like(input_ids)
                gen = model.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    do_sample=True, max_new_tokens=args.gen_len,
                    top_p=0.9, temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                    logits_processor=LogitsProcessorList([_SafeLogits()]),
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gen = time.perf_counter()
            gen_ms = (t_gen - t0) * 1000.0

            completion_ids = gen[:, args.prompt_len:]
            rewards = synthetic_reward(completion_ids).to(dtype=torch.float32)

            t1 = time.perf_counter()
            outputs = model(input_ids=gen, output_hidden_states=with_critic)
            shift_logits = outputs.logits[:, args.prompt_len - 1: -1, :]
            shift_targets = gen[:, args.prompt_len:]
            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            target_log_probs = log_probs.gather(
                2, shift_targets.unsqueeze(-1)
            ).squeeze(-1)
            mean_log_prob = target_log_probs.mean(dim=-1)

            if with_critic:
                last_hidden = outputs.hidden_states[-1][:, args.prompt_len - 1, :]
                value = value_head.net(last_hidden).squeeze(-1).float()
                advantages = (rewards - value.detach()).float()
                value_loss = F.mse_loss(value, rewards)
            else:
                r_mean = rewards.mean()
                r_std = rewards.std() + 1e-8
                advantages = ((rewards - r_mean) / r_std).float()
                value_loss = torch.tensor(0.0, device=device)

            policy_loss = -(advantages * mean_log_prob).mean()
            loss = policy_loss + 0.1 * value_loss
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.perf_counter()
            fwd_ms = (t_fwd - t1) * 1000.0

            loss_finite = bool(torch.isfinite(loss).item())
            optim.zero_grad(set_to_none=True)
            if loss_finite:
                loss.backward()
            else:
                for p_ in params:
                    if p_.grad is None:
                        p_.grad = torch.zeros_like(p_)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_bwd = time.perf_counter()
            bwd_ms = (t_bwd - t_fwd) * 1000.0

            torch.nn.utils.clip_grad_norm_(params, 0.5)
            if loss_finite:
                optim.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_opt = time.perf_counter()
            opt_ms = (t_opt - t_bwd) * 1000.0
            total_ms = (t_opt - t0) * 1000.0

            snap = teg.latest() or TegrastatsReading()
            ram_total = snap.ram_total_mb if snap.ram_total_mb > 0 else 32000
            mem_util = snap.ram_used_mb / ram_total if ram_total > 0 else 0.0
            telemetry = HardwareTelemetry(
                latency_ema_ms=total_ms, memory_util=mem_util
            )
            fired, _ = override_layer.step(telemetry)
            reason = "; ".join(override_layer.last_reasons) if fired else ""

            per_step.append(StepRecord(
                step=step, generate_ms=gen_ms, forward_ms=fwd_ms,
                backward_ms=bwd_ms, optim_ms=opt_ms, total_ms=total_ms,
                loss=float(loss.detach().item()),
                reward_mean=float(rewards.mean().item()),
                reward_std=float(rewards.std().item()),
                gpu_util_pct=float(snap.gr3d_freq_pct),
                gpu_freq_mhz=int(snap.gpu_freq_mhz),
                ram_used_mb=int(snap.ram_used_mb),
                ram_total_mb=int(snap.ram_total_mb),
                vdd_gpu_mw=int(snap.vdd_gpu_soc_mw),
                vdd_cpu_mw=int(snap.vdd_cpu_cv_mw),
                override_fired=int(bool(fired)),
                override_reason=reason,
            ))

            if step % 10 == 0 or step == args.n_steps - 1:
                fire_marker = " [OVERRIDE]" if fired else ""
                print(
                    f"  step {step:4d}  total={total_ms:6.1f}ms  "
                    f"gen={gen_ms:5.1f}  fwd={fwd_ms:5.1f}  bwd={bwd_ms:5.1f}  "
                    f"loss={loss.item():+.4f}  r={rewards.mean().item():.3f}  "
                    f"GPU%={snap.gr3d_freq_pct:4.0f}  RAM={snap.ram_used_mb}MB  "
                    f"P_GPU={snap.vdd_gpu_soc_mw}mW{fire_marker}",
                    flush=True,
                )
    except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
        oom = True
        print(f"[grpo] CUDA OOM at step={len(per_step)}: {e}", flush=True)

    elapsed_s = time.perf_counter() - t_pass_start
    if device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_alloc_mb = 0.0

    # Snapshot before iteration: TegrastatsDaemon's join has a 2s timeout, and
    # the daemon thread can append one or two more readings after teg.stop()
    # returns. Without snapshotting, dt and p_cpu disagree on length.
    snap = list(readings)
    if len(snap) >= 2:
        ts = np.array([r.ts_monotonic for r in snap])
        p_gpu = np.array([r.vdd_gpu_soc_mw for r in snap]) / 1000.0
        p_cpu = np.array([r.vdd_cpu_cv_mw for r in snap]) / 1000.0
        dt = np.diff(ts)
        energy_gpu_j = float(np.sum(0.5 * (p_gpu[:-1] + p_gpu[1:]) * dt))
        energy_cpu_j = float(np.sum(0.5 * (p_cpu[:-1] + p_cpu[1:]) * dt))
        mem_peak_mb = int(max(r.ram_used_mb for r in snap))
        gpu_util_mean = float(np.mean([r.gr3d_freq_pct for r in snap]))
    else:
        energy_gpu_j = energy_cpu_j = 0.0
        mem_peak_mb = 0
        gpu_util_mean = 0.0

    if per_step:
        totals = np.array([s.total_ms for s in per_step])
        gens = np.array([s.generate_ms for s in per_step])
        fwds = np.array([s.forward_ms for s in per_step])
        bwds = np.array([s.backward_ms for s in per_step])
        opts = np.array([s.optim_ms for s in per_step])
        final_reward_mean = float(per_step[-1].reward_mean)
    else:
        totals = gens = fwds = bwds = opts = np.array([0.0])
        final_reward_mean = 0.0

    def stats(a: np.ndarray) -> dict:
        return {
            "mean_ms": float(np.mean(a)),
            "p50_ms": float(np.percentile(a, 50)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(np.max(a)),
        }

    csv_path = out_dir / "per_step_grpo.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([fld.name for fld in fields(StepRecord)])
        for s in per_step:
            w.writerow([getattr(s, fld.name) for fld in fields(StepRecord)])
    print(
        f"[grpo] wrote {csv_path}  elapsed={elapsed_s:.1f}s  "
        f"E_total={energy_gpu_j + energy_cpu_j:.1f}J  mem_peak={mem_peak_mb}MB  "
        f"torch_peak={peak_alloc_mb:.0f}MB  GPU%_mean={gpu_util_mean:.1f}",
        flush=True,
    )

    result = {
        "n_steps_completed": len(per_step),
        "n_steps_target": args.n_steps,
        "group_size": args.group_size,
        "gen_len": args.gen_len,
        "elapsed_s": elapsed_s,
        "energy_gpu_j": energy_gpu_j,
        "energy_cpu_j": energy_cpu_j,
        "energy_total_j": energy_gpu_j + energy_cpu_j,
        "energy_per_step_j": (energy_gpu_j + energy_cpu_j) / max(1, len(per_step)),
        "mem_peak_ram_mb": mem_peak_mb,
        "torch_peak_alloc_mb": peak_alloc_mb,
        "gpu_util_mean_pct": gpu_util_mean,
        "step_total": stats(totals),
        "step_generate": stats(gens),
        "step_forward": stats(fwds),
        "step_backward": stats(bwds),
        "step_optim": stats(opts),
        "final_reward_mean": final_reward_mean,
        "oom": oom,
        "n_tegrastats_samples": len(snap),
    }
    del value_head, optim, params
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, oom


def run_grpo_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    self_path = Path(sys.argv[0]).resolve()

    spawned: list[Tuple[str, subprocess.Popen, IO]] = []
    tenants_enabled = {"perception": False, "llm_inf": False}

    if args.enable_perception:
        cmd = [
            sys.executable, str(self_path),
            "--mode", "perception",
            "--fps", str(args.perception_fps),
            "--out-csv", str(out_dir / "perception.csv"),
            "--dtype", args.dtype,
        ]
        proc, log_f = spawn_tenant(cmd, out_dir / "perception_stdout.log")
        spawned.append(("perception", proc, log_f))
        tenants_enabled["perception"] = True
        print(f"[main] spawned perception pid={proc.pid}", flush=True)

    if args.enable_llm_inf:
        cmd = [
            sys.executable, str(self_path),
            "--mode", "llm_inf",
            "--period-s", str(args.llm_inf_period_s),
            "--model", args.model,
            "--gen-len", str(args.gen_len),
            "--out-csv", str(out_dir / "llm_inf.csv"),
            "--dtype", args.dtype,
        ]
        proc, log_f = spawn_tenant(cmd, out_dir / "llm_inf_stdout.log")
        spawned.append(("llm_inf", proc, log_f))
        tenants_enabled["llm_inf"] = True
        print(f"[main] spawned llm_inf pid={proc.pid}", flush=True)

    print(f"[main] warm-up sleep {args.warmup_s}s for tenant model load", flush=True)
    time.sleep(args.warmup_s)
    for name, proc, _ in spawned:
        if proc.poll() is not None:
            print(
                f"[main] WARNING tenant {name} exited early "
                f"with rc={proc.returncode}; disabling",
                flush=True,
            )
            tenants_enabled[name] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device.type == "cuda")
    print(f"[main] grpo device={device} dtype={dtype}", flush=True)

    readings: List[TegrastatsReading] = []
    teg = TegrastatsDaemon(
        sample_hz=10.0, dispatch_hz=10.0,
        source=("noop" if args.no_tegrastats else "auto"),
        on_dispatch=lambda r: readings.append(r),
    )
    teg.start()
    time.sleep(0.5)
    override_layer = OverrideLayer(
        thresholds=OverrideThresholds(
            max_latency_ms=args.override_max_latency_ms,
            max_memory_util=args.override_max_mem_util,
        ),
        fallback_action=0,
    )

    print(f"[main] loading tokenizer from {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    print(f"[main] loading grpo model in {dtype}", flush=True)
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    hidden_size = model.config.hidden_size
    print(
        f"[main] grpo model loaded in {time.perf_counter() - t_load:.1f}s; "
        f"hidden={hidden_size}",
        flush=True,
    )
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    fixed_prompt = torch.tensor(
        [[bos] * args.prompt_len], device=device, dtype=torch.long
    )

    pass_result: dict = {}
    oom = False
    try:
        pass_result, oom = run_grpo_pass(
            condition=args.condition, model=model, tok=tok,
            fixed_prompt=fixed_prompt, hidden_size=hidden_size,
            device=device, dtype=dtype, args=args, out_dir=out_dir,
            teg=teg, readings=readings, override_layer=override_layer,
        )
    finally:
        teg.stop()
        for name, proc, log_f in spawned:
            kill_tenant(proc, log_f, name)
            print(f"[main] tenant {name} stopped (rc={proc.returncode})", flush=True)

    summary = {
        "condition": args.condition,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype),
        "host": _pyplatform.uname().node,
        "uname": " ".join(_pyplatform.uname()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "tenants_enabled": tenants_enabled,
        "warmup_s": args.warmup_s,
        "n_tegrastats_samples": int(pass_result.get("n_tegrastats_samples", 0)),
        "grpo": pass_result,
        "override": {
            "fire_count": int(override_layer.fire_count),
            "fire_rate": float(override_layer.fire_count) / max(1, args.n_steps),
            "thresholds": {
                "max_latency_ms": args.override_max_latency_ms,
                "max_memory_util": args.override_max_mem_util,
            },
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[main] wrote {summary_path}", flush=True)
    return 2 if oom else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bool_pair(parser: argparse.ArgumentParser, dest: str, default: bool) -> None:
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        f"--enable-{dest.replace('_', '-')}",
        dest=f"enable_{dest}", action="store_true",
    )
    g.add_argument(
        f"--no-{dest.replace('_', '-')}",
        dest=f"enable_{dest}", action="store_false",
    )
    parser.set_defaults(**{f"enable_{dest}": default})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    p.add_argument("--mode", choices=("grpo", "perception", "llm_inf"), required=True)

    p.add_argument("--condition", choices=("with_critic", "without_critic"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--out-dir")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--gen-len", type=int, default=12)
    p.add_argument("--prompt-len", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--no-tegrastats", action="store_true")
    p.add_argument("--override-max-latency-ms", type=float, default=2000.0)
    p.add_argument("--override-max-mem-util", type=float, default=0.90)
    p.add_argument("--warmup-s", type=float, default=5.0)
    p.add_argument("--perception-fps", type=float, default=30.0)
    p.add_argument("--llm-inf-period-s", type=float, default=2.0)
    _bool_pair(p, "perception", True)
    _bool_pair(p, "llm_inf", True)

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--input-size", type=int, default=224)

    p.add_argument("--period-s", type=float, default=2.0)
    p.add_argument("--prompt", default="Hello, how are you today?")
    p.add_argument("--out-csv")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "perception":
        if not args.out_csv:
            raise SystemExit("--out-csv is required for --mode perception")
        run_perception(args)
        return
    if args.mode == "llm_inf":
        if not args.out_csv:
            raise SystemExit("--out-csv is required for --mode llm_inf")
        run_llm_inf(args)
        return
    if args.mode == "grpo":
        if not args.condition:
            raise SystemExit("--condition is required for --mode grpo")
        if not args.out_dir:
            raise SystemExit("--out-dir is required for --mode grpo")
        rc = run_grpo_main(args)
        if rc != 0:
            sys.exit(rc)
        return


if __name__ == "__main__":
    main()
