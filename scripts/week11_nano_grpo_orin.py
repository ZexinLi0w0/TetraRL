#!/usr/bin/env python3
"""Week 11 Nano-GRPO micro-benchmark on Orin AGX.

Implements the appendix experiment specified in
``docs/action-plan-weekly.md`` Week 11 task #2: a minimalist RLHF task that
exercises both PPO-style (with critic) and GRPO-style (Shao 2024,
group-relative baseline) policy updates on a small chat LM, while
recording system-level response curves on a Jetson Orin AGX.

The reward is a synthetic computable function of the generated tokens
(fraction of even token ids). The point is **system-level measurement**,
not alignment quality — the appendix only needs to demonstrate measurable
response curves.

Per-step measurements
---------------------
- generate_ms / forward_ms / backward_ms / optim_ms / total_ms
- GPU power -> energy   (J, integrated from tegrastats VDD_GPU_SOC mW via trapezoid rule)
- RAM used              (MB, from tegrastats RAM_USED_MB; Orin uses unified memory)
- GPU utilization       (%, from tegrastats GR3D_FREQ %)
- torch.cuda.max_memory_allocated   (MB, per pass)

Two passes
----------
- ``with_critic``    — PPO-style: advantage = r - V(s); loss adds MSE(V, r).
- ``without_critic`` — GRPO-style: advantage = (r - mean_g(r)) / std_g(r).

Output
------
- ``<out-dir>/summary.json``           Aggregated stats for both passes.
- ``<out-dir>/per_step_<pass>.csv``    Per-step record (one row per step).

Usage
-----
::

    python scripts/week11_nano_grpo_orin.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --n-steps 200 \\
        --out-dir runs/w11_nano_grpo_orin/

Pass ``--no-tegrastats`` on hosts without ``tegrastats`` (Mac dev).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform as _pyplatform
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

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

from tetrarl.sys.tegra_daemon import (  # noqa: E402
    TegrastatsDaemon,
    TegrastatsReading,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


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
    vdd_gpu_mw: int
    vdd_cpu_mw: int


# ---------------------------------------------------------------------------
# Model bits
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Tiny 2-layer MLP value head over the policy hidden states.

    Used only when running the with-critic pass; the without-critic pass
    eliminates this entirely (and its forward/backward cost) per
    GRPO Shao 2024.
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512, dtype=dtype),
            nn.GELU(),
            nn.Linear(512, 1, dtype=dtype),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)


def synthetic_reward(token_ids: torch.Tensor) -> torch.Tensor:
    """Reward = fraction of generated tokens with even token id.

    Computable, deterministic, and supplies non-trivial gradient signal
    without needing a real preference dataset. Returns a ``[B]`` tensor.
    """
    even = (token_ids % 2 == 0).float()
    return even.mean(dim=-1)


class _SafeLogits(LogitsProcessor):
    """Clamp non-finite / extreme logits so FP16 sampling stays stable.

    After a few PPO/GRPO updates on FP16 weights the LM head can emit
    non-finite or huge logits, which makes ``torch.multinomial`` raise
    ``probability tensor contains inf/nan`` and trips a CUDA assert.
    Clamping to ``[-1e4, 1e4]`` is benign for sampling (softmax saturates
    well before that) and keeps the microbenchmark from crashing.
    """

    def __call__(self, input_ids, scores):  # type: ignore[override]
        scores = scores.masked_fill(~torch.isfinite(scores), -1e4)
        return scores.clamp(min=-1e4, max=1e4)


# ---------------------------------------------------------------------------
# One pass of N GRPO steps
# ---------------------------------------------------------------------------


def run_pass(
    *,
    pass_name: str,
    with_critic: bool,
    model: nn.Module,
    tok,
    fixed_prompt: torch.Tensor,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    args,
    out_dir: Path,
) -> dict:
    print(f"\n[pass={pass_name}] with_critic={with_critic}", flush=True)

    value_head = ValueHead(hidden_size, dtype).to(device)
    value_head.train()
    model.train()

    params: list = list(model.parameters())
    if with_critic:
        params += list(value_head.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    readings: List[TegrastatsReading] = []
    teg = TegrastatsDaemon(
        sample_hz=10.0,
        dispatch_hz=10.0,
        source=("noop" if args.no_tegrastats else "auto"),
        on_dispatch=lambda r: readings.append(r),
    )
    teg.start()
    time.sleep(0.5)  # warm tegrastats

    per_step: List[StepRecord] = []
    t_pass_start = time.perf_counter()

    for step in range(args.n_steps):
        t0 = time.perf_counter()
        # ---- Generate K samples (no grad) ----
        with torch.no_grad():
            input_ids = fixed_prompt.repeat(args.group_size, 1)
            attn_mask = torch.ones_like(input_ids)
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                do_sample=True,
                max_new_tokens=args.gen_len,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
                logits_processor=LogitsProcessorList([_SafeLogits()]),
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gen = time.perf_counter()
        gen_ms = (t_gen - t0) * 1000.0

        # gen: [K, prompt_len + gen_len]
        completion_ids = gen[:, args.prompt_len :]  # [K, gen_len]
        rewards = synthetic_reward(completion_ids)  # [K], float
        rewards = rewards.to(dtype=torch.float32)

        # ---- Forward with grad for logprobs (+ hidden states if with-critic) ----
        t1 = time.perf_counter()
        outputs = model(input_ids=gen, output_hidden_states=with_critic)
        logits = outputs.logits  # [K, T, V]
        # Score the last gen_len positions.
        shift_logits = logits[:, args.prompt_len - 1 : -1, :]
        shift_targets = gen[:, args.prompt_len :]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        target_log_probs = log_probs.gather(
            2, shift_targets.unsqueeze(-1)
        ).squeeze(-1)  # [K, gen_len]
        mean_log_prob = target_log_probs.mean(dim=-1)  # [K]

        if with_critic:
            last_hidden = outputs.hidden_states[-1][
                :, args.prompt_len - 1, :
            ]  # [K, H]
            value = value_head.net(last_hidden).squeeze(-1).float()  # [K]
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

        # Skip optimizer step if loss went non-finite — keeps the
        # microbenchmark from poisoning subsequent generations.
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
        per_step.append(
            StepRecord(
                step=step,
                generate_ms=gen_ms,
                forward_ms=fwd_ms,
                backward_ms=bwd_ms,
                optim_ms=opt_ms,
                total_ms=total_ms,
                loss=float(loss.detach().item()),
                reward_mean=float(rewards.mean().item()),
                reward_std=float(rewards.std().item()),
                gpu_util_pct=float(snap.gr3d_freq_pct),
                gpu_freq_mhz=int(snap.gpu_freq_mhz),
                ram_used_mb=int(snap.ram_used_mb),
                vdd_gpu_mw=int(snap.vdd_gpu_soc_mw),
                vdd_cpu_mw=int(snap.vdd_cpu_cv_mw),
            )
        )

        if step % 10 == 0 or step == args.n_steps - 1:
            print(
                f"  step {step:4d}  total={total_ms:6.1f}ms  "
                f"gen={gen_ms:5.1f}  fwd={fwd_ms:5.1f}  bwd={bwd_ms:5.1f}  "
                f"loss={loss.item():+.4f}  r={rewards.mean().item():.3f}  "
                f"GPU%={snap.gr3d_freq_pct:4.0f}  RAM={snap.ram_used_mb}MB  "
                f"P_GPU={snap.vdd_gpu_soc_mw}mW",
                flush=True,
            )

    teg.stop()
    elapsed_s = time.perf_counter() - t_pass_start

    if device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_alloc_mb = 0.0

    # Energy = trapezoidal integral of power over tegrastats samples
    if len(readings) >= 2:
        ts = np.array([r.ts_monotonic for r in readings])
        p_gpu = np.array([r.vdd_gpu_soc_mw for r in readings]) / 1000.0  # W
        p_cpu = np.array([r.vdd_cpu_cv_mw for r in readings]) / 1000.0
        dt = np.diff(ts)
        energy_gpu_j = float(np.sum(0.5 * (p_gpu[:-1] + p_gpu[1:]) * dt))
        energy_cpu_j = float(np.sum(0.5 * (p_cpu[:-1] + p_cpu[1:]) * dt))
        mem_peak_mb = int(max(r.ram_used_mb for r in readings))
        gpu_util_mean = float(np.mean([r.gr3d_freq_pct for r in readings]))
    else:
        energy_gpu_j = 0.0
        energy_cpu_j = 0.0
        mem_peak_mb = 0
        gpu_util_mean = 0.0

    totals = np.array([s.total_ms for s in per_step])
    gens = np.array([s.generate_ms for s in per_step])
    fwds = np.array([s.forward_ms for s in per_step])
    bwds = np.array([s.backward_ms for s in per_step])
    opts = np.array([s.optim_ms for s in per_step])

    def stats(a: np.ndarray) -> dict:
        return {
            "mean_ms": float(np.mean(a)),
            "p50_ms": float(np.percentile(a, 50)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(np.max(a)),
        }

    result = {
        "n_steps": args.n_steps,
        "group_size": args.group_size,
        "gen_len": args.gen_len,
        "elapsed_s": elapsed_s,
        "energy_gpu_j": energy_gpu_j,
        "energy_cpu_j": energy_cpu_j,
        "energy_total_j": energy_gpu_j + energy_cpu_j,
        "energy_per_step_j": (energy_gpu_j + energy_cpu_j) / max(1, args.n_steps),
        "mem_peak_ram_mb": mem_peak_mb,
        "torch_peak_alloc_mb": peak_alloc_mb,
        "gpu_util_mean_pct": gpu_util_mean,
        "n_tegrastats_samples": len(readings),
        "step_total": stats(totals),
        "step_generate": stats(gens),
        "step_forward": stats(fwds),
        "step_backward": stats(bwds),
        "step_optim": stats(opts),
        "final_reward_mean": float(per_step[-1].reward_mean),
    }

    csv_path = out_dir / f"per_step_{pass_name}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f.name for f in fields(StepRecord)])
        for s in per_step:
            w.writerow([getattr(s, fld.name) for fld in fields(StepRecord)])
    print(
        f"[pass={pass_name}] wrote {csv_path}  "
        f"elapsed={elapsed_s:.1f}s  E_total={result['energy_total_j']:.1f}J  "
        f"mem_peak={mem_peak_mb}MB  torch_peak={peak_alloc_mb:.0f}MB  "
        f"GPU%_mean={gpu_util_mean:.1f}",
        flush=True,
    )

    del value_head, optim, params
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--gen-len", type=int, default=12)
    p.add_argument("--prompt-len", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-tegrastats", action="store_true")
    p.add_argument(
        "--passes",
        default="with_critic,without_critic",
        help="Comma-sep list. Allowed: with_critic, without_critic.",
    )
    p.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Compute dtype on CUDA. bf16 is much more numerically stable for "
        "this microbenchmark; fp16 retained for back-compat.",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    else:
        dtype = torch.float32
    print(f"[init] device={device} dtype={dtype}", flush=True)

    print(f"[init] loading tokenizer from {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"[init] loading model in {dtype}", flush=True)
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    hidden_size = model.config.hidden_size
    print(
        f"[init] model loaded in {time.perf_counter() - t_load:.1f}s; "
        f"hidden={hidden_size}",
        flush=True,
    )

    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    fixed_prompt = torch.tensor(
        [[bos] * args.prompt_len], device=device, dtype=torch.long
    )

    pass_results: dict = {}
    for pass_name in [s.strip() for s in args.passes.split(",")]:
        if pass_name not in ("with_critic", "without_critic"):
            raise ValueError(f"unknown pass {pass_name}")
        with_critic = pass_name == "with_critic"
        pass_results[pass_name] = run_pass(
            pass_name=pass_name,
            with_critic=with_critic,
            model=model,
            tok=tok,
            fixed_prompt=fixed_prompt,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            args=args,
            out_dir=out_dir,
        )

    summary = {
        "model": args.model,
        "n_steps": args.n_steps,
        "group_size": args.group_size,
        "gen_len": args.gen_len,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "host": _pyplatform.uname().node,
        "uname": " ".join(_pyplatform.uname()),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "passes": pass_results,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
