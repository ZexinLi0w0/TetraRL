"""Preference elicitation: convert user-friendly inputs to TetraRL omega vectors.

TetraRL's Preference Plane consumes a 4-D weight vector
``omega = [w_R, w_T, w_M, w_E]`` on the simplex (sum = 1, all >= 0):

- ``w_R``: Reward (task performance)
- ``w_T``: Real-time (tail latency)
- ``w_M``: Memory (RAM footprint)
- ``w_E``: Recharge (energy / power)

End users cannot reasonably hand-craft cardinal weights. This module exposes
three more user-friendly elicitation paths:

1. **Ordinal** — `from_ordinal("R >= T > E > M")` parses a partial order over
   the four objectives and maps it to a simplex point via Borda-style
   rank-positional weights with optional softmax temperature.
2. **Preset profiles** — `from_profile("battery-saver")` returns named
   simplex points for common deployment scenarios (Windows-power-plan style).
3. **Direct (Layer 0)** — for expert users / paper baselines, callers can
   still pass a continuous omega and use `simplex_normalize` to validate it.

See `MEMORY.md` -> "Open design question: User Preference -> omega elicitation"
(2026-04-19) for the design discussion. Layer-3 auto-policy (context-driven
profile switching) is intentionally deferred to follow-up work.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

#: Canonical objective order. Indexes into the omega vector.
OBJECTIVE_NAMES: tuple[str, ...] = ("R", "T", "M", "E")

# Friendly aliases recognised on input. Case-insensitive.
_ALIAS = {
    "R": "R",
    "REWARD": "R",
    "PERF": "R",
    "PERFORMANCE": "R",
    "T": "T",
    "TIME": "T",
    "REAL-TIME": "T",
    "REALTIME": "T",
    "LATENCY": "T",
    "M": "M",
    "MEM": "M",
    "MEMORY": "M",
    "RAM": "M",
    "E": "E",
    "ENERGY": "E",
    "POWER": "E",
    "RECHARGE": "E",
    "BATTERY": "E",
}

#: Preset omega profiles (Layer 2). Order matches OBJECTIVE_NAMES.
PROFILES: dict[str, tuple[float, float, float, float]] = {
    # Balanced default; treats all four equally.
    "balanced": (0.25, 0.25, 0.25, 0.25),
    # Plug-in / wall-power: prioritise reward + latency, ignore battery.
    "performance": (0.5, 0.3, 0.1, 0.1),
    # Latency-critical (real-time control): tail p99 dominates.
    "low-latency": (0.3, 0.5, 0.1, 0.1),
    # Tight memory budget (small Jetson / Nano with co-runners).
    "low-memory": (0.2, 0.3, 0.4, 0.1),
    # Battery-powered field deployment: energy + memory matter most.
    "battery-saver": (0.2, 0.2, 0.3, 0.3),
    # Background / idle: deprioritise reward, smooth out energy.
    "idle": (0.1, 0.2, 0.3, 0.4),
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def list_profiles() -> list[str]:
    """Return the set of available preset profile names (sorted)."""
    return sorted(PROFILES.keys())


def from_profile(name: str) -> tuple[float, ...]:
    """Return the simplex omega for a named profile.

    Raises:
        KeyError: if ``name`` is not a registered profile.
    """
    if not isinstance(name, str):
        raise TypeError(f"profile name must be str, got {type(name).__name__}")
    key = name.strip().lower()
    if key not in PROFILES:
        raise KeyError(
            f"unknown profile {name!r}; choose from {list_profiles()}"
        )
    return PROFILES[key]


def simplex_normalize(omega: Iterable[float]) -> tuple[float, ...]:
    """Project an arbitrary non-negative 4-vector onto the simplex.

    - Negative entries raise ``ValueError`` (caller must clip explicitly if
      they want a different policy).
    - All-zero input falls back to a uniform vector.
    - Length must be exactly 4.
    """
    omega_list = list(omega)
    if len(omega_list) != 4:
        raise ValueError(
            f"omega must have exactly 4 entries, got {len(omega_list)}"
        )
    for i, v in enumerate(omega_list):
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"omega[{i}]={v!r} is not numeric"
            )
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"omega[{i}]={v!r} must be finite")
        if v < 0:
            raise ValueError(
                f"omega[{i}]={v!r} is negative; project explicitly first"
            )
    total = sum(omega_list)
    if total == 0:
        return tuple([0.25] * 4)
    return tuple(v / total for v in omega_list)


def from_ordinal(order: str, *, temperature: float = 1.0) -> tuple[float, ...]:
    """Convert a string-encoded partial order over objectives to omega.

    Examples:
        >>> from_ordinal("R > T > M > E")            # strict order
        (0.4, 0.3, 0.2, 0.1)
        >>> from_ordinal("R >= T > E > M")           # ties merge ranks
        (0.35, 0.35, 0.1, 0.2)
        >>> from_ordinal("Reward > Latency > Energy >= Memory")  # aliases ok
        (0.4, 0.3, 0.15, 0.15)

    Args:
        order: A string like ``"A > B >= C > D"`` listing all four objectives.
            Use ``>`` for strict preference and ``>=`` for ties. Names are
            case-insensitive and accept aliases (Reward/Latency/Memory/Energy).
        temperature: Softmax sharpness for the rank-to-weight mapping.
            ``1.0`` (default) gives the Borda-positional weights
            ``[0.4, 0.3, 0.2, 0.1]``. Lower temperature sharpens (top rank
            dominates), higher temperature smooths toward uniform.

    Raises:
        ValueError: if all four objectives are not mentioned exactly once,
            or if the order string is malformed.
    """
    if not isinstance(order, str):
        raise TypeError(f"order must be str, got {type(order).__name__}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    groups = _parse_ordinal_groups(order)
    # Validate coverage: every canonical objective appears exactly once.
    flat = [obj for grp in groups for obj in grp]
    if sorted(flat) != sorted(OBJECTIVE_NAMES):
        missing = sorted(set(OBJECTIVE_NAMES) - set(flat))
        extra = [x for x in flat if flat.count(x) > 1]
        raise ValueError(
            f"order {order!r} must mention each of {OBJECTIVE_NAMES} "
            f"exactly once (missing={missing}, duplicated={sorted(set(extra))})"
        )

    # Borda-positional base weights (rank 1..4 -> 0.4, 0.3, 0.2, 0.1).
    base = [0.4, 0.3, 0.2, 0.1]
    # Optionally re-shape via softmax(rank / temperature). When temperature=1
    # we keep the linear Borda weights to give a stable, interpretable default.
    if abs(temperature - 1.0) > 1e-9:
        # Use logits = -rank / temperature so rank 1 gets the largest weight.
        logits = [-(i + 1) / temperature for i in range(4)]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        base = [x / s for x in exps]

    # Now assign weights to each rank-group, splitting equally within ties.
    omega = [0.0] * 4
    cursor = 0
    for grp in groups:
        size = len(grp)
        share_total = sum(base[cursor : cursor + size])
        per = share_total / size
        for obj in grp:
            idx = OBJECTIVE_NAMES.index(obj)
            omega[idx] = per
        cursor += size

    return simplex_normalize(omega)


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _normalize_token(tok: str) -> str:
    key = tok.strip().upper().replace("_", "-")
    if key not in _ALIAS:
        raise ValueError(
            f"unknown objective token {tok!r}; "
            f"valid: {sorted(set(_ALIAS.keys()))}"
        )
    return _ALIAS[key]


def _parse_ordinal_groups(order: str) -> list[list[str]]:
    """Split ``"A >= B > C > D"`` into ``[[A, B], [C], [D]]`` (canonical names).

    Tokens between ``>=`` belong to the same rank group; tokens between ``>``
    start a new (lower) rank group.
    """
    cleaned = order.strip()
    if not cleaned:
        raise ValueError("order must be a non-empty string")

    # Replace >= with a sentinel so we can split on > unambiguously.
    SENTINEL = "\x00"  # NUL won't appear in user input
    tmp = re.sub(r">\s*=", SENTINEL, cleaned)
    if "<" in tmp:
        raise ValueError(
            f"order {order!r} uses '<'; only '>' / '>=' are supported"
        )
    rank_chunks = [c.strip() for c in tmp.split(">")]
    groups: list[list[str]] = []
    for chunk in rank_chunks:
        if not chunk:
            raise ValueError(
                f"order {order!r} has an empty rank (stray '>')"
            )
        tied = [t.strip() for t in chunk.split(SENTINEL) if t.strip()]
        if not tied:
            raise ValueError(
                f"order {order!r} has an empty tied group"
            )
        groups.append([_normalize_token(t) for t in tied])
    return groups
