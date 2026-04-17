# References — Third-party reference implementations

This directory contains read-only copies of third-party codebases used as references during TetraRL development. These are NOT part of TetraRL itself; they are vendored here for convenience so the implementation can be cross-checked against the original authors' code.

## Inventory

| Subdirectory | Source | License | Used For |
|---|---|---|---|
| `pd-morl-official/` | [tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm) (ICLR 2023) | See upstream | Cross-check `tetrarl/morl/agents/pd_morl.py` against the authors' reference implementation |

## Usage policy

- Treat everything in this directory as read-only. Do not modify these files in place.
- When borrowing a snippet into TetraRL, copy it into the appropriate `tetrarl/` module and cite the source file path in a comment.
- These references are excluded from `pytest` discovery, `ruff` lint, and CI.

## Re-cloning

If a reference becomes stale, re-clone with `gh repo clone` and remove the inner `.git` directory to keep this repository's working tree clean.
