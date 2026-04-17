# SustainGym Installation Notes

**Date:** 2026-04-17  
**Environment:** macOS Darwin 25.3.0, Apple M4, Python 3.14.3, CPU-only venv

## Installation Steps

```bash
pip install --upgrade pip setuptools wheel
pip install 'mo-gymnasium==1.1.0'
pip install sustaingym
pip install scikit-learn pvlib   # missing runtime deps
```

## Version Snapshot

| Package        | Version  |
|----------------|----------|
| sustaingym     | 0.1.7    |
| gymnasium      | 0.28.1   |
| mo-gymnasium   | 1.1.0    |
| scikit-learn   | 1.8.0    |
| pvlib          | 0.15.0   |
| pettingzoo     | 1.24.3   |

## Known Issues

1. **Gymnasium version conflict:** sustaingym 0.1.7 pins `gymnasium==0.28.*`, which
   downgrades from the 1.x series required by `tetrarl`. The C-MORL smoke tests
   (11/11) still pass, but this conflict must be resolved before production use.
   Options: (a) vendor sustaingym's BuildingEnv with gym 1.x compat shim,
   (b) use separate venvs, (c) wait for sustaingym update.

2. **Missing optional deps:** sustaingym does not declare `scikit-learn` and `pvlib`
   as hard dependencies, but `BuildingEnv` fails to import without them.

3. **pvlib deprecation warning:** `pvlib.iotools.parse_epw` is deprecated in
   pvlib 0.13+; sustaingym should migrate to `read_epw`.

## BuildingEnv Smoke Test Results

- **Building:** ApartmentHighRise (90 zones)
- **Action space:** Box(-1, 1, shape=(90,))
- **Observation space:** Box(shape=(94,))
- **Reward:** single scalar (not multi-objective natively)
- **reward_space:** N/A (needs MO wrapper for MORL use)
- **3 steps:** reward range [-30.6, -38.9] — env is functional

## Next Steps

- Wrap BuildingEnv with multi-objective reward (energy + comfort objectives)
- Register as mo-gymnasium compatible environment
- Run C-MORL training on wrapped BuildingEnv (Task 2 Wed/Thu)
