# Week 2 Status

## MO-SAC-HER Sanity Check on MountainCar

### v1 (Baseline)
- Status: Tier 3 BROKEN
- Issue: `dir_reg` trended strictly UP.

### v2 (Hyperparameter Fix)
- Changes: `dir_reg_coef` 1.0 -> 5.0, `target_entropy_frac` 1.0 -> 0.5, `alpha` clamp upper 1.0 -> 0.1
- Status: **Tier 2 ACCEPTABLE**
- Results:
  - Algorithm computationally healthy (completed 100k frames, no NaNs).
  - `alpha` stayed within bounded constraints (< 0.1).
  - `dir_reg` trend stabilized (oscillated around 1.55 instead of strictly trending up like v1).
  - However, `HV` remained 0.0 because MountainCar sparse reward means goal was not reached.

**Recommendation:**
Algorithm now computationally healthy AND dir_reg learning works — BUT MountainCar sparse reward means goal not reached. Recommend Walker2d or shaped reward as next step.
