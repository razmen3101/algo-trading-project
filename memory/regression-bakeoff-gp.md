---
name: regression-bakeoff-gp
description: Regression_Bakeoff results — GP added with decay weighting; NS_spline wins, GP close 2nd
metadata:
  type: project
---

Regression_Bakeoff.ipynb compares cross-sectional shadow-price regressors
(ElasticNet, NS_spline, Hybrid, GP) over 10 sector anchors, rolling+expanding,
ElasticNet top-5 feature selection upstream in `walk_forward`.

Added a **GP** model factory (`fit_gp`): kernel = `DotProduct + RBF` (WhiteKernel
dropped). sklearn GPR has no `sample_weight`, so exp-decay recency weighting is
injected as per-point observation noise `alpha_i = GP_NOISE / w_i` (GP/GLS
equivalent of weighted least squares). Knobs: `GP_NOISE=0.1`, `GP_MAX_TRAIN=400`
(caps O(n^3)). Run cell is incremental + has `RECOMPUTE_MODELS` to refresh one model.

Finding (run 2026-06-12, TEST): **NS_spline·rolling wins every metric**
(R²₁=+0.219, R²₂₀=−0.367, rz_std=1.226) and is cheapest. Decay weighting lifted
**GP·rolling** to a clear 2nd (R²₁ +0.103→+0.173, gap to spline 0.116→0.046).
GP beats linear ElasticNet → real non-linear signal, but spline still wins on
accuracy+cost. Next idea: tune `GP_NOISE`/`length_scale` to try to pass the spline.

Caveats: `pygam` not installed in this env so **GAM is excluded** from the bakeoff
(it's in the Idea notebook via repaired cache). No model gives a stationary
residual OOS (`frac_adf_stationary=0`, adf_p≈1) — property of price-level
cross-sectional regression. See [[regression-idea-bakeoff]] context.
