> **Archived project record, copied unchanged apart from this note.** File and folder names in it (for example `work/...`, `models_v6/`, `figs/`, research scripts) refer to the original project tree, which was archived outside this submission. They are not needed to run anything here. The numbers are the measured values; the machine-readable copies are in `results/*.csv` and `results/*.json`.

# Ceiling diagnostic (read-only; 2026-11-06)

**Inputs:**
- cached v3 dev OOF scores (Optuna #56, model seed 0, split seeds 1729/2026/31415, mean over repeats; dev OOF AUC 0.8495);
- the P1 pre/post mechanism fits on the 4,000-series stratified dev subsample (2,000 positives);
- the 20,000-series synthetic ceiling set from `src/dgp.py`.

Code: `src/ceiling_diag.py`. Numbers: `work/v3/dgp/ceiling_diag.json`, `ceiling_diag_coordAUC.json`. Nothing shipped was
modified.

## 1. Ceiling from the score distribution
Each positive's score is converted to u = F_neg(score), its rank within the negative-class score distribution. If a
fraction pi of positives is indistinguishable from negatives, their u are Uniform(0,1). The detectable rest sits high.
- **Storey-type estimate:** pi = P(u <= lambda)/lambda, with lambda = 0.1 as the primary value.
- **Constrained mixture:** Uniform + Beta(a > 1, b).
- **Implied maximum AUC:** 1 − pi/2, assuming the detectable part could be ranked perfectly.

| set | n_pos | v3 AUC | pi (Storey, lambda = 0.1) [95% bootstrap CI] | implied max AUC | pi range over lambda = 0.05 to 0.5 |
|---|---|---|---|---|---|
| all dev | 9,600 | 0.8495 | **0.114 [0.091, 0.133]** | **0.943 [0.933, 0.955]** | 0.077 to 0.230 (max AUC 0.962 to 0.885) |
| pre acf1 > 0.6 | 1,614 | 0.7862 | 0.174 [0.111, 0.260] | 0.913 [0.870, 0.944] | 0.136 to 0.341 |
| n_total <= 350 | 1,175 | 0.7816 | 0.272 [0.153, 0.375] | 0.864 [0.813, 0.923] | 0.306 to 0.378 |
| n_post <= 150 | 882 | 0.7905 | 0.227 [0.136, 0.374] | 0.887 [0.813, 0.932] | 0.249 to 0.327 |

- **The mixture fit is not usable on the real scores.** It collapses to pi = 0 on all dev (the Beta absorbs the
  uniform part) and is unstable on the slices (acf > 0.6: pi = 0.64 with Beta(745, 0.58)). Only the Storey numbers are
  reported as estimates.
- **Sanity check on synthetic data** (20k set; a known share of no-effect positives was injected by relabelling
  negatives):
  - With estimated-LR scores, Storey recovered an injected share of 0.300 as **0.298** and 0.333 as **0.337**.
  - With oracle scores it gave 0.325 and 0.326 for 0.300 and 0.333.
  - The mixture estimator overshot (0.385 and 0.420) and is biased.
  - Note: the second injection level was capped at half the negatives, so it tested 0.333, not 0.6.
- **What this bound means.** 0.943 is NOT an attainable ceiling. It holds only if every positive that v3 does not score
  like a negative could be ranked perfectly. With pi = 0.114, v3's ranking of the "detectable" part is
  (0.8495 − 0.057)/0.886 = **0.894**. The score distribution cannot say how much of the remaining 0.106 is irreducible
  overlap and how much is estimation loss.

## 2. Mechanism-identified vs unexplained positives (subsample, v3 OOF vs ALL 9,600 dev negatives)
**Identified** means at least one mechanism coordinate lies outside the label-0 [0.5%, 99.5%] range of its own base
stratum (sine / stationary / other):
- non-sine coordinates: level, log innovation-sd, AR(1) coefficient;
- sine coordinates: noise scale, period, amplitude.

Negatives flagged by the same rule (false-positive rate): 3.2%.

| group | n | share of positives | v3 AUC vs all negatives | mean OOF score |
|---|---|---|---|---|
| negatives (all dev) | 9,600 | - | - | 0.236 |
| **identified** | 644 | 32.2% | **0.984** | 0.950 |
| **unexplained** | 1,356 | 67.8% | **0.776** | 0.554 |
| all subsample positives | 2,000 | 100% | 0.843 | 0.681 |

By mechanism (identified):
| level (198) | scale (346) | AR phi (59) | period (26) | amplitude (15) |
|---|---|---|---|---|
| 0.984 | 0.995 | 0.947 | 0.975 | 0.923 |

By grid level (groups with n ≥ 15):
| level 1σ (103) | level 2σ (95) | var ×2 (202) | var ×4 (144) | AR Δ 0.3 (51) | period 0.3 (19) |
|---|---|---|---|---|---|
| 0.977 | 0.991 | 0.991 | 0.9996 | 0.961 | 0.991 |

- **Missing grid levels:** no identified group sits at level 0.5σ, AR Δ 0.1 or AR Δ 0.2. Shifts that small fall inside
  the label-0 range and land in the "unexplained" group.
- **By stratum:**

  | stratum | share identified | identified AUC | unexplained AUC |
  |---|---|---|---|
  | stationary | 36% | 0.988 | 0.771 |
  | other (random walk, trend…) | 15% | 0.964 | 0.810 |
  | sine | 34% | 0.979 | 0.748 |

- **Headroom arithmetic.** Overall AUC = 0.322·A_identified + 0.678·A_unexplained.
  - The identified group is saturated: at most about +0.005 of overall AUC is left there.
  - **0.87 overall needs A_unexplained of about 0.816, and 0.88 needs about 0.831**, against v3's measured 0.776, that
    is +0.04 to +0.055 on the weak-effect group.

## 3. What the unexplained 67.8% look like
Each pre-vs-post parameter difference is standardised by the label-0 robust SD (an estimation-noise proxy). The
unexplained positives are compared with the negatives:

| coordinate | robust SD ratio (unexplained / neg) | frac \|z\| > 2, unexplained vs neg | p (\|z\|, MWU) | p (signed) |
|---|---|---|---|---|
| level | **2.40** | 0.466 vs 0.253 | 3e-23 | 0.67 |
| log innovation sd | **1.28** | 0.166 vs 0.090 | 2e-11 | 0.048 (mean +0.20) |
| kurtosis | **1.29** | 0.257 vs 0.193 | 4e-7 | 0.79 |
| log t-nu | **1.41** | 0.369 vs 0.314 | 8e-6 | 0.49 |
| AR(1) phi | 1.01 | 0.114 vs 0.087 | 0.79 | 0.36 |
| AR(2) phi2 | 0.96 | 0.055 vs 0.060 | 0.44 | 0.39 |
| trend slope | 1.02 | 0.167 vs 0.144 | 0.51 | 0.036 (mean +0.13) |
| sine period / amplitude | 1.08 / 1.05 | 0.204 vs 0.170 / 0.142 vs 0.148 | 0.30 / 0.48 | 0.47 / 0.95 |
| GARCH alpha+beta | 0.96 | 0.031 vs 0.033 | 0.60 | 0.12 |
| sq-resid acf1 / acf sum10 | 1.04 / 0.94 | 0.091 vs 0.102 / 0.104 vs 0.104 | 1.0 / 0.25 | 0.42 / 0.71 |
| spike rate (post − pre) | - | nonzero 19.2% vs 18.6% | 0.64 | 0.37 |

**(a) Centred on zero, or shifted below detection?** Shifted below detection. The level and scale differences are
spread wider than the null (SD ratios 2.40 and 1.28), but most stay inside the label-0 range. The label-0 range is
wide because random-walk, trend and sine bases wander.

**(b) Any coordinate with a consistent signed shift?** None strong. The level change is symmetric. Variance increases
very slightly more often than it decreases (p = 0.048). Trend shows p = 0.036, which is weak and unconfirmed. The
changes are magnitude changes in random directions, mainly level, scale and tail shape (kurtosis, t-nu).

**(c) Gradual rather than at the boundary?**
- A freely estimated mean break falls within ±5% of n of the true boundary for **28.5%** of unexplained positives,
  against 11.6% of negatives (chance about 12.5%) and 40.2% of identified positives.
- A freely estimated variance break does so for 21.3%, against 10.5% of negatives and 64.1% of identified positives.
- A linear ramp fits the post segment better than a step for 42.8% of unexplained positives, against 54.1% of
  negatives and 33.4% of identified positives.

So the changes are at the boundary and step-like. There is no ramp signature.

**(d) Spike/outlier rate pre vs post?** No difference (p = 0.64).

**(e) Volatility persistence change not captured by GARCH?** None. The changes in GARCH alpha+beta and in
squared-residual autocorrelation (lag 1 and sum of lags 1–10) match the negatives (p = 0.25 to 1.0).

**What v3 extracts.** A 5-fold logistic regression on all 12 standardised |differences| ranks unexplained positives
against negatives at **0.638** (best single coordinate: level 0.580). On the same series v3 scores **0.779**. v3
already gets more out of these series than the parametric mechanism coordinates do.

## 4. Verdict
- The unexplained majority of positives do not show a mechanism that this analysis can detect and v3 misses. They
  look like the same mechanisms (boundary-located step changes in level, innovation scale and tail shape) with effect
  sizes comparable to estimation noise.
- There is no sign of gradual onsets, displaced breakpoints, outlier-rate changes or volatility-persistence changes,
  and no consistently signed shift in any other coordinate.
- The identified third is already ranked at 0.984. Every point of headroom is in the weak-effect group, which v3 ranks
  at 0.776.
- **A score of 0.87–0.88 would need ranking those weak-effect positives at about 0.82–0.83 instead of 0.78.** This
  analysis finds no feature or mechanism that would deliver that.
- The score-distribution bound (0.94) is only an optimistic upper limit. It assumes perfect ranking of everything v3
  already separates, so it does not show that 0.87 is reachable.
- On this evidence, **v3 appears close to what is attainable with these mechanism and feature types**, and 0.87–0.88
  is not supported by anything measured here. This is not proven: power on small effects is the binding limit, and the
  irreducible part of the weak-effect overlap cannot be separated from estimation loss with these data alone.
