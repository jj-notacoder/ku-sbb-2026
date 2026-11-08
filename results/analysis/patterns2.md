> **Archived project record, copied unchanged apart from this note.** File and folder names in it (for example `work/...`, `models_v6/`, `figs/`, research scripts) refer to the original project tree, which was archived outside this submission. They are not needed to run anything here. The numbers are the measured values; the machine-readable copies are in `results/*.csv` and `results/*.json`.

# patterns2.md — A4 error mining (dev 5-seed OOF, split 1729; 100 most confident FN / FP per weak slice)
Figures: figs/v2/a4_<slice>_{fn,fp}.png (24 shown each); numbers: work/v2/a4.json. Descriptors on pre-scaled z.

## high pre-acf1 > 0.6 (n=3252)
- FP (median acf1 .94, acf5 .71, trend F high, |mean eff| .63): random-walk-like wander / smooth global trends and
  strongly periodic signals (period 5-40) whose amplitude/phase drifts. KMeans: 39/100 FP in a "near-unit-root,
  trend F≈5, outlier-free" cluster; 19 in a "very persistent + many |z|>5 points" cluster (z inflates because pre MAD is
  small relative to the wander).
- FN (median acf1 .745, acf5 .26, |mean eff| .05, |std lr| .04): moderately autocorrelated, clearly stationary-looking,
  with NO level/scale change -> change must be in dependence/marginal shape. 75/100 FN in two clusters with acf5 ≈ .09
  and strong dominant-frequency power (AR(2)-style quasi-cycles).
- Implication: separating "persistent wander" from "level break" needs first-difference / innovation-scale tests
  (phase E near-unit-root block). FN look like changes in AR(2) cycle parameters.
## short n_total <= 350 (n=2309)
- FN vs TP: same descriptors except effect size (|mean eff| .07 vs .36) → mostly a power problem (too few points).
- 70% of errors are in a "white-noise, kurt≈0" cluster. Nothing structural to mine; variance reduction is the lever.
## high raw noise (raw pre std top 20%, n=3840)
- Raw std is a proxy for persistence: FP median acf1 .87 (vs TN .50). Same mechanism as high-acf (wander).
- FN: 46 in a low-acf5 quasi-cyclic cluster, 40 in a heavy-tailed white-noise cluster (kurt 1.35, small effects).
## bounded / platykurtic noise (pre kurtosis < -0.8, n=2300)
- FN: two clusters — (a) 42 near-white uniform-like noise (kurt ≈ -1.15, acf ≈ 0); (b) 39 quasi-periodic AR(2)
  (acf1 .75, strong dominant frequency). In both, mean/std/range/kurtosis identical across segments.
  Candidate mechanisms: change in the marginal shape beyond 4th moment (e.g. uniform→triangular/bimodal), or in the
  serial dependence of a uniform-marginal process (copula/rank dependence), i.e. PIT-transition / ordinal / Markov
  features — previously +0.0011 each; worth the joint bundle test in D.
- FP: 41/100 are near-unit-root with kurt_pre -1.2 (smooth wander looks platykurtic) → same wander issue.
## Cross-cutting
- No outlier-driven FN; FP outlier clusters exist only where z is inflated by a tiny pre MAD in persistent series.
- Error structure says: (1) wander vs break (high-acf FP) and (2) dependence-only breaks (FN) are the two systematic
  failure modes; short series are a power problem.
