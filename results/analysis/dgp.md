> **Archived project record, copied unchanged apart from this note and the paraphrased P0 section.** File and folder names in it (for example `work/...`, `models_v6/`, `figs/`, research scripts) refer to the original project tree, which was archived outside this submission. They are not needed to run anything here. The numbers are the measured values; the machine-readable copies are in `results/*.csv` and `results/*.json`.

# Data-generating process: documentation evidence and recovery notes

## P0: what the provided files say about the generator
Searched for any named distribution family, parameter range, mechanism list or generation script in:
README.md, START_HERE.md, manifest.json, starter.py and ku_sbb_visual_starter.ipynb.

**No distribution family, parameter range or generation script is named anywhere.** The characterisation below is
derived from the organisers' description, paraphrased in our own words; none of their text is reproduced here.

What the organisers' material says about the generator, in summary:
- **README.** The data are synthetic and the task is binary break detection at a known boundary. The benchmark makes
  no claim that every generating family or break mechanism is equally recoverable, and labels record whether a
  process intervention was generated, so some noisy examples are ambiguous. The reference to families and mechanisms
  implies the generator has discrete families and mechanisms, but none is named.
- **Manifest.** A dataset version string only.
- **Notebook.**
  - The illustrated cases are: an unchanged series and a short-lived fluctuation that returns to baseline, both
    label 0; and a lasting level shift and a lasting increase in spread, both label 1.
  - A break may alter the usual level, the spread, or the dependence between successive readings.
  - The label says whether a lasting break was generated. Every series has an after-boundary part regardless of
    label.
  - A single large spike that returns to baseline is transient and therefore label 0, so useful features should
    summarise the whole after-period rather than react to isolated outliers.
  - The values carry no physical unit.
  - The notebook's toy signal generator is explicitly an illustration, not the competition generator.
- **Starter script.** A few before-versus-after summary statistics with a logistic regression; no generator code.

Implications for recovery (hypotheses to test in P1, not facts):
1. Positive mechanisms named: level change, spread change, dependence change.
2. Negatives can contain transient fluctuations or spikes (a short-lived fluctuation is label 0).
3. Unnamed "families" exist: presumably noise or process families.

## P1: recovered generator (src/dgp.py, cfg work/v3/dgp/cfg_final.json) — PARTIALLY recovered
Fitted on a stratified 4,000-series dev subsample (pre and post segments, pre-scaled units). Adversarial
real-vs-synthetic AUC on the 548 v3 features: 0.791 (v1) -> 0.725 -> 0.700 -> (0.733, rejected change) -> **0.687**
(final). Synthetic-vs-test 0.732 (descriptive). Real-dev-vs-test is 0.559 for reference.

**Lengths.** n_total ~ Uniform{256..1024}; boundary fraction npre/n_total ~ Uniform(0.30, 0.70); label ~ Bernoulli(0.5),
independent of length.

**Base process (pre segment), BIC-selected shares.**
- Deterministic sinusoid + white noise: ~16%. Period ~ U(12, 32), amplitude/noise-sd ~ U(0.5, 3), residual AR ~ 0.
- iid noise: ~47% of the rest. AR(1): ~19%, AR(2): ~13% (some complex roots), linear trend (+AR): ~8%,
  random walk: ~6.5%, ARMA(1,1): ~5%.
- Innovations: Gaussian ~68%, Student-t ~22% (nu median 6.8, IQR 5.6-8.2), 2-component mixtures ~5%, GARCH(1,1) ~3.5%
  (alpha ~0.18, beta ~0.62). Moderate excess kurtosis is common: tails are best taken from the ML t fit for every series.
- Raw location/scale: arbitrary (pre-sd mostly 0.4-3, tail to ~40). Effect sizes are relative to noise scale.
- Label-independent spikes: ~8% of series (isolated |z| > 5 outliers in both labels; label-0 shows no transient excess).
- No hard grids or plateaus were found in the base-process parameters (a continuous empirical prior is used).

**Mechanisms (label 1), single change at the boundary.**
- Non-sine bases: level shift (weight ~0.40; |delta| in {0.5, 1, 2} x pre-process sd, p .65/.20/.15, random sign; for
  random-walk/trend bases in innovation-sd units), innovation-scale change (~0.35; log sd ratio +-{0.35, 0.69} = variance
  x2 or x4, either direction), AR(1)-coefficient change (~0.25; +-{0.1, 0.2, 0.3}).
- Sine bases: noise-scale change (~0.45), period change (~0.35; log ratio +-{0.3, 0.4}), amplitude change (~0.20).
- Evidence: excess density of post-minus-pre parameter differences in positives over the label-0 null shows peaks at
  those grid values. On stationary series 31% of positives are explained by exactly one coordinate, 5% by several and
  **65% by none detectable**, so the mechanism set and effect priors are incomplete. The remaining positives are either
  much smaller effects or mechanisms not identified here.

**Why "partially".** The likelihood ratio built from this mechanism model ranks REAL dev series at only AUC 0.756, while
the same statistic ranks synthetic series at 0.937 (estimated parameters) / 0.982 (true parameters). A LightGBM trained
only on synthetic series scores 0.780 on real dev. The base process is close (adversarial 0.687); the label-conditional
part (what a break looks like and how big it is) is not.

**Remaining discriminators (it5).** Squared AR(2)-residual autocorrelation in both segments and both labels (real data
has more volatility clustering / unmodelled dependence), GLS slope t-statistic.
