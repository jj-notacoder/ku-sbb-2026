# Detecting lasting breaks at a known boundary — final report (team JJSS)

**Final score:** public leaderboard **0.86276** AUC; our validation **0.86197** (15 folds) and 0.86125 (10 folds
never used for decisions). Every number below traces to a file in `results/`.

## 1. Method

**The problem.** Each series is a run of a few hundred to a thousand numbers, cut at a known point into a *before*
stretch and an *after* stretch. We had to judge whether the underlying process genuinely changed at the cut, or only
looks changed because of noise, an outlier or a disturbance that fades. The score is AUC: the chance that a randomly
chosen changed series is ranked above an unchanged one.

**Preprocessing.** Every series is rescaled once, using the median and spread of its before stretch only. The two
stretches are never rescaled separately, since that would erase the level and spread changes we are looking for.
Outliers are never removed. A spike that returns to baseline is, by the organisers' description, an unchanged series,
so the spike is evidence. Because of this scaling the model is blind to units: multiplying a series by 250 and adding
10,000 changes no prediction.

**The 781 inputs** (`config/featset_v6.json`), each computed from one series alone, are of four kinds:
- **Direct comparisons** of the two stretches: level, spread, autocorrelation (how strongly each value predicts the
  next), quantiles and volatility.
- **Distribution shape and cycles:** two-sample tests such as Anderson–Darling, and fits that first remove a
  repeating cycle so it is not mistaken for a level shift.
- **Evidence against a reference model fitted to each series individually.** For example, a series that wanders like
  a random walk is judged against a random-walk null, not against plain noise.
- **Score tests.** The series is fitted once as if nothing changed, and we measure how hard that fit "pulls" toward a
  change after the cut. When a change is small relative to the noise, this is the most sensitive test that statistics
  offers.

**Time reversal.** A lasting change read backwards is still a lasting change at the mirrored cut, and every base
process the data appears to use looks statistically the same in both directions. So every training series is also
used reversed, with the same label, and every prediction averages the forwards and backwards readings. This was the
single biggest improvement we made: +0.0082 AUC, better on all 15 folds. Swapping the two stretches,
which breaks continuity at the cut, lost 0.0025, so the gain came from a real symmetry, not from extra rows.

**The model** is LightGBM gradient-boosted trees: many small decision trees, each correcting the ones before it.
We tuned it on the real setup, reversed copies included, which settled on 1,024 trees at learning rate 0.0156 with
229 leaves. We train it ten times, with seeds 0–9, and average. A flat series, whose range is below
1e-9 × max(1, |median|), returns exactly 0.5.

All feature-extraction, validation, training and reporting code is ours. LightGBM, NumPy, SciPy, pandas and
scikit-learn are standard libraries used as they are. No public solution, pretrained weight or outside dataset was
used.

## 2. Validation

We had 24,000 labelled series. On the first day, 4,800 of them were sealed away as a holdout, leaving 19,200 for
development. A whole series is the unit throughout and is never split across folds.

**How decisions were made.** Every decision used stratified 5-fold cross-validation repeated over three random
splits (15 folds), comparing a candidate with the current best fold by fold. Two further split seeds were used only to
confirm a change already chosen, which separates "this helped" from "this got lucky on one split"
(`results/cv_fold_aucs_v5_v6.csv`).

**What is fitted where.** Everything learned is fitted on training folds and applied to the held-out fold, with
three small global exceptions: the reconstructed generator settings, one family's effect-size prior and the tuned
tree settings. The two priors kept 82–104% of their gain when refitted on half the positives
(`results/experiment_log.csv`).

**Are the series independent?** We correlated all 28,000 train and test series within matching lengths and
boundary fingerprints. Only ten training pairs passed 0.9, all smooth near-periodic series with mixed labels, which
we judged coincidental. No training series was related to a test series.

**The acceptance rule**, fixed before results were seen:
- a mean gain of at least +0.00105, which is twice the measured standard error of a 15-fold mean;
- at least 11 of 15 folds positive and all three repeats positive;
- the same sign with a second seed;
- to ship, at least 8 of 10 positive folds on the confirmation splits.

The rule was never loosened. In one session a likelihood-ratio feature family passed every dev test and the
confirmation mean, but only 7 of 10 confirmation folds were positive. We left it out. It entered later, inside a
bundle that passed on its own. The public leaderboard was never used to choose anything.

## 3. Results, errors and limits

| stage | our validation (AUC) | public leaderboard (AUC) |
|---|---|---|
| starter logistic regression supplied with the competition | 0.694 | — |
| logistic regression on our first 75 features | 0.759 | — |
| first submission (v1) | 0.8406 | 0.84372 |
| reversal + shape tests (v2) | 0.8465 | 0.84544 |
| retuned trees (v3) | 0.84803 | 0.84743 |
| reference-model evidence (v5) | 0.85638 | 0.85642 |
| **score tests, final (v6)** | **0.86197** | **0.86276** |

From v2 on, our validation predicted the leaderboard within 0.0011 on four consecutive submissions, our best
evidence that the model holds on unseen data (`results/cv_vs_leaderboard.csv`, `results/early_phase_baselines.csv`).

**What we tried and rejected.** Every track was judged under the same acceptance rule, so neither of us could put
work into the model on enthusiasm. The second team member developed and tested three:
- **Sequence models.** A temporal convolutional network on the raw series scored 0.744 alone and 0.834 with our
  tabular features added. It ranked series almost as the tree model did (rank correlation 0.92), so the best blend added only +0.00047.
- **Training on simulated series.** This scored 0.780 on real data, and pre-training on them lost 0.0072 on every
  fold. Our simulated changes turned out to be easier than the real ones.
- **Alternative learners and stacking.** XGBoost, extra-trees, other LightGBM variants and a seven-model stack all
  had rank correlations of 0.94–0.99 with the main model; the best blend gained +0.0002.

The first author developed the track that worked:
- exploiting the time-reversal symmetry (+0.0082);
- building evidence against a reference model chosen for each series (+0.0083 on all 15 folds, together with the
  likelihood-ratio features);
- the score tests (+0.0056 on all 15 folds, +0.0065 on all 10 confirmation folds).

**The problems we hit.** For several days in the exploratory phase the score sat near 0.82, whatever descriptive
statistic we added. Our first proper feature set later stalled near 0.836.

The size of a real change turned out to scale with each series' own noise level. That killed a planned family of
raw-unit features before we built it.

About two-thirds of the changed series show no detectable shift in any parameter we can estimate.

Late on, we found that a simulated-noise calibration layer inside two feature families is wrong on real data: its
"standardised" scores have a spread of 1.5–4.4 where 1 is correct (`results/fig5_null_calibration.png`). The trees
treat those columns as raw statistics. We documented the flaw rather than quietly dropping untested columns.

Some problems were mundane. The hyper-parameter search was killed three times by the machine's memory watchdog and
resumed each time from its own database. Late on, two CSV files looked different because their checksums did not
match. The number parser was dropping a final digit; an exact comparison found zero differing values.

**Where it fails** (`results/slice_aucs_v5_v6.csv`, `results/ceiling_decomposition.csv`):

| slice | series | AUC |
|---|---|---|
| all development series | 19,200 | 0.862 |
| changed series with an identifiable mechanism (32% of changes) vs all unchanged | — | 0.986 |
| changed series with no identifiable mechanism (68%) vs all unchanged | — | 0.797 |
| strongly autocorrelated before stretch | 3,252 | 0.820 |
| total length ≤ 350 | 2,309 | 0.792 |
| before stretch ≤ 150 points | 1,707 | 0.811 |
| random-walk-like series | 1,461 | 0.690 |

Short and strongly autocorrelated series sit 0.04–0.07 below the rest. We think that is a limit of the data rather
than something we missed, though we cannot prove it: a full session aimed at these slices moved length ≤ 350 by
+0.0024 and before stretch ≤ 150 by exactly 0.0000, every later idea returned +0.0004 or less, and an optimistic
upper bound from Storey's estimator puts short series near 0.864 even with perfect ranking.

Two risks remain. Time reversal assumes the hidden test series were generated like the training ones; train–test
separability of only 0.535–0.567 AUC per feature family supports this but cannot guarantee it. And the holdout was
opened four times, so it is no longer a clean estimate.

**Why the winning approach won.** It was the only line of work that improved when we gave it better-specified
statistical evidence. More statistics, a bigger or different model, and more data did not help.

## 4. How to rerun

**Setup.** `pip install -r requirements.txt`, or `conda env create -f environment.yml`. Put the competition files in
`../data`, or set `data_dir` in `config/settings.json`; they are deliberately not included.

**Two ways to rerun.** Both load the checkpoint, `models/lgb_t56_0.txt` … `lgb_t56_9.txt` plus
`models/recipe2.json`, and do not retrain.
- **Notebook.** `run_solution.ipynb` runs top to bottom. Its step 7, "Write `submission.csv`", writes the file, and
  step 8 verifies it. It took 37 minutes from a cold cache, on cores shared with a rebuild in a fresh kernel.
- **Script.** `python run_all.py --verify-only` must reproduce `submission.csv` byte for byte. It took 21–23 minutes
  from a cold cache.

A full retrain (`python run_all.py --out <work folder>`) never overwrites the shipped files. Run from scratch, it reproduced
`submission.csv` exactly (0 of 4,000 values different; 6 h 05 on shared cores;
`results/rebuild_20261107_215147.log`). Model seeds are 0–9; split seeds are in `src/h2.py`; the feature simulations use fixed seeds.

**Version lock.** Byte-identity holds for the pinned versions (Python 3.13.14, LightGBM 4.7.0, NumPy 2.3.5, SciPy
1.16.3, pandas 2.3.3; Windows 11, 12 cores). Another build may shift the last digits of some probabilities, which
changes the checksum but not, materially, the ranking or the score.

## 5. Sources, AI use and verification

**Libraries:** NumPy, pandas, SciPy, scikit-learn, joblib (BSD-3), LightGBM (MIT), pyarrow (Apache-2.0); pinned
versions in `SOURCES.md`.

**Published methods**, referenced in `SOURCES.md`: Neyman's C(α) score tests (the "pull" test above); Storey's
estimator of the share of undetectable cases (the upper bound); empirical-Bayes deconvolution (how big real changes
tend to be); the Hampel median filter (smoothing that ignores spikes); Anderson–Darling and Cramér–von Mises tests
(do two stretches share a distribution?); the Hill tail index (how heavy the extremes are); and permutation entropy
over Bandt–Pompe ordinal patterns (how predictable the up-and-down order is). No public competition solution was
copied.

**Synthetic data.** We reconstructed an approximation of the generating process; it is partial, since a classifier
still separates real from simulated series at 0.687 AUC. It fed likelihood-based features and a ceiling estimate.
Training on it was tested and rejected.

**AI use.** Claude, through Claude Code, wrote most of the code to our specification and ran the long experiments
within limits we set. A Gemini CLI was used the same way in the earliest phase. Every ship decision followed an
acceptance rule we fixed in advance; details are in `AI_DISCLOSURE.md`.

**Verification.** The shared acceptance rule is how we checked each other's work. The mechanical checks all passed:
- confirmation splits: 10 of 10 folds positive;
- the sealed holdout;
- a 32-case battery of degenerate inputs: all finite in [0, 1], and flat series exactly 0.5;
- invariance: rescaling, reversal, row order and batch composition change nothing;
- byte-identical reproduction from a clean shell (`results/final_verification.txt`).
