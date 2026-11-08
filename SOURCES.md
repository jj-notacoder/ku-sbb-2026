# Sources, libraries and data provenance
*Library versions and licences, published methods and data provenance. The plain-language summary is in `REPORT.md`, section 5.*

## Data
- **Only the competition's own files were used**: `train_series.parquet`, `train_labels.csv` and
  `test_series.parquet`, plus `sample_submission.csv` for the required id order. They are **not included** in this
  folder. Point `config/settings.json` at them (see `README.md`).
- **No external data** of any kind was used.
- **No pretrained weights** or pretrained models were used. Every model was trained from scratch on the competition
  training series.
- **Nothing that produces a prediction was fitted on the test set.** Test series were used only for per-series
  feature extraction and prediction. No scaler, rank, quantile, calibration, pseudo-label or model was fitted on them,
  and no test labels exist in this project.
- The only other use of test features is a descriptive train-vs-test adversarial AUC, a distribution-shift check
  whose classifier is discarded.
- The organisers' README, notebook and starter script were read once (session 4) for statements about the generator.
  `results/analysis/dgp.md` summarises them in our own words; no organiser text or code is reproduced.

## Third-party software (pinned in `requirements.txt`)
| package | version | licence | used for |
|---|---|---|---|
| Python | 3.13.14 | PSF License 2.0 | runtime |
| numpy | 2.3.5 | BSD-3-Clause | arrays, linear algebra |
| pandas | 2.3.3 | BSD-3-Clause | tables, CSV/parquet I/O |
| pyarrow | 26.0.0 | Apache-2.0 | parquet reading/writing |
| scipy | 1.16.3 | BSD-3-Clause | optimisation, distributions, signal filtering, special functions |
| scikit-learn | 1.8.0 | BSD-3-Clause | stratified folds, AUC, Gaussian mixtures, robustness-suite helpers |
| lightgbm | 4.7.0 | MIT | the gradient-boosted tree model |
| joblib | 1.5.3 | BSD-3-Clause | parallel feature extraction |
| threadpoolctl | 3.6.0 | BSD-3-Clause | scikit-learn dependency (thread control) |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based, BSD-compatible) | `src/make_figures.py` only |

Used in research only, not needed to run anything in this folder:
- optuna 5.0.0 (MIT): the hyper-parameter search. Its result is frozen as literals in `config/lgbm_56.json`.
- xgboost 3.1.2 (Apache-2.0), catboost 1.2.10 (Apache-2.0) and torch 2.15.0+cpu (BSD-3-Clause): rejected
  alternative learners.

## Methods from the literature (where they are used in `src/`)
| method | where used | reference |
|---|---|---|
| Neyman C(α) / efficient score (Lagrange-multiplier) tests | glr `sc_` columns (`src/feats7.py`) | J. Neyman (1959), "Optimal asymptotic tests of composite statistical hypotheses", in *Probability and Statistics* (Cramér volume), Wiley. C. R. Rao (1948), *Proc. Cambridge Phil. Soc.* 44, 50–57. |
| Generalised likelihood-ratio tests, BIC model selection and BIC weights | glr `lr_`, `ma_`; null3 model choice | G. Schwarz (1978), *Annals of Statistics* 6(2), 461–464. |
| Storey's π₀ estimator | ceiling analysis (`results/analysis/ceiling.md`) | J. D. Storey (2002), "A direct approach to false discovery rates", *JRSS B* 64(3), 479–498. |
| Empirical-Bayes deconvolution (NPMLE by EM) | dgplr2 prior (`src/s2_deconv.py`) | J. Kiefer & J. Wolfowitz (1956), *Ann. Math. Stat.* 27, 887–906. B. Efron (2016), "Empirical Bayes deconvolution estimates", *Biometrika* 103(1), 1–20. |
| Hampel-type robust scaling and filtering (median / MAD; running-median filtered views) | pre-boundary scaling (`src/feats.py: Ctx`) and `wide__mf5_*`, `wide__mf11_*` (`src/feats.py`) | F. R. Hampel (1974), "The influence curve and its role in robust estimation", *JASA* 69, 383–393. |
| Anderson–Darling and Cramér–von Mises two-sample statistics | `shape__ad`, `shape__cvm` (`src/feats.py`), `battery__bt_adz_*` (`src/feats3.py`) | T. W. Anderson & D. A. Darling (1952), *Ann. Math. Stat.* 23, 193–212. T. W. Anderson (1962), *Ann. Math. Stat.* 33, 1148–1159. |
| Hill tail-index estimator | `shape__hill_*` (`src/feats.py`), `null3__nullres_hill_d` (`src/feats6.py`) | B. M. Hill (1975), *Annals of Statistics* 3(5), 1163–1174. |
| Permutation entropy / Bandt–Pompe ordinal patterns | `shape__pe3_*`, `shape__pe4_*` (`src/feats.py`) | C. Bandt & B. Pompe (2002), *Physical Review Letters* 88, 174102. |
| GARCH(1,1) | noise law in glr, null3, dgplr | T. Bollerslev (1986), *J. Econometrics* 31, 307–327. |
| Student-t noise and a t-weighted (outlier-down-weighting) location score | glr level score | K. L. Lange, R. J. A. Little & J. M. G. Taylor (1989), *JASA* 84, 881–896. |
| Probability integral transform | pit family | M. Rosenblatt (1952), *Ann. Math. Stat.* 23, 470–472. |
| Tree-structured Parzen estimator (Optuna) | hyper-parameter search (research) | J. Bergstra et al. (2011), *NeurIPS*. T. Akiba et al. (2019), *KDD*. |
| Gradient-boosted decision trees (LightGBM) | the model | G. Ke et al. (2017), "LightGBM", *NeurIPS*. |

All implementations in `src/` are the project's own code built on the libraries above. No third-party
change-detection package was used.
