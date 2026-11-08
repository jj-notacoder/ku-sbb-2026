# ku-sbb-2026

**KU-SBB known-boundary break detection — submission of team JJSS (qualification round)**

## What this is
- A model that reads one time series cut at a known point and gives the probability that the process genuinely
  changed after it.
- **Public leaderboard 0.86276 AUC**; our validation 0.86197 over 15 folds.
- **Read `REPORT.md` first.** It explains the method, the validation, the results and the limits.

## What each file holds
| file | contents |
|---|---|
| `REPORT.md` | the report: method, validation, results and errors, how to rerun, sources and AI use |
| `README.md` | this page |
| `AI_DISCLOSURE.md` | who did what, and how AI tools were used |
| `SOURCES.md` | libraries with versions and licences; published methods with references; data provenance |
| `submission.csv` | the predictions: `id,break_probability`, 4,000 rows, md5 `dc335464fec173ae2cf42996dfdfb67b` |
| `run_solution.ipynb` | notebook: rebuilds `submission.csv` from the raw test series with the saved models, step by step |
| `run_all.py` | script entry point: `--verify-only`, the full rebuild (`--out`), or a reduced end-to-end run (`--reduced`) |
| `requirements.txt`, `environment.yml` | pinned packages for pip and for conda |
| `src/` | code: feature families (`feats*.py`, `dgp*.py`), feature caching (`extract.py`, `views.py`), the model container (`recipe2.py`), scoring of new series (`infer.py`), verify and refit (`final6.py`), and checks (`quick_checks.py`, `edge_cases.py`, `robust3.py`) |
| `models/` | **the checkpoint**: `lgb_t56_0.txt` … `lgb_t56_9.txt` (LightGBM, seeds 0–9) and `recipe2.json` (column order and settings) |
| `config/` | `settings.json` (where the data is), `featset_v6.json` (the 781 columns), `lgbm_56.json` (hyper-parameters), `calib_table.npz` and `dgplr2_priors.json` (frozen feature tables), `cfg_final.json` (reconstructed generator) |
| `results/` | the evidence the report cites: per-fold CV, per-slice AUCs, CV against the leaderboard, robustness and edge-case reports, verification logs, figures (`fig1`–`fig5`), the full experiment log, and `analysis/` (ceiling, generator, error-mining and session records) |

## How to run it
1. **Install:** `pip install -r requirements.txt`, or `conda env create -f environment.yml` then
   `conda activate kusbb`. Use Python 3.13.
2. **Data:** put `train_series.parquet`, `train_labels.csv`, `test_series.parquet` and `sample_submission.csv` in a
   folder named `data` next to this one, or set `data_dir` in `config/settings.json`.
   - The competition data is deliberately not included.
   - The `cache/` folder is absent on purpose: it would hold features computed from test data. It is regenerated
     automatically on the first run, which takes about 23 minutes.
3. **Reproduce the predictions** (loads `models/`, no retraining). Use either:
   - the notebook: run all cells of `run_solution.ipynb`. Its step 7 writes `submission.csv` and step 8 checks the
     md5. Measured 37 min (2,226 s) from a cold cache, on shared cores;
   - the script: `python run_all.py --verify-only`, which requires a byte-identical `submission.csv` (21–23 min) and
     checks `src/infer.py` on 50 series.
4. **Score new series:** `python src/infer.py --input new.parquet --output preds.csv`.
   - Columns `id,time,value,period`, where `period` is 0 before the cut and 1 after.
   - Each series is scored on its own, at about 0.33 s per series.
5. **Retrain from scratch:** `python run_all.py --out <work folder>`.
   - It writes only into that folder. It was run from scratch and reproduced `submission.csv` exactly: same md5,
     0 of 4,000 values different. It took 6 h 05 on shared cores, about 3–4 h on idle ones
     (`results/rebuild_20261107_215147.log`).
   - `python run_all.py --reduced --out <work folder>` runs the same pipeline on 600 series as a quick end-to-end
     proof. It completed without error (`results/reduced_run.log`).
6. **Version lock.** Byte-identity is verified for Python 3.13.14, LightGBM 4.7.0, NumPy 2.3.5, SciPy 1.16.3 and
   pandas 2.3.3, on Windows 11 with 12 cores.
   - Other versions may change the last digits of some probabilities; the scripts warn when versions differ.
   - The three failure messages a user may meet are listed in `results/failure_mode_messages.txt`.
