"""Fast robustness checks on the saved models (the full suite, incl. refits and adversarial AUCs, is src/robust3.py):
affine invariance, reversal-TTA symmetry, reorder, single-vs-batch, series independence, constant series -> exactly 0.5,
the cached predictions vs the per-series predictor, and that features recomputed from RAW data by this folder's code
equal the stored caches (proves the caches belong to this code).
  python src/quick_checks.py"""
import json, sys                                               # output, arguments
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays, tables
sys.path.insert(0, str(Path(__file__).parent))                 # this folder's modules
import paths, views                                            # folder locations; reversed-view cache loader
from recipe2 import Recipe2, predict_series, features          # model container, per-series predictor, raw-feature builder
from infer import series_from_frame                            # long table -> list of (values, n_pre)
from cv import load_feats                                      # original-view cache loader

R = Recipe2.load(paths.MODELS)                                 # the saved models
te = pd.read_parquet(paths.DATA / "test_series.parquet")       # raw test file
sid, ser = series_from_frame(te[te.id.isin(te.id.drop_duplicates().iloc[:60])].copy())   # 60 test series
base = predict_series(R, ser)                                  # their predictions
rep = {}
rep["affine"] = [float(np.abs(predict_series(R, [(x * a + b, n) for x, n in ser]) - base).max())   # rescale + shift every series
                 for a, b in [(3.7, -120.0), (0.013, 5.0), (250.0, 1e4)]]
rep["reversal_tta"] = float(np.abs(predict_series(R, [(x[::-1].copy(), len(x) - n) for x, n in ser]) - base).max())   # time-reversed input
rep["reorder"] = float(np.abs(predict_series(R, ser[::-1])[::-1] - base).max())   # same series, reversed order
rep["single_vs_batch"] = float(max(abs(predict_series(R, [ser[i]])[0] - base[i]) for i in range(20)))   # one at a time
_, ser2 = series_from_frame(te[te.id.isin(te.id.drop_duplicates().iloc[200:240])].copy())   # 40 other series
rep["independence"] = float(np.abs(predict_series(R, ser[:20] + ser2)[:20] - base[:20]).max())   # neighbours must not matter
sub = pd.read_csv(paths.ROOT / "submission.csv", dtype={"id": str}).set_index("id").break_probability
rep["submission_vs_predict_series"] = float(np.abs(sub.loc[sid].values - base).max())   # file vs per-series predictor
deg = [(np.ones(200), 100), (np.full(200, -1e6), 100), (5.0 + 1e-13 * np.random.default_rng(0).normal(size=200), 100)]   # constant / near-constant
rep["constants_exactly_0.5"] = bool(np.all(predict_series(R, deg) == 0.5))
Xo = features(ser, R.fams, R.cols, "orig"); Xr = features(ser, R.fams, R.cols, "rev")   # features recomputed from raw values
Co = load_feats(R.fams, "test_series").loc[sid][R.cols]; Cr = views.load_view(R.fams, "rev", "test_series").loc[sid][R.cols]   # stored caches
rep["raw_features_vs_cache_max_abs_diff"] = float(max(np.abs(Xo.values.astype(np.float32) - Co.values).max(), np.abs(Xr.values.astype(np.float32) - Cr.values).max()))
rep["PASS"] = bool(max(rep["affine"] + [rep["reversal_tta"], rep["reorder"], rep["independence"]]) == 0.0 and rep["single_vs_batch"] < 1e-12
                   and rep["submission_vs_predict_series"] < 1e-12 and rep["constants_exactly_0.5"] and rep["raw_features_vs_cache_max_abs_diff"] < 1e-3)
print(json.dumps(rep, indent=1))
