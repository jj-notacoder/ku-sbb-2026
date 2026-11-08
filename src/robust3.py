"""Full robustness suite for the shipped recipe (models/); writes cache/robustness.json.
Same checks as robust2 (affine invariance, reversal-TTA symmetry, reorder / single-vs-batch, degenerate series,
per-family adversarial AUC) plus: series independence (a series' score does not change when the other series in the
batch change), thread-count-identical refits (6 vs 12 threads), and a full CLI reproduction of submission.csv.
  python src/robust3.py [models_dir]"""
import sys, json, subprocess, tempfile, os
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from recipe2 import Recipe2, predict_series
from infer import series_from_frame
from cv import load_feats

ROOT = Path(__file__).resolve().parent.parent
MD = Path(sys.argv[1]) if len(sys.argv) > 1 else __import__("paths").MODELS
SUB = Path(os.environ.get("ROBUST_SUB", str(ROOT / "submission.csv")))      # CSV the models must reproduce
OUTP = Path(os.environ.get("ROBUST_OUT", str(__import__("paths").CACHE / "robustness.json")))
MATS = os.environ.get("ROBUST_MATS", "final6")                               # module providing mats() for the refit check
R = Recipe2.load(MD)
rep = {"models": MD.name, "n_boosters": sum(len(v) for v in R.boosters.values())}
te = pd.read_parquet(__import__("paths").DATA / "test_series.parquet")
ids = te.id.drop_duplicates().iloc[:60].tolist()
sid, ser = series_from_frame(te[te.id.isin(ids)].copy())
base = pd.Series(predict_series(R, ser), index=sid)

rep["affine_max_abs_diff"] = [float(np.abs(predict_series(R, [(x * a + b, n) for x, n in ser]) - base.values).max())
                              for a, b in [(3.7, -120.0), (0.013, 5.0), (250.0, 1e4)]]
rep["tta_reversal_max_abs_diff"] = float(np.abs(predict_series(R, [(x[::-1].copy(), len(x) - n) for x, n in ser]) - base.values).max())
rep["sign_flip_max_abs_diff(info: sign is not a TTA view)"] = float(np.abs(predict_series(R, [(-x, n) for x, n in ser]) - base.values).max())
pr = predict_series(R, [ser[i] for i in reversed(range(len(ser)))])
rep["reorder_max_abs_diff"] = float(np.abs(pr[::-1] - base.values).max())
rep["single_vs_batch_max_abs_diff"] = float(max(abs(predict_series(R, [ser[i]])[0] - base.values[i]) for i in range(20)))
# series independence: first 20 series scored next to 40 OTHER test series
ids2 = te.id.drop_duplicates().iloc[200:240].tolist()
_, ser2 = series_from_frame(te[te.id.isin(ids2)].copy())
rep["independence_max_abs_diff"] = float(np.abs(predict_series(R, ser[:20] + ser2)[:20] - base.values[:20]).max())
subm = pd.read_csv(SUB, dtype={"id": str}).set_index("id").break_probability
rep["submission_vs_predict_series_max_abs_diff"] = float(np.abs(subm.loc[sid].values - base.values).max())

rng = np.random.default_rng(0)
deg = {"constant": (np.ones(200), 100), "constant_neg_large": (np.full(200, -1e6), 100),
       "near_constant": (5.0 + 1e-13 * rng.normal(size=200), 100),
       "constant_jump": (np.r_[np.ones(100), 2 * np.ones(100)], 100),
       "spike_only": (np.r_[np.zeros(60), 50, np.zeros(139)], 100),
       "short40": (rng.normal(size=80), 40), "very_short3": (rng.normal(size=6), 3), "min1": (rng.normal(size=2), 1),
       "short_post5": (rng.normal(size=205), 200)}
pd_ = predict_series(R, list(deg.values()))
rep["degenerate"] = {k: float(v) for k, v in zip(deg, pd_)}
rep["degenerate_all_finite_in_01"] = bool(np.isfinite(pd_).all() and ((pd_ >= 0) & (pd_ <= 1)).all())
rep["constant_and_near_constant_are_0.5"] = bool(all(rep["degenerate"][k] == 0.5 for k in ("constant", "constant_neg_large", "near_constant")))

# thread-count-identical refits: one seed of the frozen member on 3000 training series, 6 vs 12 threads
import importlib; mats = importlib.import_module(MATS).mats
import h2
lab = h2.labels()
tid = load_feats(["wide"]).index[:3000]
Xo, Xr = mats("train_series", tid)
m0 = dict(R.members[0], seeds=[0])
P6 = Recipe2([m0]).fit(Xo, Xr, lab.loc[tid].values, threads=6, log=lambda s: None).predict(Xo.iloc[:500], Xr.iloc[:500])
P12 = Recipe2([m0]).fit(Xo, Xr, lab.loc[tid].values, threads=12, log=lambda s: None).predict(Xo.iloc[:500], Xr.iloc[:500])
rep["refit_threads_6_vs_12_max_abs_diff"] = float(np.abs(P6 - P12).max())

# CLI reproduction of the full submission
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "cli.csv"
    subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "src" / "infer.py"), "--input", str(__import__("paths").DATA / "test_series.parquet"),
                    "--output", str(out), "--models", str(MD)], check=True)
    cli = pd.read_csv(out, dtype={"id": str}).set_index("id").break_probability
    full = pd.read_csv(SUB, dtype={"id": str}).set_index("id").break_probability
    rep["cli_vs_submission_max_abs_diff"] = float(np.abs(cli.loc[full.index].values - full.values).max())
    rep["cli_n"] = int(len(cli))

adv = {}
for fam in R.fams:
    a = load_feats([fam]); b = load_feats([fam], "test_series")
    cols = [c for c in a.columns if c in set(R.cols)]
    Xa = pd.concat([a[cols], b[cols]]); ya = np.r_[np.zeros(len(a)), np.ones(len(b))]
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=12)
    p = cross_val_predict(m, Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    adv[fam] = round(roc_auc_score(ya, p), 4)
rep["adversarial_auc_by_family"] = adv
rep["sim_feature_determinism"] = os.environ.get("ROBUST_SIMNOTE", "see results/robustness_v6.json (two independent full-test extractions)")   # pointer to the determinism evidence
print(json.dumps(rep, indent=1))
rep["compared_csv"] = SUB.name                                   # file name only (no machine-specific path in the report)
json.dump(rep, open(OUTP, "w"), indent=1)
