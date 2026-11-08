"""Private-split safety and edge-case battery (no training, saved models only). Writes results/edge_cases.json.
Checks that a series' score depends only on that series (batch composition, order, subsets of the real test set), and
that degenerate or out-of-range inputs give a finite probability in [0, 1] (constant series exactly 0.5).
  python src/edge_cases.py"""
import os, sys, json, time                                     # environment, paths, output, timing
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # numeric libraries: one thread each ...
    os.environ.setdefault(v, "1")                              # ... so the parallel feature workers do not oversubscribe the CPU
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays, tables
sys.path.insert(0, str(Path(__file__).parent))                 # this folder's modules
import paths                                                   # folder locations
from recipe2 import Recipe2, predict_series, features          # saved-model container, per-series predictor, feature builder
from infer import series_from_frame                            # long table -> list of (values, n_pre)

t0 = time.time()
R = Recipe2.load(paths.MODELS)                                 # the shipped models
rng = np.random.default_rng(20261107)                          # one fixed seed so the synthetic edge cases are reproducible
rep = {}

# ---------------------------------------------------------------- 1. a series' score depends only on that series
te = pd.read_parquet(paths.DATA / "test_series.parquet")       # raw test file (scored series by series, never pooled)
tid = te.id.drop_duplicates().tolist()                         # test ids in file order
pick = [tid[i] for i in rng.choice(len(tid), 120, replace=False)]   # 120 random test series
sid, ser = series_from_frame(te[te.id.isin(pick)].copy())      # as (values, n_pre) pairs
base = predict_series(R, ser, n_jobs=-1)                       # scored together as one batch
perm = rng.permutation(len(ser))                               # a random re-ordering of the same series
rep["shuffled_order_max_abs_diff"] = float(np.abs(predict_series(R, [ser[i] for i in perm], n_jobs=-1)[np.argsort(perm)] - base).max())
half = predict_series(R, ser[:60], n_jobs=-1)                  # only the first half present in the batch
rep["subset_presence_max_abs_diff"] = float(np.abs(half - base[:60]).max())
one = np.array([predict_series(R, [ser[i]])[0] for i in range(30)])   # one series at a time
rep["single_series_vs_batch_max_abs_diff"] = float(np.abs(one - base[:30]).max())
_, other = series_from_frame(te[te.id.isin(tid[:200])].copy())  # 200 different test series as neighbours
mixed = predict_series(R, ser[:30] + other, n_jobs=-1)[:30]    # same 30 series next to different neighbours
rep["different_neighbours_max_abs_diff"] = float(np.abs(mixed - base[:30]).max())
sub = pd.read_csv(paths.ROOT / "submission.csv", dtype={"id": str}).set_index("id").break_probability
rep["random_120_vs_submission_max_abs_diff"] = float(np.abs(sub.loc[sid].to_numpy() - base).max())   # same values as the shipped file

# ---------------------------------------------------------------- 2. invariances on real series
x0 = ser[:20]                                                  # 20 real test series
p0 = base[:20]
rep["affine_max_abs_diff"] = {f"x*{a}+{b}": float(np.abs(predict_series(R, [(x * a + b, n) for x, n in x0]) - p0).max())
                              for a, b in [(3.7, -120.0), (1e-6, 0.0), (1e6, 0.0), (1.0, 1e9), (-1.0, 0.0)]}   # -1 = sign flip (NOT a symmetry the model claims; reported for information)
rep["time_reversal_max_abs_diff"] = float(np.abs(predict_series(R, [(x[::-1].copy(), len(x) - n) for x, n in x0]) - p0).max())

# ---------------------------------------------------------------- 3. degenerate and out-of-range inputs
N = lambda k: rng.normal(size=k)                               # standard normal noise of length k
cases = {
    "constant_400": (np.full(400, 7.0), 200),
    "constant_zero_400": (np.zeros(400), 200),
    "constant_huge_negative": (np.full(400, -1e12), 200),
    "near_constant_1e-13_noise": (5.0 + 1e-13 * N(400), 200),
    "all_identical_except_one_point_post": (np.r_[np.ones(399), 1.0 + 1e-3], 200),
    "constant_pre_noisy_post": (np.r_[np.zeros(200), N(200)], 200),
    "noisy_pre_constant_post": (np.r_[N(200), np.zeros(200)], 200),
    "constant_pre_level_jump": (np.r_[np.ones(200), 2 * np.ones(200)], 200),
    "spike_only_pre": (np.r_[np.zeros(60), 50.0, np.zeros(339)], 200),
    "spike_only_post": (np.r_[np.zeros(300), 50.0, np.zeros(99)], 200),
    "noise_with_one_huge_spike": (np.r_[N(250), 1e8, N(149)], 200),
    "total_length_2_(1+1)": (N(2), 1),
    "total_length_6_(3+3)": (N(6), 3),
    "total_length_20": (N(20), 10),
    "total_length_100": (N(100), 50),
    "total_length_200_(below_train_min_256)": (N(200), 100),
    "total_length_2048_(above_train_max_1024)": (N(2048), 1024),
    "total_length_8192": (N(8192), 4096),
    "n_pre_1": (N(500), 1),
    "n_pre_2": (N(500), 2),
    "n_post_1": (N(500), 499),
    "n_post_2": (N(500), 498),
    "n_pre_5_of_1000": (N(1000), 5),
    "n_post_5_of_1000": (N(1000), 995),
    "tiny_scale_1e-12": (1e-12 * N(400), 200),
    "huge_scale_1e12": (1e12 * N(400), 200),
    "huge_offset_1e15": (1e15 + N(400), 200),
    "obvious_level_break": (np.r_[N(200), 5 + N(200)], 200),
    "obvious_variance_break": (np.r_[N(200), 4 * N(200)], 200),
    "random_walk_no_break": (np.cumsum(N(400)), 200),
    "integer_valued_series": (np.round(3 * N(400)), 200),
    "binary_valued_series": ((N(400) > 0).astype(float), 200),
}
p = predict_series(R, list(cases.values()))                    # score every case (each on its own)
Xo = features(list(cases.values()), R.fams, R.cols, "orig")    # their features, to count missing values
rep["edge_cases"] = {k: {"n_total": int(len(x)), "n_pre": int(n), "prediction": float(v), "n_missing_features_orig_view": int(Xo.iloc[i].isna().sum())}
                     for i, ((k, (x, n)), v) in enumerate(zip(cases.items(), p))}
rep["edge_all_finite_in_0_1"] = bool(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
rep["constant_and_near_constant_exactly_0.5"] = bool(all(rep["edge_cases"][k]["prediction"] == 0.5 for k in
                                                       ("constant_400", "constant_zero_400", "constant_huge_negative", "near_constant_1e-13_noise")))
try:                                                           # non-finite input values must be rejected, never guessed
    series_from_frame(pd.DataFrame({"id": ["a"] * 4, "time": range(4), "value": [1.0, np.nan, 2.0, 3.0], "period": [0, 0, 1, 1]}))
    rep["nan_input"] = "accepted (unexpected)"
except ValueError as e:
    rep["nan_input"] = f"rejected with ValueError: {e}"
try:                                                           # a series with no post-boundary points has no question to answer
    series_from_frame(pd.DataFrame({"id": ["a"] * 4, "time": range(4), "value": [1.0, 2.0, 2.0, 3.0], "period": [0, 0, 0, 0]}))
    rep["no_post_segment"] = "accepted (unexpected)"
except ValueError as e:
    rep["no_post_segment"] = f"rejected with ValueError: {e}"

rep["PASS"] = bool(max(rep["shuffled_order_max_abs_diff"], rep["subset_presence_max_abs_diff"], rep["different_neighbours_max_abs_diff"]) == 0.0
                   and rep["single_series_vs_batch_max_abs_diff"] < 1e-12 and rep["random_120_vs_submission_max_abs_diff"] < 1e-12
                   and rep["time_reversal_max_abs_diff"] == 0.0 and rep["edge_all_finite_in_0_1"] and rep["constant_and_near_constant_exactly_0.5"])
rep["runtime_s"] = round(time.time() - t0, 1)
out = paths.ROOT / "results" / "edge_cases.json"; out.write_text(json.dumps(rep, indent=1))   # the evidence file
print(json.dumps(rep, indent=1))
