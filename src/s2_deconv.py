"""v6 S2: empirical-Bayes deconvolution of the effect-size prior + shape mechanisms -> family dgplr2 (feats8.py).
Each series has, per mechanism coordinate, a one-step shift estimate d and standard error se from the glr family. For a
positive, d ~ Normal(true shift, se^2); the true shift is drawn from an unknown prior over (mechanism, |size|) with a
random sign. EM on a fixed grid that extends far below the noise floor recovers that prior (NPMLE). se is first
inflated by the robust spread of d/se among NEGATIVES (empirical null), so model misspecification is not read as effect.
  python src/s2_deconv.py validate        -> deconvolve synthetic positives with known truth (work/v3/dgp/sim_2001_26e193)
  python src/s2_deconv.py fit             -> priors full / half (disjoint random half of positives) / pert (+-25%, flattened)
  python src/s2_deconv.py build           -> dgplr2 columns for dev (orig + rev) under each prior + standalone AUCs
  python src/s2_deconv.py screen <tag> [bundle_cols_json]"""
import os, sys, json, time                                     # arguments, json, timing
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays and tables
sys.path.insert(0, str(Path(__file__).parent))                 # find project modules
import feats8                                                  # dgplr2 formula (shared with the shipped family)

ROOT = Path(__file__).resolve().parent.parent                  # project folder
V6 = __import__("paths").CONFIG   # the fitted priors are written next to the other frozen settings
GRID = {"lvl": [0.0625, 0.125, 0.25, 0.45, 0.95, 1.95, 3.0],   # level shift in pre-sd units (old grid .45/.95/1.95 plus much smaller sizes)
        "scl": [0.025, 0.05, 0.1, 0.175, 0.35, 0.69, 1.0],     # log noise-sd change (log-variance ratio 0.05 .. 2)
        "dep": [0.03, 0.06, 0.1, 0.17, 0.3, 0.5],              # AR coefficient change (old grid .1/.17/.3 plus smaller)
        "tail": [0.1, 0.25, 0.5, 1.0, 2.0],                    # change in log(nu) of the t noise (NEW shape mechanism)
        "skew": [0.05, 0.1, 0.2, 0.4],                         # change in log skewness factor (NEW shape mechanism)
        "mixw": [0.25, 0.5, 1.0, 2.0],                         # change in logit share of wide noise (NEW shape mechanism)
        "s_amp": [0.05, 0.1, 0.2, 0.4, 0.7, 1.0],              # relative sinusoid amplitude change
        "s_frq": [0.01, 0.03, 0.1, 0.3, 0.4]}                  # relative sinusoid frequency change
STRATA = {"other": ["lvl", "scl", "dep", "tail", "skew", "mixw"],   # mechanisms allowed for non-sinusoid bases
          "sine": ["scl", "s_amp", "s_frq", "tail", "skew", "mixw"]}   # and for sinusoid bases


def aux_table(stem, view, ids=None):                           # glr aux columns (+ best model code) for a file and view
    if view == "orig":
        from cv import load_feats; X = load_feats(["glr"], stem)   # cached original-view features
    else:
        import views; X = views.load_view(["glr"], view, stem)   # cached reversed-view features
    X = X.loc[ids] if ids is not None else X                   # restrict to the requested series
    cols = [c for c in X.columns if c.startswith("glr__aux_")] + ["glr__ma_best_mean"]   # side data + stratum code
    return X[cols].rename(columns=lambda c: c[5:])             # drop the "glr__" prefix


def inflation(A, y):                                           # empirical-null widening factor per mechanism, from negatives
    out = {}
    for m in GRID:
        se = A[f"aux_se_{feats8.MECH[m]}"].values; ok = (y == 0) & (se > 0)   # negatives where the coordinate exists
        z = A[f"aux_d_{feats8.MECH[m]}"].values[ok] / se[ok]   # their standardised shift estimates (should be ~N(0,1))
        out[m] = float(max(1.0, 1.4826 * np.median(np.abs(z - np.median(z)))))   # robust SD of those, never below 1
    return out


def em(A, mask, mechs, infl, iters=400):                       # NPMLE of the prior over (mechanism, |size|) for the rows in mask
    cols, names = [np.zeros(mask.sum())], [("none", 0.0)]      # component 0 = "no detectable effect" (log ratio 0)
    for m in mechs:
        d = A[f"aux_d_{feats8.MECH[m]}"].values[mask]; se = A[f"aux_se_{feats8.MECH[m]}"].values[mask] * infl[m]   # estimates, widened SEs
        for g in GRID[m]:
            l = np.zeros(len(d)); ok = se > 0                  # coordinate missing -> no information (ratio 1)
            l[ok] = feats8.log_lr(d[ok], se[ok], [g], [1.0])   # log ratio "shift of size g in mechanism m" vs "no shift"
            cols.append(l); names.append((m, g))
    L = np.column_stack(cols); pi = np.full(L.shape[1], 1.0 / L.shape[1])   # per-row log ratios; start from equal weights
    for _ in range(iters):                                     # EM: posterior responsibilities, then average them
        lp = L + np.log(np.maximum(pi, 1e-300)); lp -= lp.max(1, keepdims=True)
        R = np.exp(lp); R /= R.sum(1, keepdims=True)           # probability each component produced each series
        pi = R.mean(0)                                         # new prior weights
    return names, pi


def to_prior(names, pi, infl, mechs):                          # convert EM weights to the dgplr2 prior format
    P = {"null": float(pi[0]), "mech_w": {}, "grid": {}, "w": {}, "infl": {m: infl[m] for m in mechs}}
    for m in mechs:
        ix = [i for i, (mm, _) in enumerate(names) if mm == m]  # components of this mechanism
        P["grid"][m] = [names[i][1] for i in ix]; ww = np.array([pi[i] for i in ix])   # sizes and their weights
        P["mech_w"][m] = float(ww.sum()); P["w"][m] = [float(v) for v in (ww / max(ww.sum(), 1e-300))]   # mechanism share; size distribution within it
    return P


def fit_prior(A, y, rows, infl):                               # both strata
    sine = A["ma_best_mean"].values == 6.0
    return {st: to_prior(*em(A, rows & (y == 1) & (sine if st == "sine" else ~sine), STRATA[st], infl), infl, STRATA[st]) for st in STRATA}


def summary(P):                                                # readable summary of a prior
    out = {}
    for st, Q in P.items():
        tot = sum(Q["mech_w"].values()) + Q["null"]
        out[st] = {"null": round(Q["null"] / tot, 4), **{m: round(v / tot, 4) for m, v in Q["mech_w"].items()}}   # shares
        if "lvl" in Q["grid"]:
            g, w = np.array(Q["grid"]["lvl"]), np.array(Q["w"]["lvl"])
            out[st]["lvl_mass_below_0.45"] = round(float(w[g < 0.45].sum()), 4)   # level mass below the old grid's smallest size
        out[st]["size_dist"] = {m: dict(zip([str(x) for x in Q["grid"][m]], [round(v, 3) for v in Q["w"][m]])) for m in Q["grid"]}
    return out


if __name__ == "__main__":
    import h2
    step = sys.argv[1]
    dev = h2.dev_ids(); y = h2.labels().loc[dev].values        # dev series and labels
    if step == "validate":                                     # synthetic data with KNOWN mechanisms and sizes
        tag = "sim_2001_26e193"; meta = pd.read_parquet(ROOT / "work" / "v3" / "dgp" / f"{tag}_meta.parquet")
        import extract                                         # series loader and the parallel family extractor
        ids, ser = extract.load_series(ROOT / "work" / "v3" / "dgp" / f"{tag}.parquet")   # all 20,000 synthetic series
        sel = np.sort(np.random.default_rng(7).choice(len(ids), int(os.environ.get("S2_NSYN", "6000")), replace=False))   # random subsample (time cap)
        X = extract.extract_family(ROOT / "work" / "v3" / "dgp" / f"{tag}.parquet", "glr", series=[ser[i] for i in sel], ids=[ids[i] for i in sel], cache=False)   # glr on them
        meta = meta.loc[X.index]                               # truth for the same series
        A = X[[c for c in X.columns if c.startswith("glr__aux_")] + ["glr__ma_best_mean"]].rename(columns=lambda c: c[5:])
        ys = meta.label.values; infl = inflation(A, ys)        # synthetic labels; empirical-null widening
        P = fit_prior(A, ys, np.ones(len(ys), bool), infl)     # deconvolve all synthetic positives
        sine_true = (meta.mean_model == "sine").values; pos = ys == 1
        truth = {st: meta[pos & (sine_true if st == "sine" else ~sine_true)].mech.value_counts(normalize=True).round(4).to_dict() for st in STRATA}
        lv = meta[pos & (meta.mech == "level")]["size"].abs().value_counts(normalize=True).round(3).to_dict()
        res = {"inflation": infl, "recovered": summary(P), "true_mech_share_by_TRUE_stratum": truth, "true_level_sizes": lv,
               "best_model_is_sine_vs_true_sine_agreement": round(float(((A.ma_best_mean.values == 6.0) == sine_true).mean()), 4)}
        (V6 / "s2_validate.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    elif step == "fit":                                        # priors on the real dev positives
        A = aux_table("train_series", "orig", dev); infl = inflation(A, y)   # orig-view aux and the empirical-null widening
        full = fit_prior(A, y, np.ones(len(y), bool), infl)    # (all positives)
        rng = np.random.default_rng(515); pos = np.flatnonzero(y == 1)   # disjoint random half of the positives
        half_rows = np.zeros(len(y), bool); half_rows[rng.choice(pos, len(pos) // 2, replace=False)] = True; half_rows |= y == 0
        half = fit_prior(A, y, half_rows, infl)                # (i) prior from that half only
        pert = json.loads(json.dumps(full))                    # (ii) perturbed copy of the full prior
        for st, Q in pert.items():
            for j, m in enumerate(Q["grid"]):
                f = 0.75 if j % 2 == 0 else 1.25               # sizes x0.75 or x1.25, alternating by mechanism
                Q["grid"][m] = [g * f for g in Q["grid"][m]]
                Q["w"][m] = [0.5 * w + 0.5 / len(Q["w"][m]) for w in Q["w"][m]]   # size weights flattened 50% toward uniform
            tot = sum(Q["mech_w"].values()); k = len(Q["mech_w"])
            Q["mech_w"] = {m: 0.5 * v / tot + 0.5 / k for m, v in Q["mech_w"].items()}   # mechanism weights flattened too
        boot = []                                              # bootstrap CI on the mechanism shares (resampling positives)
        for b in range(int(os.environ.get("S2_BOOT", "40"))):
            rb = np.random.default_rng(1000 + b); idx = np.r_[rb.choice(pos, len(pos)), np.flatnonzero(y == 0)]
            Ab = A.iloc[idx].reset_index(drop=True); yb = y[idx]
            boot.append(summary(fit_prior(Ab, yb, np.ones(len(yb), bool), infl)))
        ci = {st: {k: [round(float(np.percentile([bb[st][k] for bb in boot], q)), 4) for q in (2.5, 97.5)]
                   for k in boot[0][st] if k != "size_dist"} for st in STRATA}
        out = {"full": full, "half": half, "pert": pert}
        (V6 / "dgplr2_priors.json").write_text(json.dumps(out, indent=1))
        rep = {"inflation": infl, "full": summary(full), "half": summary(half), "boot_ci95": ci}
        (V6 / "s2_prior_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
    elif step == "build":                                      # dgplr2 columns under each prior, both views
        from sklearn.metrics import roc_auc_score
        pri = json.load(open(V6 / "dgplr2_priors.json")); res = {}
        for view in ("orig", "rev"):
            A = aux_table("train_series", view, dev)
            for tag, P in pri.items():
                F = pd.DataFrame(feats8.features_from_aux(A, P), index=A.index).add_prefix("dgplr2__").astype("float32")
                F.to_parquet(V6 / f"dgplr2_{tag}_{view}.parquet")
        for tag in pri:
            o = pd.read_parquet(V6 / f"dgplr2_{tag}_orig.parquet"); r = pd.read_parquet(V6 / f"dgplr2_{tag}_rev.parquet")
            res[tag] = {"llr_orig": round(roc_auc_score(y, o["dgplr2__llr"]), 4), "llr_tta": round(roc_auc_score(y, (o["dgplr2__llr"] + r["dgplr2__llr"]) / 2), 4),
                        "llr_max_tta": round(roc_auc_score(y, (o["dgplr2__llr_max"] + r["dgplr2__llr_max"]) / 2), 4)}
        (V6 / "s2_standalone_auc.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    elif step == "screen":                                     # paired screen on the v5 base (optionally + S1 survivors)
        import h3, s1_glr_screen
        tag = sys.argv[2]; extra = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else []   # prior tag; optional glr columns kept in S1
        M = s1_glr_screen.v5_base()                            # 690-column v5 base
        if extra:
            M, _ = h3.add_family(M, ["glr"], cols=extra, dedup=False)   # + the S1 survivors (bundle test)
        N = {v: pd.read_parquet(V6 / f"dgplr2_{tag}_{v}.parquet").loc[dev] for v in ("orig", "rev")}   # dgplr2 columns
        M2 = {v: pd.concat([M[v], N[v]], axis=1) for v in ("orig", "rev")}
        base = os.environ.get("S2_BASE", "s1_null3_bundle")    # which cached run to pair against
        name = f"s2v6_dgplr2_{tag}" + ("_bundle" if extra else "")
        c = h3.screen(name, h3.aug_fp(M2), base=base, desc=f"{'v5 base + S1 glr survivors' if extra else 'v5 base (690)'} + dgplr2[{tag}] ({N['orig'].shape[1]} cols)",
                      note="v6 S2 deconvolved-prior marginal LR")
        (V6 / f"s2_screen_{name}.json").write_text(json.dumps(c, indent=1, default=float))
