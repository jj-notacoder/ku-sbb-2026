"""v6 driver = v5 recipe (LightGBM Optuna #56 x 10 seeds, orig + time-reversed training copies, reversal TTA, constants
-> 0.5) on the v6 column set config/featset_v6.json = v5's 690 + 81 'glr' (src/feats7.py) + 10 'dgplr2' (src/feats8.py).
LightGBM spec: config/lgbm_56.json (plain literals) unless env V6_SPEC names another spec file.
  python src/final6.py refit <name> [n_seeds] [--ship] -> fit on all 24,000 series; cache/candidate_<name>.csv
                                                         (--ship: models/ + submission.csv)
  python src/final6.py verify [models_dir]             -> rescore test with saved models; require byte-identical submission.csv"""
import os, sys, json, time                                     # environment, arguments, json, timing
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays and tables
sys.path.insert(0, str(Path(__file__).parent))                 # find project modules
import h2, views                                               # data helpers and the reversed-view feature caches
from cv import load_feats                                      # original-view feature caches
from recipe2 import Recipe2                                    # the multi-seed LightGBM recipe (fit / predict / save / load)
from subcheck import write_checked                             # writes a submission file and checks its format
from constmask import const_mask                                  # flags constant series (they must get exactly 0.5)

import paths; ROOT = paths.ROOT; V3 = V6 = paths.CONFIG   # frozen settings live in config/
THREADS = int(os.environ.get("H2_THREADS", "12"))              # threads per LightGBM fit
FS = json.load(open(V6 / "featset_v6.json"))                   # families and the exact 781 columns
SPEC = json.load(open(os.environ.get("V6_SPEC", str(paths.CONFIG / "lgbm_56.json"))))   # learner settings (Optuna #56 by default)


def mats(stem, ids):                                           # feature matrices (original and reversed views) for given series
    Xo = load_feats(FS["fams"], stem).loc[ids][FS["cols"]]     # original view, columns in the frozen order
    Xr = views.load_view(FS["fams"], "rev", stem).loc[ids][FS["cols"]]   # time-reversed view, same columns
    return Xo, Xr


def members(seeds):                                            # the recipe's model members, each trained with these seeds
    return [dict(m, seeds=list(seeds)) for m in SPEC["members"]]


if __name__ == "__main__":
    step = sys.argv[1]                                         # which action
    lab = h2.labels() if step != "verify" else None            # training labels (verification needs none, so it never reads them)
    if step == "verify":                                       # saved models must reproduce submission.csv byte for byte
        import hashlib, tempfile
        tids = pd.read_csv(paths.DATA / "sample_submission.csv", dtype={"id": str}).id.values   # test ids in submission order
        To, Tr = mats("test_series", tids)                     # test features
        R = Recipe2.load((Path(sys.argv[2]) if len(sys.argv) > 2 else paths.MODELS))   # saved models
        p = R.predict(To, Tr, const_mask("test_series", tids))  # predictions
        tmp = Path(tempfile.mkdtemp()) / "verify.csv"          # temporary file
        write_checked(pd.Series(p, index=tids), tmp)           # written exactly like the real submission
        h = [hashlib.md5(open(f, "rb").read()).hexdigest() for f in (tmp, ROOT / "submission.csv")]   # checksums
        if os.environ.get("KUSBB_DUMP_PRED"):                  # optional: keep the exact in-memory predictions (used by the precision audit)
            np.save(os.environ["KUSBB_DUMP_PRED"], np.asarray(p, dtype=np.float64))
        print(f"[verify] recomputed {h[0]} vs submission.csv {h[1]} -> {'BYTE-IDENTICAL' if h[0] == h[1] else 'DIFFERENT'}")
        if h[0] != h[1]:                                       # explain the size of the difference instead of only failing
            from decimal import Decimal                        # exact decimal parsing (no float round-trip trap)
            new = [Decimal(l.split(",")[1]) for l in open(tmp).read().splitlines()[1:]]   # recomputed values, as written
            old = [Decimal(l.split(",")[1]) for l in open(ROOT / "submission.csv").read().splitlines()[1:]]   # shipped values
            diff = [abs(a - b) for a, b in zip(new, old)]      # per-row absolute differences
            rk = lambda v: np.argsort(np.argsort(np.array(v, dtype=float), kind="stable"), kind="stable")   # rank of every row
            raise SystemExit(f"[verify] NOT byte-identical: {sum(d > 0 for d in diff)} of {len(diff)} values differ, max |diff| "
                             f"{max(diff):.3e}, ranking identical: {bool((rk(new) == rk(old)).all())}. Small last-digit differences "
                             f"are expected with package versions other than the pinned ones (REPORT.md, section 4, 'Version lock').")
    elif step == "reduced":                                    # end-to-end proof: the complete pipeline on a small training subsample
        from sklearn.metrics import roc_auc_score              # fold AUC
        from sklearn.model_selection import train_test_split   # stratified subsample
        import extract, constmask                              # raw loader + original-view extractor; constant-series flags
        n_tr, n_te = int(sys.argv[2]), int(sys.argv[3]); t0 = time.time()
        out = paths.CACHE / "reduced"; out.mkdir(parents=True, exist_ok=True)   # everything goes to the scratch work folder
        assert paths.CACHE.resolve() != (ROOT / "cache").resolve() or "--allow-local" in sys.argv, "use run_all.py --reduced --out <scratch>"
        ids_all = list(lab.index); y_all = lab.values
        sub, _ = train_test_split(np.arange(len(ids_all)), train_size=n_tr, stratify=y_all, random_state=20261107)   # stratified training subsample
        sub = np.sort(sub); tids_all, tser_all = extract.load_series(paths.DATA / "test_series.parquet")
        i2, ser2 = extract.load_series(paths.DATA / "train_series.parquet"); pos = {k: j for j, k in enumerate(i2)}
        trn_ids = [ids_all[i] for i in sub]; trn_ser = [ser2[pos[i]] for i in trn_ids]; y = lab.loc[trn_ids].values
        te_ids = list(tids_all[:n_te]); te_ser = tser_all[:n_te]
        stage = lambda m: print(f"[reduced] {time.time()-t0:7.1f}s {m}", flush=True)   # timed progress line
        def feats_for(stem, ids_, ser_):                       # both views, through the SAME cached extractors the full rebuild uses
            o = pd.concat([extract.extract_family(paths.DATA / f"{stem}.parquet", f, ser_, ids_, cache=False) for f in FS["fams"]], axis=1)
            r = pd.concat([views.fam_view(f"reduced_{stem}_{len(ids_)}", f, "rev", ser_, ids_) for f in FS["fams"]], axis=1)
            return o[FS["cols"]], r[FS["cols"]]
        Xo, Xr = feats_for("train_series", trn_ids, trn_ser); stage(f"features: {len(trn_ids)} training series x 2 views x {len(FS['cols'])} columns")
        To, Tr = feats_for("test_series", te_ids, te_ser); stage(f"features: {len(te_ids)} test series x 2 views")
        aucs = []
        for k, (tr, va) in enumerate(h2.splits(y, 1729)):      # fold construction exactly as in the decision protocol
            Rk = Recipe2(members([0, 1])); Rk.fams = FS["fams"]
            Rk.fit(Xo.iloc[tr], Xr.iloc[tr], y[tr], threads=THREADS, log=lambda s: None)   # #56, 2 seeds, orig + reversed copies
            aucs.append(roc_auc_score(y[va], Rk.predict(Xo.iloc[va], Xr.iloc[va])))   # reversal-TTA predictions on the held-out fold
        stage(f"5-fold CV (split seed 1729, 2 seeds): fold AUC {[round(a, 4) for a in aucs]}, mean {np.mean(aucs):.4f}")
        Rf = Recipe2(members([0, 1])); Rf.fams = FS["fams"]; Rf.fit(Xo, Xr, y, threads=THREADS, log=lambda s: None)   # refit on the whole subsample
        Rf.save(out / "models"); Rf = Recipe2.load(out / "models"); stage("refit, saved and reloaded")
        cm = np.array([__import__("recipe2").is_near_constant(x) for x, _ in te_ser])   # constant-series rule
        p = Rf.predict(To, Tr, cm)                             # cached-feature prediction path
        from recipe2 import predict_series                     # stand-alone raw-series path used by infer.py
        q = predict_series(Rf, te_ser[:20])
        pd.DataFrame({"id": te_ids, "break_probability": p}).to_csv(out / "reduced_predictions.csv", index=False, lineterminator=chr(10))
        stage(f"predicted {len(p)} test series -> {out.name}/reduced_predictions.csv; all finite in [0,1]: {bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())}; "
              f"cached path vs raw-series path max |diff| {float(np.abs(q - p[:20]).max()):.2e}")
        print(f"[reduced] COMPLETE in {time.time()-t0:.0f}s", flush=True)
    elif step == "fitpart":                                    # fit only some seeds (two of these run side by side to fit the S5 time cap)
        a, b, out = int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])   # seeds a..b-1 and where to save them
        ids = load_feats(["wide"]).index; y = lab.loc[ids].values   # every training series and its label
        Xo, Xr = mats("train_series", ids)                     # training features
        t0 = time.time()
        R = Recipe2(members(range(a, b))); R.fams = FS["fams"]  # recipe restricted to these seeds
        R.fit(Xo, Xr, y, threads=THREADS)                      # train (each booster depends only on its own seed and the data)
        R.save(out); print(f"fitpart seeds {a}..{b-1} {time.time()-t0:.0f}s")   # save the boosters
    elif step == "assemble":                                   # merge the parts into one 10-seed recipe, then predict
        import shutil
        name, parts, ship = sys.argv[2], [Path(p) for p in sys.argv[3:] if not p.startswith("--")], "--ship" in sys.argv
        dest = (paths.MODELS if ship else paths.CACHE / f"models_{name}"); dest.mkdir(parents=True, exist_ok=True)
        meta = json.load(open(parts[0] / "recipe2.json")); seeds, k = [], 0   # metadata template; merged seed list; booster counter
        for pdir in parts:                                     # copy each part's boosters in seed order
            pm = json.load(open(pdir / "recipe2.json"))["members"][0]
            for i in range(pm["n_saved"]):
                shutil.copyfile(pdir / f"{pm['name']}_{i}.txt", dest / f"{pm['name']}_{k}.txt"); k += 1
            seeds += pm["seeds"]
        meta["members"][0]["seeds"] = seeds; meta["members"][0]["n_saved"] = k   # one member with all seeds
        json.dump(meta, open(dest / "recipe2.json", "w"), indent=1)
        R = Recipe2.load(dest)                                 # reload exactly as inference will
        tids = pd.read_csv(paths.DATA / "sample_submission.csv", dtype={"id": str}).id.values
        To, Tr = mats("test_series", tids)
        p = R.predict(To, Tr, const_mask("test_series", tids))  # test predictions (mean over the 10 boosters, reversal TTA)
        write_checked(pd.Series(p, index=tids), paths.CACHE / f"candidate_{name}.csv")
        if ship:
            write_checked(pd.Series(p, index=tids), ROOT / "submission.csv")
        print(f"assembled {k} boosters (seeds {seeds}) -> {dest}")
    elif step == "refit":                                      # final fit on all 24,000 labelled series
        name = sys.argv[2]; ship = "--ship" in sys.argv        # candidate name; ship = also write submission.csv
        seeds = list(range(int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 10))   # model seeds 0..n-1
        ids = load_feats(["wide"]).index; y = lab.loc[ids].values   # every training series and its label
        Xo, Xr = mats("train_series", ids)                     # training features
        tids = pd.read_csv(paths.DATA / "sample_submission.csv", dtype={"id": str}).id.values   # test ids
        To, Tr = mats("test_series", tids)                     # test features
        t0 = time.time()
        R = Recipe2(members(seeds)); R.fams = FS["fams"]       # recipe with the frozen families recorded
        R.fit(Xo, Xr, y, threads=THREADS)                      # train
        R.save((paths.MODELS if ship else paths.CACHE / f"models_{name}"))   # save models
        p = R.predict(To, Tr, const_mask("test_series", tids))  # test predictions (reversal TTA inside)
        write_checked(pd.Series(p, index=tids), paths.CACHE / f"candidate_{name}.csv")   # candidate file
        if ship:
            write_checked(pd.Series(p, index=tids), ROOT / "submission.csv")      # the shipped file
        print(f"refit {time.time()-t0:.0f}s")
