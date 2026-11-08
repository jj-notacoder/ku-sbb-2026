"""v2 frozen recipe: featset_v2 (wide/shape/pit/vol + battery + distfam, de-duplicated columns), training on original +
time-reversed copies of every series, prediction = average over {orig, reversed} views (TTA) and over model seeds/members.
A series' prediction depends only on that series. Constant / near-constant series return exactly 0.5."""
import os, sys, json                                           # environment, paths, json metadata
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd, lightgbm as lgb              # arrays, tables, the gradient-boosting library
sys.path.insert(0, str(Path(__file__).parent))                 # make this folder's modules importable
import feats, feats2, feats3, feats4, feats5, feats6, feats7, feats8  # noqa: F401 (register families; v6: glr, dgplr2)
from views import view                                         # builds the original or time-reversed version of a series

ROOT = Path(__file__).resolve().parent.parent                  # this version's folder
VIEWS = ["orig", "rev"]                                        # the two views every series is scored on (test-time augmentation)


def is_near_constant(x):                                       # True when a series barely moves at all
    x = np.asarray(x, float)                                   # as floating-point numbers
    return np.ptp(x) <= 1e-9 * max(1.0, float(np.median(np.abs(x))))   # range tiny relative to the typical size -> constant


def features_one(x, npre, fams, vname):                        # all feature columns of ONE series in ONE view
    xv, nv = view(x, npre, vname)                              # the series (and boundary) in this view
    c = feats.Ctx(np.asarray(xv, float), int(nv))              # shared context: robust pre-boundary scaling and cached pieces
    out = {}                                                   # column name -> value
    for fam in fams:                                           # every feature family the model uses
        try:
            with np.errstate(all="ignore"):                    # silence numeric warnings (degenerate inputs are handled below)
                d = feats.FAMILIES[fam](c)                     # compute the family's features
        except Exception:
            d = {}                                             # a family that cannot be computed contributes nothing (filled later)
        out.update({f"{fam}__{k}": v for k, v in d.items()})   # prefix every column with its family name
    return out


def features(series, fams, cols, vname, n_jobs=1):             # feature table for many series in one view
    if n_jobs == 1 or len(series) < 200:                       # small jobs: compute in this process
        rows = [features_one(x, n, fams, vname) for x, n in series]
    else:                                                      # large jobs: spread series over worker processes
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, batch_size=32)(delayed(features_one)(x, n, fams, vname) for x, n in series)
    df = pd.DataFrame(rows, dtype="float64").reindex(columns=cols)   # exactly the model's columns, in its order
    return df.astype("float32").replace([np.inf, -np.inf], np.nan)   # same number type as training; infinities become missing


class Recipe2:                                                 # a set of LightGBM members, each trained with several seeds
    def __init__(self, members):
        """members: list of dicts {name, params (native lgb), n_trees, max_bin, seeds}"""
        self.members = members                                 # model settings
        self.boosters = {}                                     # trained models, by member name
        self.cols = None                                       # feature columns, fixed at training time
        self.fams = None                                       # feature families (needed to recompute features from raw data)

    def fit(self, Xo, Xr, y, threads=12, log=print):           # train on original + time-reversed copies of every series
        self.cols = list(Xo.columns)                           # remember the column order
        Xtr = np.vstack([Xo.to_numpy(np.float32), Xr[self.cols].to_numpy(np.float32)]); ytr = np.concatenate([y, y])   # both views, same labels
        for m in self.members:                                 # each member ...
            d = lgb.Dataset(Xtr, ytr, params={"max_bin": m["max_bin"], "verbose": -1, "feature_pre_filter": False})   # binned training data
            self.boosters[m["name"]] = []
            for s in m["seeds"]:                               # ... trained once per seed
                self.boosters[m["name"]].append(lgb.train(dict(m["params"], seed=s, num_threads=threads), d, num_boost_round=m["n_trees"]))
                log(f"  fitted {m['name']} seed {s}")          # progress
        return self

    def predict(self, Xo, Xr, const_mask=None):                # probability of a lasting break for each series
        A, B = Xo[self.cols].to_numpy(np.float32), Xr[self.cols].to_numpy(np.float32)   # original-view and reversed-view features
        per_member = []
        for m in self.members:
            bs = self.boosters[m["name"]]
            per_member.append(np.mean([0.5 * (b.predict(A) + b.predict(B)) for b in bs], 0))   # average the two views, then the seeds
        p = np.mean(per_member, 0)                             # average the members
        if const_mask is not None:
            p = np.where(const_mask, 0.5, p)                   # constant series: nothing to detect -> exactly 0.5
        return np.clip(p, 0, 1)                                # guarantee the [0, 1] range

    def save(self, d):                                         # write boosters + metadata to a folder
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        meta = {"cols": self.cols, "fams": self.fams, "views": VIEWS, "members": []}   # everything needed to reuse the models
        for m in self.members:
            for i, b in enumerate(self.boosters[m["name"]]):
                b.save_model(str(d / f"{m['name']}_{i}.txt"))  # one text file per booster
            meta["members"].append({k: v for k, v in m.items()} | {"n_saved": len(self.boosters[m["name"]])})
        json.dump(meta, open(d / "recipe2.json", "w"), indent=1)

    @classmethod
    def load(cls, d):                                          # read a saved recipe back
        d = Path(d); meta = json.load(open(d / "recipe2.json"))   # metadata
        r = cls(meta["members"]); r.cols = meta["cols"]; r.fams = meta["fams"]
        for m in meta["members"]:
            r.boosters[m["name"]] = [lgb.Booster(model_file=str(d / f"{m['name']}_{i}.txt")) for i in range(m["n_saved"])]   # the boosters
        return r


def predict_series(R, series, n_jobs=1):                       # raw series -> predictions (each series on its own)
    Xo = features(series, R.fams, R.cols, "orig", n_jobs)      # original-view features
    Xr = features(series, R.fams, R.cols, "rev", n_jobs)       # time-reversed-view features
    const = np.array([is_near_constant(x) for x, _ in series])  # constant-series flags
    return R.predict(Xo, Xr, const)
