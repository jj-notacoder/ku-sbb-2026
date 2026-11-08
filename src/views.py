"""Series-level transforms (views) for augmentation/TTA and cached family extraction under a view."""
import os                                                      # environment variables
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # numeric libraries: one thread each ...
    os.environ.setdefault(_v, "1")                             # ... because series are spread over worker processes
import sys, time, hashlib, inspect                             # paths, timing, checksums, source inspection
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays and tables
from joblib import Parallel, delayed                           # parallel workers
sys.path.insert(0, str(Path(__file__).parent))                 # make this folder's modules importable
import feats, extract                                          # feature families; raw loader and cache-key helpers

ROOT = Path(__file__).resolve().parent.parent                  # this version's folder
import paths; V2 = paths.CACHE / "views"   # reversed-view feature caches


VIEW_VERSION = {"orig": "v1", "swap": "v1", "sign": "v1", "swapsign": "v1", "rev": "v1", "revsign": "v1",
                "croppre": "v1", "croppost": "v1", "croppre_rev": "v1", "croppost_rev": "v1"}   # version tag per view (part of the cache key)


def view(x, npre, name):                                       # the series under a transform; only "orig" and "rev" are used by the recipe
    x = np.asarray(x, float)
    if name == "orig":
        return x, npre                                         # unchanged
    if name == "swap":      # new pre = old post, new post = old pre; internal time order kept
        return np.concatenate([x[npre:], x[:npre]]), len(x) - npre
    if name == "sign":
        return -x, npre                                        # flipped sign (research only)
    if name == "swapsign":
        return -np.concatenate([x[npre:], x[:npre]]), len(x) - npre
    if name == "rev":       # whole-series time reversal (diagnostic)
        return x[::-1].copy(), len(x) - npre                   # read the series backwards; the old post segment becomes the pre segment
    if name == "revsign":
        return -x[::-1].copy(), len(x) - npre
    if name.startswith("crop"):   # drop the outer 25% of one segment (boundary neighbourhood untouched), min 32 kept
        npost = len(x) - npre
        if name.startswith("croppre"):
            k = min(npre // 4, max(npre - 32, 0)); x2, n2 = x[k:], npre - k
        else:
            k = min(npost // 4, max(npost - 32, 0)); x2, n2 = x[:len(x) - k], npre
        if name.endswith("_rev"):
            return x2[::-1].copy(), len(x2) - n2
        return x2.copy(), n2
    raise ValueError(name)                                     # unknown view name


def _work(chunk, fam, vname):                                  # one worker: a family's features for a chunk of series under a view
    import feats2, feats3, feats4, feats5, feats6, feats7, feats8  # noqa: F401 register families (v6: glr, dgplr2)
    out = []
    for x, npre in chunk:
        xv, nv = view(x, npre, vname)                          # transformed series
        try:
            with np.errstate(all="ignore"):
                out.append(feats.FAMILIES[fam](feats.Ctx(xv, nv)))   # its features
        except Exception:
            out.append({})                                     # failure -> no features (filled with 0 below)
    return out


def fam_view(stem, fam, vname, series=None, ids=None):         # cached feature table of one family under one view
    import feats2, feats3, feats4  # noqa
    key = hashlib.md5((extract.fam_src(fam) + extract._helper_hash() + VIEW_VERSION[vname] + stem).encode()).hexdigest()[:10]   # cache key
    cp = V2 / f"vfeat_{stem}_{fam}_{vname}_{key}.parquet"      # cache file
    if cp.exists():
        return pd.read_parquet(cp)                             # reuse
    if series is None:
        ids, series = extract.load_series(paths.DATA / f"{stem}.parquet")   # raw series from ../data
    t0 = time.time()
    chunks = [series[i:i + 150] for i in range(0, len(series), 150)]   # work units
    res = Parallel(n_jobs=int(os.environ.get("FEAT_JOBS", "-1")), batch_size=1)(delayed(_work)(c, fam, vname) for c in chunks)
    df = pd.DataFrame([r for rr in res for r in rr], index=pd.Index(ids, name="id"))   # one row per series
    df.columns = [f"{fam}__{c}" for c in df.columns]           # prefix with the family name
    df = df.astype("float32").replace([np.inf, -np.inf], np.nan)   # finite float32 only
    n_nan = int(df.isna().sum().sum())                         # how many values had to be filled
    df = df.fillna(0.0)
    print(f"[view] {stem} {fam} {vname}: {df.shape} nan->0: {n_nan} {time.time()-t0:.0f}s", flush=True)
    df.to_parquet(cp)                                          # store
    return df


def load_view(fams, vname, stem="train_series"):               # several families under one view, side by side
    from cv import DROP                                        # columns excluded everywhere (identifiers etc.)
    X = pd.concat([fam_view(stem, f, vname) for f in fams], axis=1)
    return X.drop(columns=[c for c in DROP if c in X.columns])


if __name__ == "__main__":                                     # command-line use: python src/views.py --views rev --fams a,b --stems test_series
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--fams", default="wide,shape,pit,vol"); ap.add_argument("--views", default="swap,sign,swapsign,rev")
    ap.add_argument("--stems", default="train_series,test_series"); a = ap.parse_args()
    for stem in a.stems.split(","):
        ids, ser = extract.load_series(paths.DATA / f"{stem}.parquet")   # read once per file
        for v in a.views.split(","):
            for f in a.fams.split(","):
                fam_view(stem, f, v, ser, ids)                 # compute (or reuse)
