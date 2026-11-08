"""Feature extraction with per-family caching keyed by (input file sha256, family code hash, helper hash)."""
import os                                                      # environment variables
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # numeric libraries: one thread each ...
    os.environ[v] = "1"                                        # ... because series are spread over many worker processes instead
import sys, json, hashlib, inspect, time, argparse             # paths, json, checksums, reading source code, timing, arguments
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays and tables
from joblib import Parallel, delayed                           # parallel worker processes
sys.path.insert(0, str(Path(__file__).parent))                 # make this folder's modules importable
import feats, feats2, feats3, feats4, feats5, feats6, feats7, feats8  # noqa: F401 (register families; v6 adds glr + dgplr2)

ROOT = Path(__file__).resolve().parent.parent                  # this version's folder
import paths; WORK = paths.CACHE   # feature caches live in cache/


def load_series(path):                                         # raw long table -> ids and list of (values, n_pre)
    df = pd.read_parquet(path)                                 # columns id, time, value, period
    ids = df["id"].to_numpy()                                  # series id of every row
    chg = np.flatnonzero(ids[1:] != ids[:-1]) + 1              # rows where a new series starts
    starts = np.concatenate([[0], chg]); ends = np.concatenate([chg, [len(df)]])   # first / one-past-last row of each series
    uid = ids[starts]                                          # one id per series
    assert len(set(uid)) == len(uid), "ids not contiguous"     # every series must be one block of rows
    vals = df["value"].to_numpy(float); per = df["period"].to_numpy()   # values and before/after flags
    out = []
    for s, e in zip(starts, ends):                             # each series
        p = per[s:e]                                           # its before (0) / after (1) flags
        npre = int((p == 0).sum())                             # number of points before the boundary
        out.append((vals[s:e], npre))
    return list(uid), out


def _helper_hash():                                            # fingerprint of the shared feature helpers (part of every cache key)
    src = inspect.getsource(feats)                             # source code of feats.py
    # helper hash = everything except the family function bodies (approximation: whole module minus families)
    return hashlib.md5(src.split("# ----------------------------------------------------------------- families")[0].encode()).hexdigest()[:10]


def fam_src(fam):                                              # the code that defines a family (changes -> its cache is invalid)
    """code identity of a family: its function source, or its whole module for families flagged _hash_module
    (v3 simulation families, whose behaviour lives in module-level helpers)."""
    fn = feats.FAMILIES[fam]                                   # the family's function
    return inspect.getsource(inspect.getmodule(fn)) if getattr(fn, "_hash_module", False) else inspect.getsource(fn)


def fam_key(fam, data_sha):                                    # cache key = family code + helper code + input data fingerprint
    h = hashlib.md5((fam_src(fam) + _helper_hash() + data_sha).encode()).hexdigest()[:12]
    return h


def _work(chunk, fam):                                         # one worker: compute a family for a chunk of series
    import feats2, feats3, feats4, feats5, feats6, feats7, feats8  # noqa: F401 registers P2..v6 families in worker processes (v6: glr, dgplr2)
    out = []
    for x, npre in chunk:
        d = feats.FAMILIES[fam](feats.Ctx(x, npre))            # features of one series (robust pre-boundary scaling inside Ctx)
        out.append(d)
    return out


def file_sha(path):                                            # fingerprint of a raw data file
    man = json.loads((paths.DATA / "manifest.json").read_text())   # the competition's manifest lists sha256 per file
    name = Path(path).name
    if name in man["files_sha256"]:
        return man["files_sha256"][name]                       # use the published checksum
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()   # otherwise compute it


def extract_family(path, fam, series=None, ids=None, n_jobs=-1, cache=True, chunk=150):   # features of one family for a whole file
    sha = file_sha(path) if cache else "nocache"               # data fingerprint
    key = fam_key(fam, sha)                                    # cache key
    cp = WORK / f"feat_{Path(path).stem}_{fam}_{key}.parquet"  # cache file name
    if cache and cp.exists():
        return pd.read_parquet(cp)                             # already computed for this exact code and data
    if series is None:
        ids, series = load_series(path)                        # read the raw series
    t0 = time.time()
    chunks = [series[i:i + chunk] for i in range(0, len(series), chunk)]   # groups of series for the workers
    res = Parallel(n_jobs=int(os.environ.get("FEAT_JOBS", n_jobs)), batch_size=1)(delayed(_work)(c, fam) for c in chunks)
    rows = [r for rr in res for r in rr]                       # back in series order
    df = pd.DataFrame(rows, index=pd.Index(ids, name="id")).astype("float32")   # one row per series
    df.columns = [f"{fam}__{c}" for c in df.columns]           # prefix columns with the family name
    if not np.isfinite(df.to_numpy()).all():                   # features must be finite: missing values become 0
        bad = df.columns[~np.isfinite(df.to_numpy()).all(0)]
        print(f"[warn] {fam}: non-finite in {list(bad)[:10]} -> set to 0 (finite fallback)")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"[extract] {Path(path).stem} {fam}: {df.shape} in {time.time()-t0:.1f}s", flush=True)
    if cache:
        df.to_parquet(cp)                                      # store for next time
    return df


if __name__ == "__main__":                                     # command-line use: python src/extract.py --fams a,b --files x.parquet
    ap = argparse.ArgumentParser()
    ap.add_argument("--fams", default="wide,shape,pit,vol")    # which families
    ap.add_argument("--files", default="train_series.parquet,test_series.parquet")   # which raw files (in ../data)
    a = ap.parse_args()
    for f in a.files.split(","):
        ids, series = load_series(paths.DATA / f)              # read once per file
        for fam in a.fams.split(","):
            extract_family(paths.DATA / f, fam, series, ids)   # compute (or reuse) each family
