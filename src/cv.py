"""Reads the cached original-view feature tables (one parquet file per feature family, written by extract.py)."""
import os, re                                                  # file times, file-name pattern matching
import pandas as pd                                            # tables
import paths                                                   # folder locations
WORK = paths.CACHE                                             # feature caches live in cache/

# zero-by-construction feature (residual mean difference with segment dummies in the model): pure floating-point noise,
# found by the affine-invariance test and removed from every model since v1
DROP = ["shape__cycres_mean_shift"]


def load_feats(fams, stem="train_series"):                     # several families side by side for one raw file
    dfs = []
    for fam in fams:
        pat = re.compile(rf"^feat_{stem}_{re.escape(fam)}_[0-9a-f]{{12}}\.parquet$")   # exact family name + 12-hex cache key
        fs = sorted([f for f in WORK.glob(f"feat_{stem}_{fam}_*.parquet") if pat.match(f.name)], key=os.path.getmtime)
        if not fs:
            raise FileNotFoundError(f"{stem} {fam}")           # not extracted yet: run run_all.py (or src/extract.py) first
        dfs.append(pd.read_parquet(fs[-1]))                    # newest cache of this family
    X = pd.concat(dfs, axis=1)                                 # columns of all requested families
    return X.drop(columns=[c for c in DROP if c in X.columns])
