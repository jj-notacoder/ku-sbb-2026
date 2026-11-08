"""Inference for new unlabeled series.
usage: python src/infer.py --input new_series.parquet --output preds.csv [--models models]
Input columns: id, time, value, period (0 before boundary, 1 after). Each series is scored independently."""
import os                                                      # environment variables
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # numeric libraries: one thread each ...
    os.environ.setdefault(v, "1")                              # ... so parallel feature workers do not oversubscribe the CPU
import sys, argparse                                           # command-line arguments
from pathlib import Path                                       # file paths
import numpy as np, pandas as pd                               # arrays and tables
sys.path.insert(0, str(Path(__file__).parent))                 # make this folder's modules importable
from recipe2 import Recipe2, predict_series                    # saved-model container and the per-series predictor

ROOT = Path(__file__).resolve().parent.parent                  # this version's folder


def series_from_frame(df):                                     # long table (one row per time point) -> list of (values, n_pre)
    if not {"id", "time", "value", "period"} <= set(df.columns):   # the four required columns must be present
        raise ValueError("need columns id,time,value,period")
    out, ids = [], []                                          # series and their ids, in first-seen order
    for i, g in df.groupby("id", sort=False):                  # one series at a time, keeping input order
        g = g.sort_values("time")                              # put its points in time order
        v = g["value"].to_numpy(float)                         # the measured values
        if not np.isfinite(v).all():                           # missing / infinite values are rejected, never guessed
            raise ValueError(f"non-finite value in series {i}")
        npre = int((g["period"].to_numpy() == 0).sum())        # number of points before the known boundary
        if npre < 1 or npre >= len(g):                         # need at least one point on each side
            raise ValueError(f"series {i} needs at least one pre and one post observation")
        out.append((v, npre)); ids.append(i)                   # keep the series and its id
    return ids, out


def default_models():                                          # where the saved models live if none is given
    import paths; return paths.MODELS   # this version's saved models


def predict_frame(df, models=None, n_jobs=-1, recipe=None):    # score every series in a table; returns one probability per id
    ids, ser = series_from_frame(df)                           # split the table into series
    R = recipe if recipe is not None else Recipe2.load(Path(models) if models is not None else default_models())   # load the models
    p = predict_series(R, ser, n_jobs=n_jobs)                  # features from raw values -> predictions (each series on its own)
    return pd.Series(p, index=pd.Index(ids, name="id"), name="break_probability")   # labelled result


if __name__ == "__main__":                                     # command-line use
    ap = argparse.ArgumentParser()                             # argument parser
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)   # input table, output csv
    ap.add_argument("--models", default=None, help="model dir (default: models_v6)")   # optional other model folder
    a = ap.parse_args()                                        # read the arguments
    df = pd.read_parquet(a.input) if a.input.endswith(".parquet") else pd.read_csv(a.input)   # parquet or csv input
    p = predict_frame(df, a.models)                            # predictions
    p.reset_index().to_csv(a.output, index=False)              # write id,break_probability
    print(f"wrote {len(p)} predictions to {a.output}")          # confirmation
