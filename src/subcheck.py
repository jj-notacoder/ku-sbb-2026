import numpy as np, pandas as pd                               # arrays and tables
from pathlib import Path                                       # file paths
ROOT = Path(__file__).resolve().parent.parent                  # this version's folder
def write_checked(pred: pd.Series, out):                       # write a submission file and check it is valid
    import paths; ss = pd.read_csv(paths.DATA / "sample_submission.csv", dtype={"id": str})   # the official id list and order
    assert not pred.index.has_duplicates and set(pred.index) == set(ss.id)   # exactly the official ids, each once
    v = pred.reindex(ss.id).to_numpy(float)                    # predictions in the official order
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()   # finite probabilities in [0, 1]
    sub = pd.DataFrame({"id": ss.id.to_numpy(), "break_probability": v})   # the two required columns
    sub.to_csv(out, index=False, lineterminator="\n")          # no index column, Unix line ends (byte-reproducible)
    back = pd.read_csv(out, dtype={"id": str})                 # read it back ...
    assert list(back.columns) == ["id", "break_probability"] and (back.id.values == ss.id.values).all()   # ... and re-check columns / order
    assert len(back) == len(ss) and back.break_probability.between(0, 1).all()
    print(f"[submission] {Path(out).name}: {len(back)} rows OK, mean={v.mean():.4f}")   # confirmation (file name only, no machine-specific path)
