"""Flags constant / near-constant series, which the recipe always scores as exactly 0.5 (there is nothing to detect)."""
import numpy as np                                             # arrays
import extract, paths                                          # raw-series loader; folder locations
from recipe2 import is_near_constant                           # the rule: range tiny relative to the level


def const_mask(stem, ids):                                     # one True/False per requested series id
    i2, ser = extract.load_series(paths.DATA / f"{stem}.parquet")   # all raw series of that file
    pos = {k: j for j, k in enumerate(i2)}                     # id -> position
    return np.array([is_near_constant(ser[pos[i]][0]) for i in ids])   # apply the rule to each requested series
