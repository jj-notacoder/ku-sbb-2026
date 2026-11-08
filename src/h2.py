"""Data helpers shared by the shipped scripts: training labels, the fixed dev split, and the validation folds.
(The research harness that ran the experiments lived in a larger version of this module; only the parts the
shipped path and the rebuild scripts call are kept here.)"""
import numpy as np, pandas as pd                               # arrays and tables
from sklearn.model_selection import StratifiedKFold            # stratified K-fold splitter
import paths                                                   # folder locations
WORK = paths.CACHE                                             # dev/holdout id lists are written here by run_all.py

REPEATS = [1729, 2026, 31415]                                  # split seeds used for every decision (3 x 5 = 15 folds)
FRESH = [5150, 6260]                                           # split seeds held back for confirmation only (2 x 5 = 10 folds)


def labels():                                                  # training labels, indexed by series id
    return pd.read_csv(paths.DATA / "train_labels.csv", dtype={"id": str}).set_index("id").label


def dev_ids():                                                 # the 19,200 development series (the other 4,800 are the sealed holdout)
    return pd.read_csv(WORK / "dev_ids.csv", dtype={"id": str}).id.values


def splits(y, seed):                                           # stratified 5-fold split of the dev series for one split seed
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))   # whole series per fold, class balance kept
