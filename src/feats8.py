"""v6 S2: family 'dgplr2' = marginal likelihood ratio "break vs no break" under a DECONVOLVED effect-size prior.
For each mechanism coordinate c (level, noise scale, AR dependence, tail weight, skewness, mixture share, sinusoid
amplitude / frequency) the glr family (feats7.py) gives a one-step shift estimate d_c with standard error se_c from the
whole-series no-break fit with the nuisance parameters projected out (the Laplace approximation of integrating them out).
Locally, the likelihood of "true shift = delta" relative to "no shift" is exp((d*delta - delta^2/2) / se^2). The prior
over (mechanism, |delta|) comes from empirical-Bayes deconvolution of the TRAINING positives (src/s2_deconv.py) and is
frozen in work/v6/dgplr2_priors.json; random sign. The family averages the ratio over that prior (per stratum: sinusoid
base vs other bases) and emits the total and per-mechanism log ratios."""
import json, hashlib                                           # reading the frozen prior, checking it is unchanged
from pathlib import Path                                       # file paths
import numpy as np                                             # arrays
from feats import FAMILIES                                     # registry of feature families

ROOT = Path(__file__).resolve().parent.parent                  # project folder
PRIOR_PATH = __import__("paths").CONFIG / "dgplr2_priors.json"  # the frozen prior used by the shipped family
PRIOR_MD5 = "b6ac0141a8ca6a5d9c26f629308697d6"                 # its checksum, written in when the prior is frozen (s2_deconv.py fit, 2026-11-07)
MECH = {"lvl": "lvl", "scl": "scl", "dep": "dep", "tail": "tail", "skew": "skew", "mixw": "mixw", "s_amp": "s_amp", "s_frq": "s_frq"}   # mechanism -> aux coordinate
_P = None                                                      # the prior, loaded once per process


def log_lr(d, se, grid, w):                                    # log of the prior-averaged likelihood ratio for ONE mechanism (vectorised over series)
    d, se = np.asarray(d, float)[:, None], np.maximum(np.asarray(se, float), 1e-9)[:, None]   # estimates and SEs as columns
    g = np.asarray(grid, float)[None, :]                       # candidate effect sizes as a row
    a = -0.5 * (g / se) ** 2                                   # penalty for a shift that big
    lp = a + np.logaddexp(d * g / se ** 2, -d * g / se ** 2) - np.log(2.0)   # average over the two signs (cosh form, computed safely)
    lw = np.log(np.maximum(np.asarray(w, float), 1e-300))[None, :]   # log prior weights of the sizes
    return np.logaddexp.reduce(lp + lw, axis=1) - np.logaddexp.reduce(lw, axis=1)   # weighted average of ratios, on the log scale


def features_from_aux(A, prior):                               # dgplr2 columns from a table of glr aux values (rows = series)
    sine = np.asarray(A["ma_best_mean"]) == 6.0                # series whose best no-break model is the sinusoid
    out = {}
    tot = np.full(len(sine), -np.inf)                          # running log of the total (mechanism-weighted) ratio
    for st, mask in (("sine", sine), ("other", ~sine)):        # separate priors for sinusoid and other bases
        P = prior[st]                                          # this stratum's prior
        mw = {m: P["mech_w"][m] for m in P["mech_w"]}          # weight of each mechanism
        z = sum(mw.values())                                   # normaliser (mechanism weights exclude the "no effect" part)
        for m in MECH:
            if m not in mw:
                continue                                       # mechanism not used in this stratum
            se = np.asarray(A[f"aux_se_{MECH[m]}"], float); ok = mask & (se > 0)   # rows of this stratum where the coordinate exists
            l = np.zeros(len(sine)); l[ok] = log_lr(np.asarray(A[f"aux_d_{MECH[m]}"], float)[ok], se[ok] * P["infl"][m], P["grid"][m], P["w"][m])   # its log ratio
            out.setdefault(f"llr_{m}", np.zeros(len(sine)))    # per-mechanism column (0 = no information)
            out[f"llr_{m}"][mask] = l[mask]
            tot[mask] = np.logaddexp(tot[mask], np.log(mw[m] / z) + l[mask])   # add this mechanism's share to the total
    out["llr"] = np.where(np.isfinite(tot), tot, 0.0)          # total log marginal likelihood ratio
    out["llr_max"] = np.max(np.column_stack([out[k] for k in out if k.startswith("llr_")]), 1)   # strongest single mechanism
    return {k: np.asarray(v, float) for k, v in out.items()}


def _prior():                                                  # load the frozen prior once
    global _P
    if _P is None:
        raw = PRIOR_PATH.read_bytes()                          # file contents
        if PRIOR_MD5 != "unset":
            assert hashlib.md5(raw).hexdigest() == PRIOR_MD5, "dgplr2 prior changed"   # refuse an altered prior
        _P = json.loads(raw)["full"]                           # the prior fitted on all training positives
    return _P


def fam_dgplr2(c):
    import feats7                                              # the glr family supplies the shift estimates
    o = feats7.fam_glr(c)                                      # compute them for this series
    if not o:
        return {}                                              # too short / constant: no features
    A = {k: np.array([o[k]]) for k in o if k.startswith("aux_") or k == "ma_best_mean"}   # one-row table
    f = features_from_aux(A, _prior())                         # apply the frozen prior
    return {k: float(np.nan_to_num(v[0])) for k, v in f.items()}   # plain finite numbers


fam_dgplr2._hash_module = True                                 # cache key covers this whole module
FAMILIES.update({"dgplr2": fam_dgplr2})                        # register with the feature extractor
