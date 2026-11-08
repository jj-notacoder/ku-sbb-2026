"""v4 P2b: marginal likelihood-ratio features from the recovered generator (family 'dgplr').
Per series (pre-scaled units, so affine-invariant): fit the generator's model set on the PRE segment only (BIC), then
  llr        = log sum_{m,s} w p(x | theta_hat, m, s) - log p(x | theta_hat, none)   (mechanism/effect-grid mixture)
  llr_<m>    = per-mechanism LR (prior-weighted over that mechanism's grid);  pm_<m> = posterior mechanism probability
  llr_int    = POST-segment marginal LR integrating theta_pre over an approximate posterior (K=4 draws); llr_int_sd
  diagnostics: is_sine, mean-model / family codes, AR(2) spectral radius, sigma ratio, t nu, pre kurtosis, SE widths.
Depends on dgp.py, dgp_oracle.py, dgp_fit.py, dgp_fit2.py and work/v3/dgp/cfg_final.json; DEP_MD5 below is the md5 of
those sources when the cache key was frozen (extract/views hash THIS module's source)."""
import json
from pathlib import Path
import numpy as np
import feats
from feats import FAMILIES

DEP_MD5 = "45d8549cae9e"
_CFG = None
MECHS = ("level", "scale", "phi", "amp", "period")
MM = {k: i for i, k in enumerate(("iid", "trend", "ar1", "ar2", "ar1t", "ar2t", "arma11", "rw", "ll"))}
FAM = {k: i for i, k in enumerate(("gauss", "t", "laplace", "uniform", "skewnorm", "gmm2", "garch"))}


def _cfg():
    global _CFG
    if _CFG is None:
        import dgp
        _CFG = dict(dgp.CFG, **json.load(open(__import__("paths").CONFIG / "cfg_final.json")))
    return _CFG


def fam_dgplr(c):
    import dgp_oracle as O
    x, npre = np.asarray(c.x, float), int(c.npre)
    n = len(x)
    if npre < 20 or n - npre < 8 or np.std(c.z[:npre]) < 1e-9:
        return {}
    d, r = O.est_llr(x, npre, _cfg(), detail=True)
    out = {"llr": d["llr"], "llr_int": d.get("llr_int", 0.0), "llr_int_sd": d.get("llr_int_sd", 0.0)}
    for m in MECHS:
        out[f"llr_{m}"] = d.get(f"llr_{m}", 0.0); out[f"pm_{m}"] = d.get(f"pm_{m}", 0.0)
    pm = np.array([out[f"pm_{m}"] for m in MECHS])
    out["argmax_mech"] = float(np.argmax(pm)) if pm.max() > 0 else -1.0
    out["pm_any"] = float(pm.sum())
    out["is_sine"] = float(bool(r["is_sine"]))
    out["mm_code"] = float(MM.get(r["mean_model"], -1)); out["fam_code"] = float(FAM.get(r["family"], -1))
    out["ar2_rho"] = r["ar2_rho"]; out["sig_ratio"] = r["sig_best"] / max(r["sd"], 1e-9)
    out["log_t_nu"] = float(np.log(min(r["t_nu"], 1e4))); out["pre_kurt"] = r["kurt"]
    ph = float(np.clip(r["ar1_phi"], -0.99, 0.99))
    out["se_phi"] = float(np.sqrt(max(1 - ph * ph, 1e-3) / npre)); out["se_logsig"] = float(1 / np.sqrt(2 * npre))
    out["se_mean"] = float(r["sd"] / np.sqrt(npre) * np.sqrt((1 + abs(ph)) / (1 - abs(ph))))
    out["llr_per_obs"] = d["llr"] / max(n - npre, 1)
    return {k: float(np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)) for k, v in out.items()}


fam_dgplr._hash_module = True
FAMILIES.update({"dgplr": fam_dgplr})
