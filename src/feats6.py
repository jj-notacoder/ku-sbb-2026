"""v5 S1: correct-null evidence. ONE cached family 'null3' (one shared pre-segment model fit) emitting three column groups:
  nullres_* : the pre-segment model is selected by BIC over the recovered family set (iid/trend/AR1/AR2/AR+trend/ARMA11/
              random walk/local level + deterministic sinusoid; innovation family gauss/t/laplace/uniform/skewnorm/gmm2/GARCH,
              src/dgp_fit*.py). It is removed from BOTH segments -> one-step innovations in fitted-innovation units (GARCH:
              standardised by the conditional sd). Shifts post vs pre: level (plain, studentized with the model variance,
              HAC, Welch-t, median/MAD), log-scale (sd, MAD, mean e^2 score), skew, kurtosis, Hill tail index, quantiles,
              exceedance rate.
  nullsim_* : the phase-1 1c statistic battery (feats4.battery statistics) on the post segment, calibrated against B
              post-continuations simulated from the SELECTED model and innovation family (not AR+IID). One fixed global
              RNG seed reused for every series. Emits rank, z, log tail probability floored at 1/(B+1), simulated SD.
  nullid_*  : selected mean-model / innovation family codes, BIC margin over the runner-up, spectral radius,
              sinusoid period and amplitude (0 if not selected), pre-residual Ljung-Box(10) and kurtosis.
Everything is computed in pre-scaled units, so the family is affine-invariant; reversal TTA is handled by the view layer.
Depends on dgp.py / dgp_fit.py / dgp_fit2.py / dgp_oracle.py / feats4.py (DEP_MD5 records their md5 when frozen)."""
import os, json
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.signal import lfilter, lfiltic
import feats
from feats import FAMILIES

DEP_MD5 = "set-at-freeze"
SIM_SEED = 20261106
B_SIM = int(os.environ.get("NULL3_B", "200"))
MM = {k: i for i, k in enumerate(("iid", "trend", "ar1", "ar2", "ar1t", "ar2t", "arma11", "rw", "ll", "sine"))}
FAM = {k: i for i, k in enumerate(("gauss", "t", "laplace", "uniform", "skewnorm", "gmm2", "garch", "garch_t"))}
_CFG = None


def _cfg():
    global _CFG
    if _CFG is None:
        import dgp
        _CFG = dict(dgp.CFG, **json.load(open(__import__("paths").CONFIG / "cfg_final.json")))
    return _CFG


# ---------------------------------------------------------------- model pieces (shared with the simulator)
def _det(row, n, npre, c, t0=0, m=None):
    """deterministic part over times t0..t0+m-1 for a non-sine row (level offset + trend)."""
    m = n if m is None else m; tt = (t0 + np.arange(m)) / n
    tr = float(row["trend_slope"]) * (tt - npre / n / 2) if row["mean_model"] in ("trend", "ar1t", "ar2t") else 0.0
    return c + tr


def _coefs(row):
    import dgp
    phi, theta = dgp._ar_coefs(row); return dgp._stab(phi), theta


def resid_cols(Z, row, npre_tail, sine_par, eff, lag_d=None, lag_u=0.0, h0=1.0):
    """standardised one-step innovations of columns Z (m, K) under the fitted model.
    lag_d: last 2 values of the de-trended pre process (for AR/rw continuation)."""
    m, K = Z.shape
    if sine_par is not None:
        mean, A, om, ph0, sg = sine_par
        return (Z - (mean + A * np.sin(ph0 + om * np.arange(m)))[:, None]) / sg
    mm = row["mean_model"]; sig = float(row["sig_best"]); phi, theta = _coefs(row)
    d = Z
    full = np.vstack([np.repeat(np.asarray(lag_d, float)[:, None], K, 1), d])
    if mm in ("rw", "ll"):
        u = full[2:] - full[1:-1]
    elif phi:
        p = phi + [0.0] * (2 - len(phi))
        u = full[2:] - p[0] * full[1:-1] - p[1] * full[:-2]
    else:
        u = d.copy()
    if theta:
        zi = np.full((1, K), -theta * lag_u)
        u = lfilter([1.0], [1.0, theta], u, axis=0, zi=zi)[0]
    e = u / sig
    if eff in ("garch", "garch_t"):
        a, b = float(row["g_alpha"]), float(row["g_beta"]); b = min(b, 0.98 - a); w = 1 - a - b
        h = np.empty_like(e); hp = np.full(K, h0); ep = np.zeros(K)
        for t in range(m):
            hp = w + a * ep * ep + b * hp; h[t] = hp; ep = e[t]
        e = e / np.sqrt(np.maximum(h, 1e-8))
    return e


def simulate_post(row, eff, m, B, rng, lag_d, sine_par, det_post):
    """B post continuations (m, B) from the selected model (de-trended units for non-sine, then + det)."""
    import dgp
    if sine_par is not None:
        mean, A, om, ph0, sg = sine_par
        return (mean + A * np.sin(ph0 + om * np.arange(m)))[:, None] + sg * rng.standard_normal((m, B))
    sig = float(row["sig_best"]); phi, theta = _coefs(row); mm = row["mean_model"]
    if eff in ("garch", "garch_t"):
        E = np.column_stack([dgp._std_innov(dict(row, family=eff, fbic_garch=-1e18, fbic_t=0, fbic_gauss=0), m, rng, {"garch_rule": "beats_t", "garch_t_innov": eff == "garch_t"})
                             for _ in range(B)])
    else:
        r2 = dict(row); r2["family"] = eff
        E = dgp._std_innov(r2, m * B, rng, {"garch_rule": "none"}).reshape(m, B)   # family fixed to the selected one
    u = sig * E
    if theta:
        u = u + theta * np.vstack([np.zeros((1, B)), u[:-1]])
    if mm in ("rw", "ll"):
        X = lag_d[-1] + np.cumsum(u, 0)
    elif phi:
        p = phi + [0.0] * (2 - len(phi)); a = np.array([1.0, -p[0], -p[1]])
        zi = lfiltic([1.0], a, y=[lag_d[-1], lag_d[-2]])
        X = lfilter([1.0], a, u, axis=0, zi=np.repeat(zi[:, None], B, 1))[0]
    else:
        X = u
    return X + det_post[:, None]


def _hill(a, frac=0.1):
    a = np.sort(np.abs(a))[::-1]; k = max(int(frac * len(a)), 5)
    if len(a) <= k + 1 or a[k] <= 1e-12:
        return 0.0
    return float(np.mean(np.log(a[:k] / a[k])))


def _lrv_sd(e):
    n = len(e); L = int(np.floor(4 * (n / 100.0) ** (2 / 9))); ec = e - e.mean(); g0 = ec @ ec / n; s = g0
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * (ec[k:] @ ec[:-k]) / n
    return float(np.sqrt(max(s, 1e-12)))


def fam_null3(c):
    import dgp, dgp_oracle as O, feats4
    x, npre = np.asarray(c.x, float), int(c.npre); n = len(x); m = n - npre
    if npre < 20 or m < 8 or np.std(c.z[:npre]) < 1e-9:
        return {}
    z = c.z; cfg = _cfg()
    r = O.est_row(z[:npre]); sine = bool(r["is_sine"]); eff = "gauss" if sine else dgp.eff_family(r, cfg)
    f = {}
    # ---------- residuals of the fitted pre-model on BOTH segments
    if sine:
        om = 2 * np.pi / float(r["sine_period"]); sine_full = (float(r["sine_c"]), float(r["sine_amp"]), om, float(r["sine_ph"]), float(r["sine_sig"]))
        e_all = resid_cols(z[:, None], r, None, sine_full, eff)[:, 0]
        det = None; lag_d = None
    else:
        tt = np.arange(n) / n
        trp = float(r["trend_slope"]) * (tt - npre / n / 2) if r["mean_model"] in ("trend", "ar1t", "ar2t") else 0.0
        cc = float(np.mean((z - trp)[:npre])); det = cc + trp if np.ndim(trp) else np.full(n, cc + trp)
        d = z - det
        e_all = np.r_[np.nan, np.nan, resid_cols(d[2:, None], r, None, None, eff, lag_d=d[:2])[:, 0]]
        lag_d = d[npre - 2:npre]
    ep, eq = e_all[2:npre], e_all[npre:]
    ep = ep[np.isfinite(ep)]; eq = eq[np.isfinite(eq)]
    n1, n2 = len(ep), len(eq)
    dm = eq.mean() - ep.mean()
    f["nullres_level"] = dm
    f["nullres_level_z"] = dm / np.sqrt(1 / n1 + 1 / n2)                       # model (unit) innovation variance
    f["nullres_level_hac"] = dm / (_lrv_sd(ep) * np.sqrt(1 / n1 + 1 / n2))
    f["nullres_level_welch"] = dm / np.sqrt(ep.var() / n1 + eq.var() / n2 + 1e-12)
    mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a))) + 1e-9
    f["nullres_level_rob"] = (np.median(eq) - np.median(ep)) / mad(ep)
    f["nullres_lsd"] = np.log(eq.std() + 1e-9) - np.log(ep.std() + 1e-9)
    f["nullres_lsd_post"] = np.log(eq.std() + 1e-9)                              # vs the fitted unit scale
    f["nullres_lmad"] = np.log(mad(eq)) - np.log(mad(ep))
    f["nullres_e2_score"] = (np.mean(eq ** 2) - np.mean(ep ** 2)) * np.sqrt(n2 / 2)
    f["nullres_skew_d"] = float(stats.skew(eq) - stats.skew(ep)); f["nullres_kurt_d"] = float(stats.kurtosis(eq) - stats.kurtosis(ep))
    f["nullres_hill_d"] = _hill(eq) - _hill(ep)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95); qp, qq = np.quantile(ep, qs), np.quantile(eq, qs)
    for q, a_, b_ in zip(qs, qp, qq):
        f[f"nullres_q{int(q*100):02d}_d"] = b_ - a_
    f["nullres_iqr_lr"] = np.log((qq[3] - qq[1] + 1e-9) / (qp[3] - qp[1] + 1e-9))
    f["nullres_exc3_d"] = np.mean(np.abs(eq) > 3) - np.mean(np.abs(ep) > 3)
    f["nullres_cusum"] = np.abs(np.cumsum(eq - ep.mean())).max() / np.sqrt(n2)
    # ---------- simulated null from the SELECTED model: phase-1 1c battery
    rng = np.random.default_rng(SIM_SEED); B = B_SIM
    if sine:
        ph_post = float(r["sine_ph"]) + om * npre
        sims = simulate_post(r, eff, m, B, rng, None, (float(r["sine_c"]), float(r["sine_amp"]), om, ph_post, float(r["sine_sig"])), None)
    else:
        sims = simulate_post(r, eff, m, B, rng, lag_d, None, det[npre:])
    Bm = np.column_stack([z[npre:], sims])
    M = feats4.fit_null(z[:npre])                       # only used for the battery's AR innovation statistics
    for nm, (kind, v) in feats4.battery(z[:npre], Bm, M).items():
        feats4._calib(f, "nullsim_", nm, kind, v)
    for nm, (kind, v) in feats4.lrs(z[:npre], Bm, M).items():
        feats4._calib(f, "nullsim_", nm, kind, v)
    lps = np.array([f[k] for k in f if k.startswith("nullsim_") and k.endswith("_lp")])
    f["nullsim_fisher"] = float(-2 * lps.sum()); f["nullsim_minlp"] = float(lps.min())
    # ---------- identification / adequacy
    bics = {k: r[f"bic_{k}"] for k in ("iid", "trend", "ar1", "ar2", "ar1t", "ar2t", "arma11", "rw", "ll")}; bics["sine"] = r["sine_bic"]
    srt = sorted(bics.values())
    f["nullid_mm"] = float(MM["sine"] if sine else MM.get(r["mean_model"], -1)); f["nullid_fam"] = float(FAM.get(eff, -1))
    f["nullid_bic_margin"] = float(srt[1] - srt[0])
    phi, _ = _coefs(r)
    f["nullid_rho"] = float(np.max(np.abs(np.roots([1] + [-p for p in phi])))) if phi else (1.0 if r["mean_model"] in ("rw", "ll") else 0.0)
    f["nullid_sine_period"] = float(r["sine_period"]) if sine else 0.0
    f["nullid_sine_amp"] = float(r["sine_amp"] / max(r["sine_sig"], 1e-9)) if sine else 0.0
    es = (ep - ep.mean()) / (ep.std() + 1e-9); k = len(es)
    racf = [float(es[:-j] @ es[j:] / (es @ es + 1e-9)) for j in range(1, 11)]
    f["nullid_lb10"] = float(np.log1p(k * (k + 2) * sum(rj * rj / max(k - j, 1) for j, rj in enumerate(racf, 1))))
    f["nullid_kurt"] = float(stats.kurtosis(ep)); f["nullid_t_nu"] = float(np.log(min(r["t_nu"], 1e4)))
    return {kk: float(np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)) for kk, v in f.items()}


fam_null3._hash_module = True
FAMILIES.update({"null3": fam_null3})
