"""v4 P2a/P2b: likelihood-ratio statistics for the recovered generator.
logLR(x) = log sum_{m,s} w_{m,s} p(x | theta, mechanism m, size s) - log p(x | theta, no break), mixture over the
generator's mechanism set, effect grid and sign (weights from the dgp cfg).
  ORACLE   : theta = the TRUE generating parameters and latents (fitted row, level offset, sine phase).
  ESTIMATED: theta fitted on the PRE segment only (same battery + sinusoid model, BIC selection) -> plug-in LR.
  python src/dgp_oracle.py ceiling <sim_tag>   -> work/v3/dgp/ceiling_<tag>.json (oracle + estimated AUC, slices)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.signal import lfilter
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).parent))
import dgp

LOG2PI = np.log(2 * np.pi)


def logf(e, f, row):
    if f == "t":
        nu = max(float(row["t_nu"]), 2.2); s = np.sqrt(nu / (nu - 2)); return stats.t.logpdf(e * s, nu) + np.log(s)
    if f == "laplace":
        b = 1 / np.sqrt(2); return -np.log(2 * b) - np.abs(e) / b
    if f == "uniform":
        return np.where(np.abs(e) <= np.sqrt(3), -np.log(2 * np.sqrt(3)), -50.0)
    if f == "skewnorm":
        a = float(row["skew_a"]); d = a / np.sqrt(1 + a * a); m = d * np.sqrt(2 / np.pi); v = 1 - m * m
        return stats.skewnorm.logpdf(e * np.sqrt(v) + m, a) + 0.5 * np.log(v)
    if f == "gmm2":
        w, sep, sdr = float(row["gmm_w"]), float(row["gmm_sep"]), float(row["gmm_sdr"])
        mu = w * sep; sd = np.sqrt(w * sdr ** 2 + (1 - w) + w * (1 - w) * sep ** 2); r = e * sd + mu
        return np.logaddexp(np.log(w) + stats.norm.logpdf(r, sep, sdr), np.log(1 - w) + stats.norm.logpdf(r)) + np.log(sd)
    return -0.5 * (LOG2PI + e * e)


def _pad2(p):
    return list(p) + [0.0] * (2 - len(p))


def ll_nonsine(xz, npre, row, c, cfg, mech="none", size=0.0, lo=2):
    n = len(xz); mm = row["mean_model"]; sig = float(row["sig_best"]); tt = np.arange(n) / n
    d = xz - c
    if mm in ("trend", "ar1t", "ar2t"):
        d = d - float(row["trend_slope"]) * (tt - npre / n / 2)
    if mech == "level":
        unit = float(row["sig_best"]) if (cfg.get("level_unit_rw") == "sig" and mm in ("rw", "ll", "trend", "ar1t", "ar2t")) else float(row["sd"])
        d = d.copy(); d[npre:] -= size * unit
    phi, theta = dgp._ar_coefs(row); phi = dgp._stab(phi); phi_post = list(phi)
    if mech == "phi":
        base = phi if phi else [0.0]
        phi_post = dgp._stab([base[0] + size] + list(base[1:])); phi = list(base) if not phi else phi
    if mm in ("rw", "ll"):
        u = np.r_[0.0, np.diff(d)]
    elif phi or mech == "phi":
        p0 = _pad2(phi) if phi else [0.0, 0.0]; p1 = _pad2(phi_post)
        d1 = np.r_[0.0, d[:-1]]; d2 = np.r_[0.0, 0.0, d[:-2]]
        u0 = d - p0[0] * d1 - p0[1] * d2; u1 = d - p1[0] * d1 - p1[1] * d2
        u = np.where(np.arange(n) < npre, u0, u1)
    else:
        u = d.copy()
    if theta:
        u = lfilter([1.0], [1.0, theta], u)
    s = np.ones(n)
    if mech == "scale":
        s[npre:] = np.exp(size)
    e = u / (sig * s)
    f = dgp.eff_family(row, cfg)
    if f in ("garch", "garch_t"):
        a, b = float(row["g_alpha"]), float(row["g_beta"])
        if a + b >= 0.98:
            b = 0.98 - a
        w = 1 - a - b
        h = lfilter([1.0], [1.0, -b], w + a * np.r_[0.0, e[:-1] ** 2], zi=[b * 1.0])[0]; h = np.maximum(h, 1e-8)   # h_{-1} = 1 (unconditional)
        if f == "garch_t":
            nu = max(float(row["t_nu"]) * 1.5, 4.5); sc = np.sqrt(nu / (nu - 2)); z = e / np.sqrt(h)
            lp = stats.t.logpdf(z * sc, nu) + np.log(sc) - 0.5 * np.log(h)
        else:
            lp = -0.5 * (LOG2PI + e * e / h + np.log(h))
    else:
        lp = logf(e, f, row)
    lp = lp - np.log(sig * s)
    return float(lp[lo:].sum())


def ll_sine(xz, npre, row, ph, mean, cfg, mech="none", size=0.0, lo=2):
    n = len(xz); sig = float(row["sine_sig"]); A = float(row["sine_amp"]); P = float(row["sine_period"])
    sg = np.full(n, sig); Aa = np.full(n, A); om = np.full(n, 2 * np.pi / P)
    if mech == "scale":
        sg[npre:] *= np.exp(size)
    elif mech == "amp":
        Aa[npre:] *= np.exp(size)
    elif mech == "period":
        om[npre:] /= np.exp(size)
    phase = ph + np.cumsum(np.r_[0.0, om[:-1]])
    r = xz - mean - Aa * np.sin(phase)
    return float((-0.5 * LOG2PI - np.log(sg) - 0.5 * (r / sg) ** 2)[lo:].sum())


def components(cfg, sine):
    W = cfg["sine_mech_w"] if sine else cfg["mech_w"]; tot = sum(W.values()); out = []
    for m, wm in W.items():
        for s, ps in zip(cfg[f"{m}_grid"], cfg[f"{m}_p"]):
            for sg in (1, -1):
                out.append((m, sg * s, wm / tot * ps * 0.5))
    return out


def loglr_integrated(xz, npre, row, cfg, c, ph, mean, K=4, seed=0):
    """post-segment marginal LR integrating theta_pre over an approximate (Laplace-style) posterior from the pre segment:
    LR = mean_k sum_m w_m p(post | theta_k, m) / mean_k p(post | theta_k, none), theta_k ~ N(theta_hat, SE^2)."""
    rng = np.random.default_rng(seed); n = npre
    sine = bool(row.get("is_sine", False)) and cfg.get("use_sine", True)
    comps = components(cfg, sine); lw = np.log([w for _, _, w in comps]); a1, a0 = [], []
    for k in range(K):
        r = dict(row); cc, pp, mm_ = c, ph, mean
        if k > 0:
            if sine:
                sg = float(r["sine_sig"]); A = max(float(r["sine_amp"]), 1e-6); P = float(r["sine_period"])
                r["sine_sig"] = sg * np.exp(rng.normal(0, 1 / np.sqrt(2 * n)))
                r["sine_amp"] = A + rng.normal(0, sg * np.sqrt(2 / n))
                r["sine_period"] = P * np.exp(rng.normal(0, min(0.05, 2 * sg / (A * n * np.sqrt(n / 6)) * P / (2 * np.pi))))
                pp = ph + rng.normal(0, sg / (A * np.sqrt(n / 2))); mm_ = mean + rng.normal(0, sg / np.sqrt(n))
            else:
                r["sig_best"] = float(r["sig_best"]) * np.exp(rng.normal(0, 1 / np.sqrt(2 * n)))
                ph1 = float(r["ar1_phi"]); se = np.sqrt(max(1 - ph1 * ph1, 1e-3) / n); dphi = rng.normal(0, se)
                for key in ("ar1_phi", "ar2_phi1", "arma_phi"):
                    r[key] = float(r[key]) + dphi
                cc = c + rng.normal(0, float(r["sd"]) / np.sqrt(n) * np.sqrt((1 + min(abs(ph1), .95)) / (1 - min(abs(ph1), .95))))
        if sine:
            l0 = ll_sine(xz, npre, r, pp, mm_, cfg, lo=npre); ll = np.array([ll_sine(xz, npre, r, pp, mm_, cfg, m, s, lo=npre) for m, s, _ in comps])
        else:
            l0 = ll_nonsine(xz, npre, r, cc, cfg, lo=npre); ll = np.array([ll_nonsine(xz, npre, r, cc, cfg, m, s, lo=npre) for m, s, _ in comps])
        a1.append(logsumexp(lw + ll)); a0.append(l0)
    return float(logsumexp(a1) - logsumexp(a0)), float(np.std(np.array(a1) - np.array(a0)))


def loglr(xz, npre, row, cfg, c=None, ph=None, mean=None, detail=False):
    sine = bool(row.get("is_sine", False)) and cfg.get("use_sine", True)
    comps = components(cfg, sine)
    if sine:
        l0 = ll_sine(xz, npre, row, ph, mean, cfg)
        ll = np.array([ll_sine(xz, npre, row, ph, mean, cfg, m, s) for m, s, _ in comps])
    else:
        l0 = ll_nonsine(xz, npre, row, c, cfg)
        ll = np.array([ll_nonsine(xz, npre, row, c, cfg, m, s) for m, s, _ in comps])
    lw = np.log([w for _, _, w in comps])
    tot = float(logsumexp(lw + ll) - l0)
    if not detail:
        return tot
    mechs = sorted(set(m for m, _, _ in comps)); out = {"llr": tot}
    post = lw + ll - logsumexp(np.r_[lw + ll, l0])                  # posterior over components (prior odds 1:1 vs no-break)
    for mname in mechs:
        k = [i for i, (m, _, _) in enumerate(comps) if m == mname]
        out[f"llr_{mname}"] = float(logsumexp(lw[k] + ll[k]) - logsumexp(lw[k]) - l0)
        out[f"pm_{mname}"] = float(np.exp(logsumexp(post[k])))
    return out


def est_row(zpre):
    """fit the generator's model set on the PRE segment only (pre-scaled units)."""
    import dgp_fit, dgp_fit2
    r = dgp_fit.fit_seg(zpre); s = dgp_fit2.fit_sine(zpre)
    bb = min(r[f"bic_{k}"] for k in dgp_fit.MEAN_MODELS)
    r.update({k: s[k] for k in ("sine_bic", "sine_amp", "sine_period", "sine_sig", "sine_phi")})
    r["is_sine"] = (bb - s["sine_bic"]) > 10
    n = len(zpre); t = np.arange(n, dtype=float); f = 1 / s["sine_period"] if np.isfinite(s["sine_period"]) else 0.0
    X = np.column_stack([np.ones(n), np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    b, *_ = np.linalg.lstsq(X, zpre, rcond=None)
    r["sine_c"] = float(b[0]); r["sine_ph"] = float(np.arctan2(b[2], b[1]))
    return r


def prescale(x, npre):
    pre = x[:npre]; med = np.median(pre); s = max(1.4826 * np.median(np.abs(pre - med)), 0.25 * pre.std())
    s = s if s > 1e-12 * (abs(med) + 1) else 1.0
    return (x - med) / s


def est_llr(x, npre, cfg, detail=False):
    z = prescale(np.asarray(x, float), npre)
    if np.std(z[:npre]) < 1e-9:
        return ({"llr": 0.0} if detail else 0.0), None
    r = est_row(z[:npre])
    tt = np.arange(len(z)) / len(z)
    tr = float(r["trend_slope"]) * (tt - npre / len(z) / 2) if r["mean_model"] in ("trend", "ar1t", "ar2t") else 0.0
    c = float(np.mean((z - tr)[:npre]))
    out = loglr(z, npre, r, cfg, c=c, ph=r["sine_ph"], mean=r["sine_c"], detail=detail)
    if detail:
        out["llr_int"], out["llr_int_sd"] = loglr_integrated(z, npre, r, cfg, c, r["sine_ph"], r["sine_c"])
    return out, r


def _ceiling_chunk(items, cfg):
    out = []
    for sid, x, npre, row, c, ph in items:
        z = (np.asarray(x, float) - float(row["raw_med"])) / float(row["raw_scale"])
        o = loglr(z, npre, row, cfg, c=c, ph=ph, mean=float(row["mean"]))
        e, _ = est_llr(x, npre, cfg)
        zz = prescale(np.asarray(x, float), npre); p = zz[:npre] - zz[:npre].mean()
        out.append((sid, o, e, float(p[1:] @ p[:-1] / max(p @ p, 1e-12))))
    return out


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score
    import extract
    tag = sys.argv[2]; D = dgp.D
    cfg = dict(dgp.CFG, **json.load(open(D / "cfg_ceiling.json")))
    meta = pd.read_parquet(D / f"sim_{tag}_meta.parquet"); fits = dgp.load_fits()
    ids, ser = extract.load_series(D / f"sim_{tag}.parquet"); pos = {k: j for j, k in enumerate(ids)}
    items = [(i, ser[pos[i]][0], ser[pos[i]][1], fits.loc[meta.at[i, "src"]], meta.at[i, "c_off"], meta.at[i, "phase"]) for i in meta.index]
    t0 = time.time()
    ch = [items[i:i + 200] for i in range(0, len(items), 200)]
    res = [r for part in Parallel(n_jobs=11)(delayed(_ceiling_chunk)(c, cfg) for c in ch) for r in part]
    R = pd.DataFrame(res, columns=["id", "oracle", "est", "acf1"]).set_index("id").join(meta)
    R.to_parquet(D / f"ceiling_{tag}.parquet")
    y = R.label.values; npost = R.n - R.npre
    sl = {"all": np.ones(len(R), bool), "acf>0.6": (R.acf1 > 0.6).values, "ntot<=350": (R.n <= 350).values, "npost<=150": (npost <= 150).values}
    out = {k: {"n": int(m.sum()), "oracle_auc": round(roc_auc_score(y[m], R.oracle.values[m]), 4), "estimated_auc": round(roc_auc_score(y[m], R.est.values[m]), 4)} for k, m in sl.items()}
    out["by_base"] = {mm: round(roc_auc_score(R.label[R.mean_model == mm], R.oracle[R.mean_model == mm]), 4) for mm in R.mean_model.unique() if (R.mean_model == mm).sum() > 100}
    out["runtime_s"] = round(time.time() - t0)
    (D / f"ceiling_{tag}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
