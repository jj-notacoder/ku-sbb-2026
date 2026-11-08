"""v4 P1d: recovered-generator simulator (deterministic from one seed).
Base process = EMPIRICAL PRIOR: a fitted pre-segment model (mean model, parameters, innovation family) resampled from
the P1a fits of real DEV training series (work/v3/dgp/fits_pre.parquet), with that series' own lengths and raw
location/scale. Label ~ Bernoulli(0.5). Positives get one mechanism at the boundary (weights / effect grids in CFG).
All effect sizes are relative to the process scale (sd of the pre segment / innovation sd), per the A3 forensics.
  python src/dgp.py <n> <seed> [cfg.json] -> work/v3/dgp/sim_<tag>.parquet (+ labels/meta)"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = __import__("paths").CONFIG   # generator config (research-only P1 fit tables are in the full archive)

CFG = {
    "mech_w": {"level": 0.40, "scale": 0.35, "phi": 0.25},
    "level_grid": [0.5, 1.0, 2.0], "level_p": [0.55, 0.25, 0.20],          # x pre-process sd, random sign
    "scale_grid": [0.3466, 0.6931], "scale_p": [0.6, 0.4],                # log sd ratio (var x2, x4), random sign
    "phi_grid": [0.1, 0.2, 0.3], "phi_p": [0.4, 0.35, 0.25],              # AR(1)-coefficient change, random sign
    "transient_rate_neg": 0.0, "transient_rate_pos": 0.0,                 # brief fluctuation then return
    "transient_len": [5, 30], "transient_amp": [2.0, 5.0],
    "burn": 200,
    # iteration 2: deterministic sinusoid family (fits2) + its mechanisms; spike contamination (label-independent)
    "use_sine": True,
    "sine_mech_w": {"scale": 0.45, "period": 0.35, "amp": 0.20},
    "period_grid": [0.3, 0.4], "period_p": [0.5, 0.5],                    # log period ratio, random sign
    "amp_grid": [0.35, 0.7], "amp_p": [0.5, 0.5],                         # log amplitude ratio, random sign
    "spike_rate": 0.08, "spike_amp": [4.0, 8.0],
    # iteration 3: innovation tails from the ML t fit for every row; GARCH whenever it beats t; smaller level shifts
    "tails_from_t_fit": True, "garch_if_beats_t": True, "level_unit_rw": "sig",
}


def eff_family(row, cfg=CFG):
    """innovation family actually simulated for a fitted row (shared with the oracle likelihood)."""
    f = row["family"]
    gmode = cfg.get("garch_rule", "beats_t")
    if (gmode == "beats_t" and row["fbic_garch"] < min(row["fbic_t"], row["fbic_gauss"]) - 2) or        (gmode == "beats_gauss" and row["fbic_garch"] < row["fbic_gauss"] and float(row["g_alpha"]) > 0.03):
        f = "garch_t" if (cfg.get("garch_t_innov") and float(row["t_nu"]) < 200) else "garch"
    elif cfg.get("tails_from_t_fit") and f in ("gauss", "t") and float(row["t_nu"]) < 200:
        f = "t"
    return f


def _std_innov(row, n, rng, cfg=CFG):
    f = eff_family(row, cfg)
    if f == "t":
        nu = max(float(row["t_nu"]), 2.2); e = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    elif f == "laplace":
        e = rng.laplace(0, 1 / np.sqrt(2), n)
    elif f == "uniform":
        e = rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    elif f == "skewnorm":
        from scipy import stats
        a = float(row["skew_a"]); d = a / np.sqrt(1 + a * a); m = d * np.sqrt(2 / np.pi); v = 1 - m * m
        e = (stats.skewnorm.rvs(a, size=n, random_state=rng) - m) / np.sqrt(v)
    elif f == "gmm2":
        w, sep, sdr = float(row["gmm_w"]), float(row["gmm_sep"]), float(row["gmm_sdr"])
        c = rng.random(n) < w
        e = np.where(c, sep + sdr * rng.standard_normal(n), rng.standard_normal(n))
        e = (e - e.mean()) / (e.std() + 1e-12)
    elif f in ("garch", "garch_t"):
        a, b = float(row["g_alpha"]), float(row["g_beta"])
        if a + b >= 0.98:
            b = 0.98 - a
        w = 1 - a - b; e = np.empty(n); h = 1.0; prev = 0.0
        if f == "garch_t":
            nu = max(float(row["t_nu"]) * 1.5, 4.5); z = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
        else:
            z = rng.standard_normal(n)
        for i in range(n):
            h = w + a * prev * prev + b * h; e[i] = np.sqrt(h) * z[i]; prev = e[i]
    else:
        e = rng.standard_normal(n)
    return e


def _ar_coefs(row):
    mm = row["mean_model"]
    if mm in ("ar1", "ar1t"):
        return [float(row["ar1_phi"])], 0.0
    if mm in ("ar2", "ar2t"):
        return [float(row["ar2_phi1"]), float(row["ar2_phi2"])], 0.0
    if mm == "arma11":
        return [float(row["arma_phi"])], float(row["arma_theta"])
    return [], 0.0


def _stab(phi):
    if not phi:
        return phi
    r = np.max(np.abs(np.roots([1] + [-p for p in phi])))
    if r > 0.995:
        phi = [p * (0.995 / r) ** (j + 1) for j, p in enumerate(phi)]
    return phi


def simulate_sine(row, ntot, npre, label, rng, cfg, lat=None):
    t = np.arange(ntot, dtype=float); sig = float(row["sine_sig"]); A = float(row["sine_amp"]); P = float(row["sine_period"])
    ph = float(rng.uniform(0, 2 * np.pi))
    if lat is not None:
        lat["ph"] = ph
    mech, size = "none", 0.0
    if label == 1:
        ks = list(cfg["sine_mech_w"]); w = np.array([cfg["sine_mech_w"][k] for k in ks]); mech = ks[rng.choice(len(ks), p=w / w.sum())]
        size = float(rng.choice(cfg[f"{mech}_grid"], p=cfg[f"{mech}_p"])) * (1 if rng.random() < 0.5 else -1)
    sg = np.full(ntot, sig); Aa = np.full(ntot, A); om = np.full(ntot, 2 * np.pi / P)
    if mech == "scale":
        sg[npre:] *= np.exp(size)
    elif mech == "amp":
        Aa[npre:] *= np.exp(size)
    elif mech == "period":
        om[npre:] /= np.exp(size)
    phase = ph + np.cumsum(np.r_[0.0, om[:-1]])                           # phase-continuous at the boundary
    x = float(row["mean"]) + Aa * np.sin(phase) + sg * rng.standard_normal(ntot)
    return x, mech, size


def _spikes(x, row, rng, cfg):
    if cfg.get("spike_rate", 0) > 0 and rng.random() < cfg["spike_rate"]:
        k = 1 + rng.poisson(0.5); at = rng.integers(0, len(x), k)
        x[at] += rng.uniform(*cfg["spike_amp"], k) * float(row["sd"]) * np.where(rng.random(k) < 0.5, -1, 1)
    return x


def simulate_one(row, ntot, npre, label, rng, cfg=CFG, lat=None):
    if cfg.get("use_sine") and row.get("is_sine", False):
        x, mech, size = simulate_sine(row, ntot, npre, label, rng, cfg, lat)
        x = _spikes(x, row, rng, cfg)
        return float(row["raw_med"]) + float(row["raw_scale"]) * x, mech, size
    B = cfg["burn"]; n = ntot + B; b0 = B + npre
    mm = row["mean_model"]; sig = float(row["sig_best"])
    e = _std_innov(row, n, rng, cfg)
    mech, size = "none", 0.0
    if label == 1:
        ks = list(cfg["mech_w"]); w = np.array([cfg["mech_w"][k] for k in ks]); mech = ks[rng.choice(len(ks), p=w / w.sum())]
        g, p = cfg[f"{mech}_grid"], cfg[f"{mech}_p"]
        size = float(rng.choice(g, p=p)) * (1 if rng.random() < 0.5 else -1)
    s = np.ones(n)
    if mech == "scale":
        s[b0:] = np.exp(size)
    u = sig * s * e
    phi, theta = _ar_coefs(row)
    phi = _stab(phi)
    phi_post = list(phi)
    if mech == "phi":
        base = phi if phi else [0.0]
        phi_post = _stab([base[0] + size] + list(base[1:])); phi = list(base) if not phi else phi
    if theta:
        u = u + theta * np.r_[0.0, u[:-1]]
    x = np.empty(n)
    if mm in ("rw", "ll"):
        x = np.cumsum(u)
    elif phi or mech == "phi":
        p0 = phi + [0.0] * (2 - len(phi)) if phi else [0.0, 0.0]
        p1 = phi_post + [0.0] * (2 - len(phi_post))
        x[0] = u[0]; x[1] = u[1]
        for t in range(2, n):
            a = p0 if t < b0 else p1
            x[t] = a[0] * x[t - 1] + a[1] * x[t - 2] + u[t]
    else:
        x = u.copy()
    x = x[B:]
    tt = np.arange(ntot) / ntot
    if mm in ("trend", "ar1t", "ar2t"):
        x = x + float(row["trend_slope"]) * (tt - npre / ntot / 2)
    c_off = float(row["mean"]) - x[:npre].mean(); x = x + c_off
    if lat is not None:
        lat["c"] = c_off
    if mech == "level":
        unit = float(row["sig_best"]) if (cfg.get("level_unit_rw") == "sig" and mm in ("rw", "ll", "trend", "ar1t", "ar2t")) else float(row["sd"])
        x[npre:] += size * unit
    # transient (brief fluctuation then return) - label unchanged
    tr = cfg["transient_rate_pos"] if label == 1 else cfg["transient_rate_neg"]
    if tr > 0 and rng.random() < tr:
        L = int(rng.integers(cfg["transient_len"][0], cfg["transient_len"][1] + 1))
        lo = npre if cfg.get("transient_where", "post") == "post" else 0
        st = int(rng.integers(lo, max(lo + 1, ntot - L)))
        x[st:st + L] += float(rng.uniform(*cfg["transient_amp"])) * float(row["sd"]) * (1 if rng.random() < 0.5 else -1)
    x = _spikes(x, row, rng, cfg)
    return float(row["raw_med"]) + float(row["raw_scale"]) * x, mech, size


def load_fits():
    P = pd.read_parquet(D / "fits_pre.parquet")
    S = pd.read_parquet(D / "fits2_pre.parquet").loc[P.index]
    bb = P[[f"bic_{k}" for k in ("iid", "trend", "ar1", "ar2", "ar1t", "ar2t", "arma11", "rw", "ll")]].min(1)
    P = P.join(S[["sine_bic", "sine_amp", "sine_period", "sine_sig", "sine_phi"]])
    P["is_sine"] = (bb - P["sine_bic"]) > 10
    return P


def generate(n, seed, cfg=CFG, fits=None):
    fits = load_fits() if fits is None else fits
    from extract import load_series
    sys.path.insert(0, str(ROOT / "src"))
    ids, ser = load_series(__import__("paths").DATA / "train_series.parquet"); pos = {k: j for j, k in enumerate(ids)}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(fits), n)
    rows, meta = [], []
    for j, r in enumerate(pick):
        row = fits.iloc[r]; src = fits.index[r]
        x0, npre = ser[pos[src]]; ntot = len(x0)
        lab = int(rng.random() < 0.5)
        lat = {}
        x, mech, size = simulate_one(row, ntot, npre, lab, rng, cfg, lat)
        sid = f"syn{seed}_{j:06d}"
        rows.append(pd.DataFrame({"id": sid, "time": np.arange(ntot), "value": x, "period": (np.arange(ntot) >= npre).astype(int)}))
        meta.append(dict(id=sid, label=lab, mech=mech, size=size, src=src, family=row["family"], mean_model="sine" if (cfg.get("use_sine") and row["is_sine"]) else row["mean_model"], n=ntot, npre=npre, c_off=lat.get("c", np.nan), phase=lat.get("ph", np.nan)))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(meta).set_index("id")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    cfg = dict(CFG, **json.load(open(sys.argv[3]))) if len(sys.argv) > 3 else CFG
    df, meta = generate(n, seed, cfg)
    tag = f"{seed}_" + hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    df.to_parquet(D / f"sim_{tag}.parquet", index=False); meta.to_parquet(D / f"sim_{tag}_meta.parquet")
    print(tag, len(meta), meta.mech.value_counts().to_dict(), flush=True)
