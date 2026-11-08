"""v3 phase 1: simulation-calibrated evidence (parametric bootstrap of a PRE-only null model).
Per series (on the pre-robust-scaled z, so everything is affine invariant):
 1a null model on PRE only: AR(p), p by BIC over 0..5, with / without a linear trend; companion spectral radius
    constrained to <= 0.995; residuals standardised.
 1b B continuations of length n_post conditioned on the last p pre values, innovations resampled from the
    standardised pre residuals (IID or moving-block). ONE fixed RNG seed reused identically for every series, so a
    series' features depend only on that series (batch / order / single-series invariant, deterministic).
 1c calibrated battery: for each statistic T compare the real post with the B simulated posts:
    rank, z vs simulated mean/SD, log tail probability (floored at 1/(B+1)), simulated SD; + Fisher/Stouffer/min/max.
 1d calibrated likelihood ratios (AR Chow, Gaussian coef+variance LR, piecewise-vs-common trend LR): raw + same calibration.
 1e null-model adequacy (p, trend, spectral radius, residual ACF/kurtosis, Ljung-Box, R^2, BIC gain).
Column prefixes: c_ (1c), d_ (1d), e_ (1e). Families: sim (IID, B=200), simmbb (moving-block, B=200), simB{50,100,400}.
Caching: extract/views hash this whole module for families with _hash_module=True."""
import numpy as np
from scipy.signal import lfilter, lfiltic
from scipy.special import ndtri
from feats import FAMILIES, EPS

SIM_SEED = 20261104
PMAX = 5
RHO_MAX = 0.995
TWO, UP = 0, 1
NB_SPEC = 16
NFFT = 1024


# ------------------------------------------------------------------ 1a null model
def _design(pre, p, trend, start):
    n = len(pre)
    cols = [np.ones(n - start)]
    if trend:
        cols.append(np.arange(start, n) / n)
    for j in range(1, p + 1):
        cols.append(pre[start - j:n - j])
    return np.column_stack(cols), pre[start:]


def _roots_radius(phi):
    if len(phi) == 0:
        return 0.0
    comp = np.zeros((len(phi), len(phi)))
    comp[0] = phi
    if len(phi) > 1:
        comp[1:, :-1] = np.eye(len(phi) - 1)
    return float(np.abs(np.linalg.eigvals(comp)).max())


def fit_null(pre):
    n = len(pre)
    pmax = int(min(PMAX, max(0, (n - 20) // 10)))
    best = None
    for p in range(pmax + 1):
        for tr in (0, 1):
            X, y = _design(pre, p, tr, pmax)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            ssr = max(float(r @ r), 1e-12 * len(y))
            bic = len(y) * np.log(ssr / len(y)) + X.shape[1] * np.log(len(y))
            if p == 0 and tr == 0:
                bic0 = bic
            if best is None or bic < best[0]:
                best = (bic, p, tr)
    _, p, tr = best
    X, y = _design(pre, p, tr, p)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    k0 = 1 + tr
    phi = beta[k0:].copy()
    rho = _roots_radius(phi)
    constrained = rho > RHO_MAX
    if constrained:   # shrink all companion roots by RHO_MAX / rho, then re-estimate the deterministic part
        phi = phi * (RHO_MAX / rho) ** np.arange(1, p + 1)
        yy = y - X[:, k0:] @ phi
        det, *_ = np.linalg.lstsq(X[:, :k0], yy, rcond=None)
        beta = np.concatenate([det, phi])
    e = y - X @ beta
    sig = float(e.std())
    return dict(p=p, trend=tr, c=float(beta[0]), d=float(beta[1]) if tr else 0.0, phi=phi, e=e, sig=sig,
                rho=rho, constrained=constrained, bic_gain=float((bic0 - best[0]) / len(y)), n=n)


def mu_path(M, n, t0, m):
    t = (t0 + np.arange(m)) / n
    return M["c"] + M["d"] * t if M["trend"] else np.full(m, M["c"])


def simulate(M, pre, m, B, boot, rng):
    e = M["e"]
    es = (e - e.mean()) / (e.std() + EPS)
    ne = len(es)
    if boot == "mbb":
        L = max(1, int(round(M["n"] ** (1 / 3))))
        L = min(L, ne)
        nb = -(-m // L)
        st = rng.integers(0, ne - L + 1, size=(nb, B))
        idx = (st[:, None, :] + np.arange(L)[None, :, None]).reshape(nb * L, B)[:m]
    else:
        idx = rng.integers(0, ne, size=(m, B))
    u = M["sig"] * es[idx] + mu_path(M, M["n"], M["n"], m)[:, None]
    p = M["p"]
    if p == 0:
        return u
    a = np.concatenate([[1.0], -M["phi"]])
    zi = lfiltic([1.0], a, y=pre[::-1][:p])
    return lfilter([1.0], a, u, axis=0, zi=np.repeat(zi[:, None], B, 1))[0]


def innovations(M, pre, Bm):
    """null-model one-step innovations of each post column (lags from the pre end), in units of sigma."""
    m = Bm.shape[0]; p = M["p"]
    mu = mu_path(M, M["n"], M["n"], m)[:, None]
    e = Bm - mu
    if p:
        full = np.vstack([np.repeat(pre[-p:][:, None], Bm.shape[1], 1), Bm])
        for j in range(1, p + 1):
            e = e - M["phi"][j - 1] * full[p - j:p - j + m]
    return e / (M["sig"] + EPS)


# ------------------------------------------------------------------ statistics (vectorised over columns)
def _acf_cols(bc, d, k):
    if bc.shape[0] <= k + 1:
        return np.zeros(bc.shape[1])
    return (bc[:-k] * bc[k:]).sum(0) / d


def _band_logspec(xc):
    P = np.abs(np.fft.rfft(xc, NFFT, axis=0)) ** 2 / xc.shape[0]
    P = P[1:1 + (NFFT // 2) // NB_SPEC * NB_SPEC]
    P = P.reshape(NB_SPEC, -1, *P.shape[1:]).mean(1)
    return np.log(P + 1e-10)


def _longest_run_rows(O):
    """longest run of True along axis 1 for each row (vectorised)."""
    c = np.cumsum(O, 1)
    reset = np.maximum.accumulate(np.where(O, 0, c), 1)
    return (c - reset).max(1).astype(float)


def _quant_sorted(S, q):
    """linear-interpolated quantiles of row-sorted matrix S (K, m) at probs q -> (K, len(q))."""
    m = S.shape[1]
    pos = q * (m - 1); lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, m - 1); w = pos - lo
    return S[:, lo] * (1 - w) + S[:, hi] * w


def _ks_rows(Sa, S):
    """two-sample KS sup between sorted ref Sa and each row of row-sorted S (K, m)."""
    K, m = S.shape; n = len(Sa)
    i = np.arange(1, m + 1)
    Fr = np.searchsorted(Sa, S.ravel(), "right").reshape(K, m) / n
    Fl = np.searchsorted(Sa, S.ravel(), "left").reshape(K, m) / n
    return np.maximum(np.abs(i / m - Fr), np.abs((i - 1) / m - Fl)).max(1)


QG = (np.arange(50) + 0.5) / 50


def battery(pre, Bm, M):
    """returns dict name -> (kind, values over columns)."""
    n, m = len(pre), Bm.shape[0]
    out = {}
    ma, va = pre.mean(), pre.var() + EPS
    mb = Bm.mean(0); vb = Bm.var(0) + EPS
    St = np.sort(Bm.T, 1); Sa = np.sort(pre)
    out["mshift"] = (TWO, mb - ma)
    out["medshift"] = (TWO, _quant_sorted(St, np.array([0.5]))[:, 0] - np.median(pre))
    out["lvr"] = (TWO, np.log(vb / va))
    out["ks"] = (UP, _ks_rows(Sa, St))
    out["w1"] = (UP, np.abs(_quant_sorted(St, QG) - np.quantile(pre, QG)[None]).mean(1))
    lo1, hi1, lo10, hi10 = np.quantile(pre, [0.01, 0.99, 0.10, 0.90])
    out["exc"] = (UP, np.maximum(np.maximum(St[:, -1] - hi1, lo1 - St[:, 0]), 0.0))
    O = (Bm.T < lo10) | (Bm.T > hi10)
    out["runout"] = (UP, _longest_run_rows(O))
    out["fracout"] = (TWO, O.mean(1))
    ta = np.arange(n) - (n - 1) / 2; tb = np.arange(m) - (m - 1) / 2
    sl_a = (ta @ pre) / (ta @ ta); sl_b = (tb @ Bm) / (tb @ tb)
    out["slope"] = (TWO, (sl_b - sl_a) * n)
    ac = pre - ma; da = ac @ ac + EPS
    bc = Bm - mb; db = (bc * bc).sum(0) + EPS
    r = {}
    for k in (1, 2, 5):
        ra = float(ac[:-k] @ ac[k:] / da) if n > k + 1 else 0.0
        r[k] = (ra, _acf_cols(bc, db, k))
        out[f"dacf{k}"] = (TWO, r[k][1] - ra)
    # AR(2) Yule-Walker coefficient Mahalanobis distance (pre asymptotic covariance)
    def yw(r1, r2):
        den = 1 - r1 * r1 + 1e-6
        return r1 * (1 - r2) / den, (r2 - r1 * r1) / den
    a1, a2 = yw(r[1][0], r[2][0]); b1, b2 = yw(r[1][1], r[2][1])
    V = np.array([[1 - a2 * a2, -a1 * (1 + a2)], [-a1 * (1 + a2), 1 - a2 * a2]]) * (1 / n + 1 / m)
    Vi = np.linalg.pinv(V + 1e-4 * np.eye(2) * (1 / n + 1 / m))
    D = np.vstack([b1 - a1, b2 - a2])
    out["ar2maha"] = (UP, np.einsum("ik,ij,jk->k", D, Vi, D))
    La = _band_logspec(ac); Lb = _band_logspec(bc)
    out["lsd"] = (UP, np.sqrt(((Lb - La[:, None]) ** 2).mean(0)))
    E = innovations(M, pre, Bm)
    es = M["e"] / (M["sig"] + EPS); Se = np.sort(es)
    out["inn_mean"] = (TWO, E.mean(0) * np.sqrt(m))
    out["inn_lvr"] = (TWO, np.log(E.var(0) + EPS))
    out["inn_cusum"] = (UP, np.abs(np.cumsum(E - es.mean(), 0)).max(0) / np.sqrt(m))
    out["inn_ks"] = (UP, _ks_rows(Se, np.sort(E.T, 1)))
    out["inn_maxabs"] = (UP, np.abs(E).max(0))
    k3 = max(m // 3, 1)
    out["last3"] = (TWO, Bm[-k3:].mean(0) - ma)
    return out


def lrs(pre, Bm, M):
    n, m = len(pre), Bm.shape[0]
    out = {}
    pc = int(min(max(M["p"], 1), 4, max(1, min(n, m) // 10)))
    k = pc + 1
    Xa = np.column_stack([np.ones(n - pc)] + [pre[pc - j:n - j] for j in range(1, pc + 1)]); ya = pre[pc:]
    Ga = Xa.T @ Xa; ha = Xa.T @ ya; yya = ya @ ya
    Xb = np.stack([np.ones((m - pc, Bm.shape[1]))] + [Bm[pc - j:m - j] for j in range(1, pc + 1)], 1)  # (t, k, K)
    yb = Bm[pc:]
    Gb = np.einsum("tik,tjk->kij", Xb, Xb); hb = np.einsum("tik,tk->ki", Xb, yb); yyb = (yb * yb).sum(0)
    rid = 1e-8 * np.eye(k)
    ssr_a = max(yya - ha @ np.linalg.solve(Ga + rid, ha), 1e-10)
    ssr_b = np.maximum(yyb - np.einsum("ki,ki->k", hb, np.linalg.solve(Gb + rid, hb[..., None])[..., 0]), 1e-10)
    Gc = Ga[None] + Gb; hc = ha[None] + hb
    ssr_c = np.maximum(yya + yyb - np.einsum("ki,ki->k", hc, np.linalg.solve(Gc + rid, hc[..., None])[..., 0]), 1e-10)
    Na, Nb = n - pc, m - pc; N = Na + Nb
    ssr_c = np.maximum(ssr_c, ssr_a + ssr_b)
    out["chow"] = (UP, np.log1p(((ssr_c - ssr_a - ssr_b) / k) / ((ssr_a + ssr_b) / max(N - 2 * k, 1))))
    out["lrfull"] = (UP, N * np.log(ssr_c / N) - Na * np.log(ssr_a / Na) - Nb * np.log(ssr_b / Nb))
    # piecewise (separate line per segment) vs common line
    ta = np.arange(n, dtype=float); tb = np.arange(n, n + m, dtype=float)
    def line_ssr(t, Y):
        tc = t - t.mean(); Yc = Y - Y.mean(0)
        return (Yc * Yc).sum(0) - (tc @ Yc) ** 2 / (tc @ tc)
    sep = line_ssr(ta, pre) + line_ssr(tb, Bm)
    t_all = np.concatenate([ta, tb]); tm = t_all.mean(); stt = ((t_all - tm) ** 2).sum()
    sy = pre.sum() + Bm.sum(0); sty = (ta - tm) @ pre + (tb - tm) @ Bm
    yy = pre @ pre + (Bm * Bm).sum(0)
    com = yy - sy ** 2 / (n + m) - sty ** 2 / stt
    out["trendlr"] = (UP, (n + m) * np.log(np.maximum(com, 1e-10) / np.maximum(sep, 1e-10)))
    return out


def _calib(f, pfx, name, kind, v, raw=False):
    T, S = v[0], v[1:]
    B = len(S)
    f[f"{pfx}{name}_rk"] = (np.sum(S < T) + 0.5 * np.sum(S == T)) / B
    mu, sd = S.mean(), S.std()
    f[f"{pfx}{name}_z"] = (T - mu) / max(sd, 1e-9)
    nu, nl = 1 + np.sum(S >= T), 1 + np.sum(S <= T)
    p = nu / (B + 1) if kind == UP else min(1.0, 2 * min(nu, nl) / (B + 1))
    f[f"{pfx}{name}_lp"] = np.log(p)
    f[f"{pfx}{name}_sd"] = sd
    if raw:
        f[f"{pfx}{name}_raw"] = T
    return p


def adequacy(f, M):
    e = M["e"]; es = (e - e.mean()) / (e.std() + EPS); n = len(es)
    f["e_p"] = M["p"]; f["e_trend"] = M["trend"]; f["e_rho"] = M["rho"]; f["e_constrained"] = float(M["constrained"])
    r = [float(es[:-k] @ es[k:] / (es @ es + EPS)) if n > k + 1 else 0.0 for k in range(1, 11)]
    f["e_racf1"], f["e_racf2"] = r[0], r[1]
    f["e_rkurt"] = float((es ** 4).mean() - 3)
    f["e_lb10"] = float(np.log1p(n * (n + 2) * sum(rk * rk / max(n - k, 1) for k, rk in enumerate(r, 1))))
    f["e_bic_gain"] = M["bic_gain"]


def sim_family(c, B=200, boot="iid"):
    pre, post = c.pre, c.post
    if len(pre) < 12 or len(post) < 8 or pre.std() < 1e-9:
        return {}
    M = fit_null(pre)
    if not np.isfinite(M["sig"]) or M["sig"] < 1e-9:
        return {}
    rng = np.random.default_rng(SIM_SEED)
    sims = simulate(M, pre, len(post), B, boot, rng)
    Bm = np.column_stack([post, sims])
    f = {}
    ps = []
    for nm, (kind, v) in battery(pre, Bm, M).items():
        ps.append((_calib(f, "c_", nm, kind, v), f[f"c_{nm}_rk"]))
    P = np.array([p for p, _ in ps]); R = np.array([r for _, r in ps])
    f["c_fisher"] = -2 * np.log(P).sum()
    f["c_stouffer"] = ndtri(np.clip(1 - P, 1e-12, 1 - 1e-12)).sum() / np.sqrt(len(P))   # clip is only the probit domain guard
    f["c_minrank"] = R.min(); f["c_maxrank"] = R.max(); f["c_minlp"] = np.log(P).min()
    f["c_n01"] = float((P <= 0.01).sum())
    for nm, (kind, v) in lrs(pre, Bm, M).items():
        _calib(f, "d_", nm, kind, v, raw=True)
    adequacy(f, M)
    return {k: float(v) for k, v in f.items()}


def fam_sim(c):
    return sim_family(c, 200, "iid")


def fam_simmbb(c):
    return sim_family(c, 200, "mbb")


def fam_simB50(c):
    return sim_family(c, 50, "iid")


def fam_simB100(c):
    return sim_family(c, 100, "iid")


def fam_simB400(c):
    return sim_family(c, 400, "iid")


for _fn in (fam_sim, fam_simmbb, fam_simB50, fam_simB100, fam_simB400):
    _fn._hash_module = True
FAMILIES.update({"sim": fam_sim, "simmbb": fam_simmbb, "simB50": fam_simB50, "simB100": fam_simB100, "simB400": fam_simB400})
