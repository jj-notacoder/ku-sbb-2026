"""Phase D/E families: transform battery, distribution-family, AR Wald/Chow, Whittle/log-spectral, near-unit-root block.
Same contract: single series, pre-boundary robust scaling, no clipping, affine invariant, finite with fallbacks."""
import numpy as np
from scipy.special import ndtri, digamma
from scipy.ndimage import uniform_filter1d
from feats import (FAMILIES, Ctx, acf, lrv_bartlett, ks_stat, lratio, ols, ar_design, pit, longest_run, kurt, skew, EPS)


def _rs(v, n):
    """pre-robust scale of a view: (v - med_pre)/max(MAD_pre, .25 std_pre)."""
    p = v[:n]
    m = np.median(p); s = max(1.4826 * np.median(np.abs(p - m)), 0.25 * p.std(), 1e-9)
    return (v - m) / s


def _battery(f, pfx, a, b, full=True):
    na, nb = len(a), len(b)
    if na < 4 or nb < 4:
        return
    ma, mb, sa, sb = a.mean(), b.mean(), a.std(), b.std()
    f[f"{pfx}_welch"] = (mb - ma) / np.sqrt(sa ** 2 / na + sb ** 2 / nb + EPS)
    allv = np.concatenate([a, b])
    r = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    f[f"{pfx}_mw"] = (r[na:].sum() - nb * (nb + 1) / 2) / (na * nb) - 0.5
    f[f"{pfx}_ks"] = ks_stat(a, b)[0]
    f[f"{pfx}_lsr"] = lratio(sb, sa)
    f[f"{pfx}_med"] = np.median(b) - np.median(a)
    if not full:
        return
    g = np.linspace(0.02, 0.98, 25)
    f[f"{pfx}_wass"] = np.abs(np.quantile(b, g) - np.quantile(a, g)).mean()
    da, db = np.abs(a - np.median(a)), np.abs(b - np.median(b))
    f[f"{pfx}_bf"] = (db.mean() - da.mean()) / np.sqrt(da.var() / na + db.var() / nb + EPS)
    qa, qb = np.quantile(a, [0.025, 0.1, 0.9, 0.975]), np.quantile(b, [0.1, 0.9])
    f[f"{pfx}_q10"] = qb[0] - qa[1]; f[f"{pfx}_q90"] = qb[1] - qa[2]
    for k in (1, 2, 5):
        f[f"{pfx}_acf{k}d"] = acf(b, k) - acf(a, k)
    f[f"{pfx}_band"] = ((b < qa[0]) | (b > qa[3])).mean() - 0.05
    f[f"{pfx}_hac"] = (mb - ma) / np.sqrt(lrv_bartlett(a) / na + lrv_bartlett(b) / nb)


def _views(c):
    z, n = c.z, c.npre
    V = {}
    V["z"] = z
    dz = np.diff(z); V["dz"] = _rs(dz, n - 1)
    V["d2z"] = _rs(np.diff(dz), n - 2)
    V["az"] = _rs(np.abs(z), n)
    V["z2"] = _rs(z * z, n)
    V["adz"] = _rs(np.abs(dz), n - 1)
    V["zz1"] = _rs(z[1:] * z[:-1], n - 1)
    cs = np.cumsum(z - np.median(z[:n])); V["cs"] = _rs(cs, n)
    e1 = c.ar(1)[0]; V["ar1"] = _rs(e1, n - 1)
    e2 = c.ar(2)[0]; V["ar2"] = _rs(e2, n - 2)
    V["pit"] = ndtri(pit(np.sort(z[:n]), z))
    rk = (np.argsort(np.argsort(z, kind="mergesort"), kind="mergesort") + 0.5) / len(z)
    V["rank"] = rk
    offs = {"z": n, "dz": n - 1, "d2z": n - 2, "az": n, "z2": n, "adz": n - 1, "zz1": n - 1, "cs": n, "ar1": n - 1, "ar2": n - 2, "pit": n, "rank": n}
    return V, offs


def fam_battery(c):
    f = {}
    V, offs = _views(c)
    for k, v in V.items():
        nb = offs[k]
        _battery(f, f"bt_{k}", v[:nb], v[nb:])
    return f


def fam_battery_win(c):
    f = {}
    V, offs = _views(c)
    for k, v in V.items():
        nb = offs[k]
        for w in (16, 32, 64):
            ww = min(w, nb // 2, (len(v) - nb) // 2)
            if ww < 6:
                continue
            _battery(f, f"bw{w}_{k}", v[nb - ww:nb], v[nb:nb + ww], full=False)
    return f


# ---------------------------------------------------------------- distribution family
def _knn_entropy(x, k=4):
    s = np.sort(x); n = len(s)
    if n <= 2 * k + 1:
        return 0.0
    i = np.arange(n)
    lo = np.clip(i - k, 0, n - 1); hi = np.clip(i + k, 0, n - 1)
    # k-th neighbour distance in 1D: min over windows containing k neighbours
    cand = []
    for j in range(k + 1):
        a = np.clip(i - j, 0, n - 1); b = np.clip(i + (k - j), 0, n - 1)
        valid = (i - j >= 0) & (i + (k - j) <= n - 1)
        d = np.maximum(s[i] - s[a], s[b] - s[i]); d[~valid] = np.inf
        cand.append(d)
    eps = np.min(cand, 0)
    eps = np.maximum(eps, 1e-9)
    return float(digamma(n) - digamma(k) + np.log(2) + np.log(eps).mean())


def _dist_stats(x, edges):
    n = len(x); sd = x.std() + EPS
    q25, q75 = np.quantile(x, [0.25, 0.75])
    rng = np.ptp(x)
    out = {"range_std": rng / sd, "range_iqr": rng / (q75 - q25 + EPS), "kurt": kurt(x)}
    u = np.sort((x - x.min()) / (rng + EPS))
    out["unif_ks"] = float(np.abs(u - np.arange(1, n + 1) / n).max())
    h = np.histogram(x, edges)[0] + 0.5; p = h / h.sum()
    out["hent"] = float(-(p * np.log(p)).sum() / np.log(len(p)))
    g, k = skew(x), kurt(x)
    out["bimod"] = (g ** 2 + 1) / (k + 3 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) - 3 + EPS) if n > 3 else 0.0
    out["knn_h"] = _knn_entropy(x)
    # central-mass / shoulder ratios (flat-top detection)
    out["mass_1sd"] = float((np.abs(x - x.mean()) < sd).mean())
    out["mass_halfsd"] = float((np.abs(x - x.mean()) < 0.5 * sd).mean())
    return out


def fam_distfam(c):
    f = {}
    pre, post = c.pre, c.post
    edges = np.quantile(pre, np.linspace(0, 1, 17)); edges[0] = min(edges[0], post.min()) - 1e-9; edges[-1] = max(edges[-1], post.max()) + 1e-9
    edges = np.unique(edges)
    if len(edges) < 4:
        edges = np.linspace(min(pre.min(), post.min()) - 1e-9, max(pre.max(), post.max()) + 1e-9, 17)
    a, b = _dist_stats(pre, edges), _dist_stats(post, edges)
    for k in a:
        f[f"df_{k}_pre"] = a[k]; f[f"df_{k}_post"] = b[k]; f[f"df_{k}_d"] = b[k] - a[k]
    # same on AR(2) innovations (marginal shape of the noise, dependence removed)
    _, _, ep, eq = c.ar(2)
    e_edges = np.quantile(ep, np.linspace(0, 1, 17)); e_edges[0] = min(e_edges[0], eq.min()) - 1e-9; e_edges[-1] = max(e_edges[-1], eq.max()) + 1e-9
    e_edges = np.unique(e_edges)
    a, b = _dist_stats(ep, e_edges), _dist_stats(eq, e_edges)
    for k in a:
        f[f"dfe_{k}_d"] = b[k] - a[k]
    f["dfe_kurt_pre"] = a["kurt"]
    return f


# ---------------------------------------------------------------- AR Wald / Chow p=3..6, spectral distances
def fam_arwald(c):
    f = {}
    z, n = c.z, c.npre
    for p in (3, 4, 6):
        Xa, ya = ar_design(z[:n], p); Xb, yb = ar_design(z[n:], p); Xc, yc = ar_design(z, p)
        try:
            ba, ra, sa = ols(Xa, ya); bb, rb, sb = ols(Xb, yb); _, _, sc = ols(Xc, yc)
            va = sa / max(len(ya) - p - 1, 1); vb = sb / max(len(yb) - p - 1, 1)
            Ca = va * np.linalg.pinv(Xa.T @ Xa); Cb = vb * np.linalg.pinv(Xb.T @ Xb)
            d = bb - ba; S = Ca + Cb
            w_all = float(d @ np.linalg.pinv(S) @ d)
            w_dep = float(d[1:] @ np.linalg.pinv(S[1:, 1:]) @ d[1:])
            k = p + 1
            chow = ((sc - sa - sb) / k) / ((sa + sb) / max(len(yc) - 2 * k, 1) + EPS)
            f[f"arw{p}_wald_all"] = np.log1p(max(w_all, 0) / k); f[f"arw{p}_wald_dep"] = np.log1p(max(w_dep, 0) / p)
            f[f"arw{p}_chow"] = np.log1p(max(chow, 0)); f[f"arw{p}_svr"] = 0.5 * lratio(vb, va)
            # sum of AR coefs (persistence) change
            f[f"arw{p}_persist_d"] = bb[1:].sum() - ba[1:].sum()
        except Exception:
            pass
    # Whittle / log-spectral on smoothed periodograms (common 512 grid)
    def spec(s):
        s = s - s.mean()
        P = np.abs(np.fft.rfft(s, 1024)) ** 2 / len(s)
        return P[1:]
    Pa, Pb = spec(c.pre), spec(c.post)
    for w in (9, 33):
        Sa, Sb = uniform_filter1d(Pa, w) + 1e-8, uniform_filter1d(Pb, w) + 1e-8
        L = np.log(Sb / Sa)
        f[f"spec{w}_lsd"] = float(np.sqrt((L ** 2).mean()))
        f[f"spec{w}_is"] = float((Sb / Sa - L - 1).mean())        # Itakura-Saito
        f[f"spec{w}_whittle"] = float((Pb / Sa - np.log(Pb / Sa + 1e-12) - 1).mean())
        f[f"spec{w}_lmean"] = float(L.mean())
        lo = L[: len(L) // 8].mean(); hi = L[len(L) // 2:].mean()
        f[f"spec{w}_tilt"] = float(lo - hi)
    return f


# ---------------------------------------------------------------- E: near-unit-root / first-difference block
def fam_nur(c):
    f = {}
    z, n = c.z, c.npre
    dz = np.diff(z); s = max(1.4826 * np.median(np.abs(dz[:n - 1] - np.median(dz[:n - 1]))), 0.25 * dz[:n - 1].std(), 1e-9)
    u = dz / s
    a, b = u[:n - 1], u[n - 1:]
    f["nur_jump"] = u[n - 1]                                       # boundary jump in dz units
    f["nur_jump_abs"] = abs(u[n - 1])
    for w in (3, 10):
        ww = min(w, len(b))
        f[f"nur_jump{w}"] = z[n - 1 + ww] - z[n - 1] if n - 1 + ww < len(z) else 0.0
        f[f"nur_jump{w}_dz"] = f[f"nur_jump{w}"] / (s * np.sqrt(ww))
    f["nur_dz_mean_t"] = (b.mean() - a.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b) + EPS)
    f["nur_dz_lsr"] = lratio(b.std(), a.std())
    f["nur_dz_mad_lr"] = lratio(np.median(np.abs(b - np.median(b))), np.median(np.abs(a - np.median(a))))
    def vr(x, q):
        if len(x) <= 2 * q + 2:
            return 1.0
        d = np.diff(x); dq = x[q:] - x[:-q]
        return dq.var() / (q * d.var() + EPS)
    for q in (2, 5, 10, 20):
        va, vb = vr(z[:n], q), vr(z[n:], q)
        f[f"nur_vr{q}_pre"] = va; f[f"nur_vr{q}_lr"] = lratio(vb, va)
    # local-level innovation outlier at / after the boundary (random-walk-plus-noise filter)
    for alpha in (0.3, 0.7):
        lv = np.zeros(len(z)); lv[0] = z[0]
        for t in range(1, len(z)):
            lv[t] = alpha * z[t] + (1 - alpha) * lv[t - 1]
        e = z[1:] - lv[:-1]
        se = max(1.4826 * np.median(np.abs(e[:n - 1] - np.median(e[:n - 1]))), 1e-9)
        en = e / se
        f[f"nur_ll{int(alpha*10)}_bmax"] = np.abs(en[n - 1:n + 4]).max()
        f[f"nur_ll{int(alpha*10)}_bsum"] = en[n - 1:n + 9].sum() / 3.0
    # fraction of post inside pre band / range
    lo, hi = np.quantile(z[:n], [0.01, 0.99])
    post = z[n:]
    f["nur_in_band"] = ((post >= lo) & (post <= hi)).mean()
    f["nur_in_range"] = ((post >= z[:n].min()) & (post <= z[:n].max())).mean()
    f["nur_post_vs_last_pre"] = (np.median(post[:max(5, len(post) // 10)]) - np.median(z[n - max(5, n // 10):n])) / (s * np.sqrt(max(5, n // 10)))
    return f


FAMILIES.update({"battery": fam_battery, "battery_win": fam_battery_win, "distfam": fam_distfam, "arwald": fam_arwald, "nur": fam_nur})
