"""P2 new-mechanism families. Same contract as feats.py: single-series, pre-scaled, no clipping."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.special import ndtri
from feats import Ctx, FAMILIES, pit, acf, ar_design, ols, longest_run, EPS, lratio


def _ecdf_feats(f, pfx, ref, x):
    u = pit(np.sort(ref), x); pz = ndtri(u)
    f[f"{pfx}_mean"] = pz.mean(); f[f"{pfx}_absmean"] = np.abs(pz).mean()
    f[f"{pfx}_max"] = pz.max(); f[f"{pfx}_min"] = pz.min()
    f[f"{pfx}_hi"] = (u > 0.95).mean(); f[f"{pfx}_lo"] = (u < 0.05).mean()
    return u, pz


# (a) PIT extensions -------------------------------------------------------
def fam_pitx(c):
    f = {}
    pre, post = c.pre, c.post
    for b in (8, 24):
        if len(pre) < 3 * b or len(post) < 2 * b:
            b = max(4, min(len(pre), len(post)) // 6)
        wp = swv(pre, b)
        npb = len(post) // b
        wq = post[:npb * b].reshape(npb, b)
        _ecdf_feats(f, f"pitbm{b}", wp.mean(1), wq.mean(1))
        _ecdf_feats(f, f"pitbv{b}", np.log(wp.var(1) + 1e-6), np.log(wq.var(1) + 1e-6))
        _ecdf_feats(f, f"pitbr{b}", np.log(np.ptp(wp, 1) + 1e-6), np.log(np.ptp(wq, 1) + 1e-6))
    # direct h-step AR(2) forecast errors, fitted on pre
    z, npre = c.z, c.npre
    for h in (1, 2, 4, 8):
        n = len(z)
        idx = np.arange(1, n - h)
        X = np.column_stack([np.ones(len(idx)), z[idx], z[idx - 1]]); y = z[idx + h]
        pre_m = idx + h < npre
        beta, *_ = np.linalg.lstsq(X[pre_m], y[pre_m], rcond=None)
        e = y - X @ beta
        post_m = idx >= npre  # forecast origin in post
        if post_m.sum() < 5:
            post_m = idx + h >= npre
        u, pz = _ecdf_feats(f, f"pith{h}", e[pre_m], e[post_m])
        f[f"pith{h}_std"] = pz.std()
    # rank dependence / transition matrix of PIT (quintile states)
    u_pre = (np.argsort(np.argsort(pre)) + 0.5) / len(pre)
    u_post = pit(np.sort(pre), post)
    def trans(u):
        s = np.minimum((u * 5).astype(int), 4)
        M = np.zeros((5, 5))
        np.add.at(M, (s[:-1], s[1:]), 1)
        return M / M.sum(1, keepdims=True).clip(1), s
    Mp, sp = trans(u_pre); Mq, sq = trans(u_post)
    f["ptm_diag_pre"] = np.trace(Mp) / 5; f["ptm_diag_post"] = np.trace(Mq) / 5
    f["ptm_diag_diff"] = f["ptm_diag_post"] - f["ptm_diag_pre"]
    f["ptm_l1"] = np.abs(Mq - Mp).sum() / 5
    f["ptm_ext_stay"] = (Mq[0, 0] + Mq[4, 4]) - (Mp[0, 0] + Mp[4, 4])
    pzp, pzq = ndtri(np.clip(u_pre, 1e-4, 1 - 1e-4)), ndtri(u_post)
    for k in (1, 2, 5):
        f[f"pitacf{k}_pre"] = acf(pzp, k); f[f"pitacf{k}_post"] = acf(pzq, k)
        f[f"pitacf{k}_diff"] = f[f"pitacf{k}_post"] - f[f"pitacf{k}_pre"]
    f["pit_absacf1_diff"] = acf(np.abs(pzq), 1) - acf(np.abs(pzp), 1)
    return f


# (b) calibrated forecast residuals -----------------------------------------
def fam_fres(c):
    f = {}
    z, npre = c.z, c.npre
    ntr = int(0.7 * npre)
    for p in (1, 3, 6):
        X, y = ar_design(z, p)  # rows correspond to t = p..n-1
        t = np.arange(p, len(z))
        trm = t < ntr; cal = (t >= ntr) & (t < npre); pm = t >= npre
        beta, *_ = np.linalg.lstsq(X[trm], y[trm], rcond=None)
        e = y - X @ beta
        ec = e[cal]
        mu = np.median(ec)
        s = max(1.4826 * np.median(np.abs(ec - mu)), 0.25 * ec.std(), 1e-6)
        q = (e[pm] - mu) / s
        r = (ec - mu) / s
        f[f"fr{p}_mean"] = q.mean(); f[f"fr{p}_meanz"] = q.mean() * np.sqrt(len(q)) / (r.std() + 1e-6)
        f[f"fr{p}_rms_lr"] = lratio(np.sqrt((q ** 2).mean()), np.sqrt((r ** 2).mean()))
        f[f"fr{p}_mad_lr"] = lratio(np.median(np.abs(q)), np.median(np.abs(r)))
        f[f"fr{p}_tail3"] = (np.abs(q) > 3).mean() - (np.abs(r) > 3).mean()
        f[f"fr{p}_cusum"] = np.abs(np.cumsum(q - r.mean())).max() / (np.sqrt(len(q)) * (r.std() + 1e-6))
        f[f"fr{p}_acf1"] = acf(q, 1) - acf(r, 1)
        if p == 3:
            th = np.array_split(q, 3)
            for i, tt in enumerate(th):
                f[f"fr3_t{i}_mean"] = tt.mean(); f[f"fr3_t{i}_rms"] = np.sqrt((tt ** 2).mean())
    # nonlinear: threshold AR(1) (regimes split at pre median of lagged value)
    X, y = ar_design(z, 1); t = np.arange(1, len(z))
    trm = t < ntr; cal = (t >= ntr) & (t < npre); pm = t >= npre
    reg = X[:, 1] > np.median(z[:ntr])
    e = np.zeros_like(y)
    for g in (False, True):
        m = trm & (reg == g)
        if m.sum() < 5:
            m = trm
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        e[reg == g] = (y - X @ beta)[reg == g]
    ec = e[cal]; mu = np.median(ec)
    s = max(1.4826 * np.median(np.abs(ec - mu)), 0.25 * ec.std(), 1e-6)
    q = (e[pm] - mu) / s; r = (ec - mu) / s
    f["tar_mean"] = q.mean(); f["tar_rms_lr"] = lratio(np.sqrt((q ** 2).mean()), np.sqrt((r ** 2).mean()))
    return f


# (c) delay-embedding energy distance / MMD ---------------------------------
def _sub(a, k, rng):
    if len(a) <= k:
        return a
    return a[np.sort(rng.choice(len(a), k, replace=False))]


def _energy(A, B):
    def md(P, Q):
        return np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).mean()
    return 2 * md(A, B) - md(A, A) - md(B, B)


def _mmd(A, B, gam):
    def k(P, Q):
        return np.exp(-gam * ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).mean()
    return k(A, A) + k(B, B) - 2 * k(A, B)


def fam_embed(c):
    f = {}
    pre, post = c.pre, c.post
    rng = np.random.default_rng(len(pre) * 1009 + len(post))  # deterministic, depends only on this series' lengths
    h = len(pre) // 2
    for m in (2, 3, 4):
        Ep, Eq = swv(pre, m), swv(post, m)
        A = _sub(Ep, 160, rng); B = _sub(Eq, 160, rng)
        A1 = _sub(swv(pre[:h], m), 120, rng); A2 = _sub(swv(pre[h:], m), 120, rng)
        ed = _energy(A, B); ed0 = _energy(A1, A2)
        f[f"emb{m}_energy"] = ed; f[f"emb{m}_energy_cal"] = np.log((ed + 1e-4) / (abs(ed0) + 1e-2))
        d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        gam = 1.0 / (np.median(d2[d2 > 0]) + 1e-6)
        mm = _mmd(A, B, gam); mm0 = _mmd(A1, A2, gam)
        f[f"emb{m}_mmd"] = mm; f[f"emb{m}_mmd_cal"] = np.log((mm + 1e-4) / (abs(mm0) + 1e-3))
        # embedding of first differences (shape of local dynamics, level-free)
        if m == 3:
            Dp, Dq = swv(np.diff(pre), m), swv(np.diff(post), m)
            A = _sub(Dp, 160, rng); B = _sub(Dq, 160, rng)
            f["embd3_energy"] = _energy(A, B)
    return f


# (d) ordinal patterns + quantile Markov transitions ------------------------
def _ord_dist(x, m):
    w = swv(x, m)
    pat = np.argsort(w, axis=1, kind="stable")
    code = (pat * (m ** np.arange(m))).sum(1)
    return code


def _js(p, q):
    mm = 0.5 * (p + q)
    def kl(a, b):
        mk = a > 0
        return (a[mk] * np.log(a[mk] / b[mk])).sum()
    return 0.5 * kl(p, mm) + 0.5 * kl(q, mm)


def fam_ordinal(c):
    f = {}
    pre, post = c.pre, c.post
    h = len(pre) // 2
    for m in (3, 4, 5):
        cp, cq = _ord_dist(pre, m), _ord_dist(post, m)
        c1, c2 = _ord_dist(pre[:h], m), _ord_dist(pre[h:], m)
        allc = np.unique(np.concatenate([cp, cq]))
        def hist(cc):
            return np.searchsorted(allc, cc)
        K = len(allc)
        P = np.bincount(hist(cp), minlength=K) + 0.5; Q = np.bincount(hist(cq), minlength=K) + 0.5
        P1 = np.bincount(np.searchsorted(allc, c1[np.isin(c1, allc)]), minlength=K) + 0.5
        P2 = np.bincount(np.searchsorted(allc, c2[np.isin(c2, allc)]), minlength=K) + 0.5
        js = _js(P / P.sum(), Q / Q.sum()); js0 = _js(P1 / P1.sum(), P2 / P2.sum())
        f[f"ord{m}_js"] = js; f[f"ord{m}_js_cal"] = np.log((js + 1e-4) / (js0 + 1e-4))
        # chi2 homogeneity
        Tt = P + Q; E1 = Tt * P.sum() / Tt.sum(); E2 = Tt * Q.sum() / Tt.sum()
        f[f"ord{m}_chi2n"] = (((P - E1) ** 2 / E1).sum() + ((Q - E2) ** 2 / E2).sum()) / K
        if m == 3:
            # monotone patterns share (persistence) pre vs post
            mono = lambda cc: np.isin(cc, [0 * 1 + 1 * 3 + 2 * 9, 2 * 1 + 1 * 3 + 0 * 9]).mean()
            f["ord3_mono_diff"] = mono(cq) - mono(cp)
    # quantile-binned Markov transitions (pre quintile bins)
    edges = np.quantile(pre, [0.2, 0.4, 0.6, 0.8])
    def tm(x):
        s = np.searchsorted(edges, x)
        M = np.zeros((5, 5)); np.add.at(M, (s[:-1], s[1:]), 1)
        return M
    Mp, Mq = tm(pre), tm(post)
    Mp1, Mp2 = tm(pre[:h]), tm(pre[h:])
    def rowjs(A, B):
        tot = 0.0
        for i in range(5):
            a = A[i] + 0.25; b = B[i] + 0.25
            tot += _js(a / a.sum(), b / b.sum()) * (A[i].sum() + B[i].sum())
        return tot / (A.sum() + B.sum())
    v = rowjs(Mp, Mq); v0 = rowjs(Mp1, Mp2)
    f["mtm_js"] = v; f["mtm_js_cal"] = np.log((v + 1e-4) / (v0 + 1e-4))
    Pn = Mp / Mp.sum(); Qn = Mq / Mq.sum()
    f["mtm_joint_js"] = _js((Pn + 1e-4).ravel() / (Pn + 1e-4).sum(), (Qn + 1e-4).ravel() / (Qn + 1e-4).sum())
    f["mtm_occ_l1"] = np.abs(Mq.sum(1) / Mq.sum() - Mp.sum(1) / Mp.sum()).sum()
    return f


FAMILIES.update({"pitx": fam_pitx, "fres": fam_fres, "embed": fam_embed, "ordinal": fam_ordinal})


# (e) boundary regression discontinuity, calibrated with placebo boundaries -----
def _rdd(z, b, w):
    a0 = max(0, b - w); a1 = min(len(z), b + w)
    L_ = z[a0:b]; R_ = z[b:a1]
    tl = np.arange(-len(L_), 0, dtype=float); tr = np.arange(len(R_), dtype=float)
    def fitl(t, y):
        X = np.column_stack([np.ones(len(t)), t])
        beta, r, ssr = ols(X, y)
        return beta, ssr, len(y)
    bl, sl, nl = fitl(tl, L_); br, sr, nr = fitl(tr, R_)
    s2 = (sl + sr) / max(nl + nr - 4, 1) + EPS
    se = np.sqrt(s2 * (4.0 / nl + 4.0 / nr))
    jump = br[0] - bl[0]
    return jump / se, br[1] - bl[1], np.log((sr / nr + EPS) / (sl / nl + EPS))


def fam_rdd(c):
    f = {}
    z, npre = c.z, c.npre
    npost = len(z) - npre
    for w in (20, 50, 120):
        ww = min(w, npre // 2, npost // 2)
        t0, ds, dv = _rdd(z, npre, ww)
        tp, dsp, dvp = _rdd(z, npre // 2, min(ww, npre // 4))
        tq, dsq, dvq = _rdd(z, npre + npost // 2, min(ww, npost // 4))
        f[f"rdd{w}_t"] = t0; f[f"rdd{w}_abs_t"] = abs(t0)
        f[f"rdd{w}_slope"] = ds; f[f"rdd{w}_var_lr"] = dv
        f[f"rdd{w}_t_cal"] = abs(t0) / (1.0 + 0.5 * (abs(tp) + abs(tq)))
        f[f"rdd{w}_var_cal"] = abs(dv) - 0.5 * (abs(dvp) + abs(dvq))
    # pre-trend extrapolation: post mean vs linear extrapolation of pre trend (drift-aware shift)
    tt = np.arange(len(z), dtype=float)
    X = np.column_stack([np.ones(npre), tt[:npre]])
    beta, r, ssr = ols(X, z[:npre])
    pred = beta[0] + beta[1] * tt[npre:]
    res = z[npre:] - pred
    s = np.sqrt(ssr / max(npre - 2, 1)) + EPS
    f["extrap_mean"] = res.mean() / s
    f["extrap_early"] = res[:max(10, npost // 5)].mean() / s
    f["detr_shift_vs_trend"] = (z[npre:].mean() - z[:npre].mean()) / (abs(beta[1]) * len(z) / 2 + s)
    return f


FAMILIES["rdd"] = fam_rdd
