"""Per-series feature families for KU-SBB. Every feature is computed from a single series only,
after PRE-boundary robust scaling z = (x - median_pre) / max(1.4826*MAD_pre, 0.25*std_pre, tiny).
No clipping / outlier removal anywhere. All features are invariant to x -> a*x + b (a > 0)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.ndimage import median_filter
from scipy.signal import lfilter
from scipy.special import ndtri, ndtr
from scipy.stats import rankdata

EPS = 1e-9
NFFT = 4096


# ----------------------------------------------------------------- helpers
def robust_scale(x, npre):
    pre = x[:npre]
    med = np.median(pre)
    mad = 1.4826 * np.median(np.abs(pre - med))
    sd = pre.std()
    s = max(mad, 0.25 * sd)
    if not np.isfinite(s) or s <= 1e-12 * (abs(med) + 1.0):
        s = 1.0  # constant pre segment: degenerate but finite
    return (x - med) / s


def acf(x, k):
    x = x - x.mean()
    d = np.dot(x, x)
    if d <= EPS or len(x) <= k + 1:
        return 0.0
    return float(np.dot(x[:-k], x[k:]) / d)


def longest_run(b):
    if not b.any():
        return 0
    d = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1)
    return int((e - s).max())


def skew(x):
    m = x.mean(); s = x.std()
    return float(((x - m) ** 3).mean() / (s ** 3 + EPS))


def kurt(x):
    m = x.mean(); s = x.std()
    return float(((x - m) ** 4).mean() / (s ** 4 + EPS) - 3.0)


def lratio(a, b):
    return float(np.log((a + 1e-6) / (b + 1e-6)))


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, r, float(r @ r)


def ar_design(y, p):
    n = len(y)
    X = np.ones((n - p, p + 1))
    for j in range(1, p + 1):
        X[:, j] = y[p - j:n - j]
    return X, y[p:]


def ar_resid_full(z, npre, p):
    """Fit AR(p) on pre, return residuals for full series (index aligned from p) and coefs."""
    X, y = ar_design(z[:npre], p)
    beta, r, _ = ols(X, y)
    Xf, yf = ar_design(z, p)
    e = yf - Xf @ beta
    return e, beta  # e[t-p] corresponds to time t


def lrv_bartlett(u):
    n = len(u)
    u = u - u.mean()
    L = int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    g0 = u @ u / n
    v = g0
    for l in range(1, L + 1):
        v += 2 * (1 - l / (L + 1)) * (u[:-l] @ u[l:]) / n
    return max(v, 0.05 * g0, EPS)


def huber_loc(x, c=1.345, it=12):
    m = np.median(x)
    s = 1.4826 * np.median(np.abs(x - m)) + EPS
    for _ in range(it):
        r = (x - m) / s
        w = np.minimum(1.0, c / (np.abs(r) + EPS))
        m = (w @ x) / w.sum()
    return float(m), float(s)


def t_loc(x, nu=3.0, it=15):
    m = np.median(x)
    s = 1.4826 * np.median(np.abs(x - m)) + EPS
    for _ in range(it):
        r = (x - m) / s
        w = (nu + 1) / (nu + r * r)
        m = (w @ x) / w.sum()
        s = np.sqrt((w * (x - m) ** 2).mean()) + EPS
    return float(m), float(s)


def ks_stat(a, b):
    a = np.sort(a); b = np.sort(b)
    allv = np.concatenate([a, b])
    fa = np.searchsorted(a, allv, side="right") / len(a)
    fb = np.searchsorted(b, allv, side="right") / len(b)
    d = fa - fb
    return float(np.abs(d).max()), float(d.max()), float(-d.min())


def ad_cvm_mw(a, b):
    n, m = len(a), len(b)
    N = n + m
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    isa = (order < n).astype(float)
    Ma = np.cumsum(isa)[:-1]
    j = np.arange(1, N)
    Mb = j - Ma
    ad = ((N * Ma - j * n) ** 2 / n + (N * Mb - j * m) ** 2 / m) / (j * (N - j))
    ad = float(ad.sum() / N)
    r = rankdata(allv)
    ra = np.sort(r[:n]); rb = np.sort(r[n:])
    U = n * ((ra - np.arange(1, n + 1)) ** 2).sum() + m * ((rb - np.arange(1, m + 1)) ** 2).sum()
    cvm = U / (n * m * N) - (4 * m * n - 1) / (6 * N)
    mw = (r[n:].sum() - m * (m + 1) / 2) / (n * m)  # P(post > pre)
    return ad, float(cvm), float(mw)


def lmom(x):
    x = np.sort(x); n = len(x)
    i = np.arange(n)
    b0 = x.mean()
    b1 = (i * x).sum() / (n * (n - 1))
    b2 = (i * (i - 1) * x).sum() / (n * (n - 1) * (n - 2))
    b3 = (i * (i - 1) * (i - 2) * x).sum() / (n * (n - 1) * (n - 2) * (n - 3))
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l2, l3 / (l2 + EPS), l4 / (l2 + EPS)


def hill(x, frac=0.1):
    a = np.sort(np.abs(x))[::-1]
    k = max(5, int(len(a) * frac))
    if a[k] <= EPS:
        return 0.0
    return float(np.mean(np.log(a[:k] / a[k] + EPS)))


def _fact(m):
    f = 1
    for i in range(2, m + 1):
        f *= i
    return f


def perm_entropy2(x, m):
    w = swv(x, m)
    pat = np.argsort(w, axis=1, kind="stable")
    code = (pat * (m ** np.arange(m))).sum(1)
    _, c = np.unique(code, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(_fact(m)))


def rolling_std(x, w):
    c1 = np.cumsum(np.concatenate(([0.0], x)))
    c2 = np.cumsum(np.concatenate(([0.0], x * x)))
    s1 = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    v = s2 / w - (s1 / w) ** 2
    return np.sqrt(np.maximum(v, 0))


def arch_lm(e, q=5):
    e2 = e * e
    if len(e2) <= q + 10:
        return 0.0
    X, y = ar_design(e2, q)
    _, r, ssr = ols(X, y)
    sst = ((y - y.mean()) ** 2).sum() + EPS
    return float(len(y) * (1 - ssr / sst))


def pit(pre_sorted, x):
    n = len(pre_sorted)
    u = (np.searchsorted(pre_sorted, x, "left") + np.searchsorted(pre_sorted, x, "right")) / (2.0 * n)
    return np.clip(u, 0.5 / n, 1 - 0.5 / n)


# ----------------------------------------------------------------- context
class Ctx:
    def __init__(self, x, npre):
        self.x = x
        self.npre = npre
        self.z = robust_scale(np.asarray(x, float), npre)
        self.pre = self.z[:npre]
        self.post = self.z[npre:]
        self._c = {}

    def ar(self, p):
        k = ("ar", p)
        if k not in self._c:
            e, beta = ar_resid_full(self.z, self.npre, p)
            nb = self.npre - p
            ep, eq = e[:nb], e[nb:]
            s = max(1.4826 * np.median(np.abs(ep - np.median(ep))), 0.25 * ep.std(), 1e-6)
            self._c[k] = (e, beta, ep / s, eq / s)
        return self._c[k]


# ----------------------------------------------------------------- families
def fam_wide(c):
    f = {}
    pre, post, z, npre = c.pre, c.post, c.z, c.npre
    npost = len(post); n = len(z)
    f["n_pre"] = npre; f["n_post"] = npost; f["log_len_ratio"] = np.log(npost / npre)
    sp = pre.std(); sq = post.std()
    mp, mq = pre.mean(), post.mean()
    pooled = np.sqrt((sp ** 2 + sq ** 2) / 2) + EPS
    f["mean_shift"] = mq - mp; f["abs_mean_shift"] = abs(mq - mp)
    f["mean_eff"] = (mq - mp) / pooled; f["abs_mean_eff"] = abs(f["mean_eff"])
    f["welch_t"] = (mq - mp) / np.sqrt(sp ** 2 / npre + sq ** 2 / npost + EPS)
    f["med_shift"] = np.median(post) - np.median(pre); f["abs_med_shift"] = abs(f["med_shift"])
    for tr in (0.1, 0.25):
        def tm(a):
            a = np.sort(a); k = int(len(a) * tr); return a[k:len(a) - k].mean()
        d = tm(post) - tm(pre); f[f"trim{int(tr*100)}_shift"] = d; f[f"abs_trim{int(tr*100)}_shift"] = abs(d)
    def wm(a, q=0.1):
        lo, hi = np.quantile(a, [q, 1 - q]); return np.minimum(np.maximum(a, lo), hi).mean()  # winsorised *statistic*, data untouched
    f["wins_shift"] = wm(post) - wm(pre)
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    qp = np.quantile(pre, qs); qq = np.quantile(post, qs)
    for q, a, b in zip(qs, qp, qq):
        f[f"q{int(q*100)}_shift"] = b - a
    f["qshift_absmean"] = np.abs(qq - qp).mean()
    iqp = qp[5] - qp[3]; iqq = qq[5] - qq[3]
    madp = np.median(np.abs(pre - np.median(pre))); madq = np.median(np.abs(post - np.median(post)))
    f["std_lr"] = lratio(sq, sp); f["abs_std_lr"] = abs(f["std_lr"])
    f["mad_lr"] = lratio(madq, madp); f["abs_mad_lr"] = abs(f["mad_lr"])
    f["iqr_lr"] = lratio(iqq, iqp); f["abs_iqr_lr"] = abs(f["iqr_lr"])
    f["range9010_lr"] = lratio(qq[6] - qq[2], qp[6] - qp[2])
    f["range9901_lr"] = lratio(qq[8] - qq[0], qp[8] - qp[0])
    f["std_pre"] = sp; f["std_post"] = sq
    sk_p, sk_q, ku_p, ku_q = skew(pre), skew(post), kurt(pre), kurt(post)
    f["skew_pre"], f["skew_post"], f["skew_diff"] = sk_p, sk_q, sk_q - sk_p
    f["kurt_pre"], f["kurt_post"], f["kurt_diff"] = ku_p, ku_q, ku_q - ku_p
    # Wasserstein via quantile grid, KS
    g = np.linspace(0.005, 0.995, 100)
    f["wass"] = np.abs(np.quantile(post, g) - np.quantile(pre, g)).mean()
    ks, ksp, ksn = ks_stat(pre, post)
    f["ks"], f["ks_up"], f["ks_dn"] = ks, ksp, ksn
    f["ks_scaled"] = ks * np.sqrt(npre * npost / (npre + npost))
    # thirds
    for nm, seg in (("post", post), ("pre", pre)):
        th = np.array_split(seg, 3)
        for i, t in enumerate(th):
            f[f"{nm}_t{i}_med"] = np.median(t) - (np.median(pre) if nm == "post" else np.median(pre))
            f[f"{nm}_t{i}_std_lr"] = lratio(t.std(), sp)
        if nm == "post":
            meds = np.array([np.median(t) for t in th])
            f["post_thirds_signmin"] = np.sign(meds[0]) * np.abs(meds).min() if np.all(np.sign(meds) == np.sign(meds[0])) else 0.0
            f["post_thirds_range"] = meds.max() - meds.min()
            f["post_late_minus_early"] = meds[2] - meds[0]
    # bands
    for thr in (2.0, 3.0, 5.0):
        f[f"frac_abs_gt{int(thr)}_post"] = (np.abs(post) > thr).mean()
        f[f"frac_abs_gt{int(thr)}_pre"] = (np.abs(pre) > thr).mean()
        f[f"frac_abs_gt{int(thr)}_diff"] = f[f"frac_abs_gt{int(thr)}_post"] - f[f"frac_abs_gt{int(thr)}_pre"]
    lo5, hi95, lo1, hi99 = qp[1], qp[7], qp[0], qp[8]
    up, dn = post > hi95, post < lo5
    f["post_frac_above95"], f["post_frac_below05"] = up.mean(), dn.mean()
    f["post_frac_out95"] = up.mean() + dn.mean()
    f["post_frac_out99"] = (post > hi99).mean() + (post < lo1).mean()
    f["run_above95"] = longest_run(up) / npost; f["run_below05"] = longest_run(dn) / npost
    f["run_above2"] = longest_run(post > 2) ; f["run_below2"] = longest_run(post < -2)
    f["run_above2_pre"] = longest_run(pre > 2); f["run_below2_pre"] = longest_run(pre < -2)
    f["run_pos_post"] = longest_run(post > np.median(pre)) / npost
    f["run_neg_post"] = longest_run(post < np.median(pre)) / npost
    f["frac_above_med"] = (post > 0).mean()
    # outliers
    f["maxabs_pre"] = np.abs(pre).max(); f["maxabs_post"] = np.abs(post).max()
    f["maxabs_lr"] = lratio(f["maxabs_post"], f["maxabs_pre"])
    f["n_out5_post"] = (np.abs(post) > 5).sum(); f["n_out5_pre"] = (np.abs(pre) > 5).sum()
    # Hampel / median filtered views
    for w in (5, 11):
        mf = median_filter(z, size=w, mode="nearest")
        a, b = mf[:npre], mf[npre:]
        f[f"mf{w}_mean_shift"] = b.mean() - a.mean()
        f[f"mf{w}_std_lr"] = lratio(b.std(), a.std())
        f[f"mf{w}_t"] = (b.mean() - a.mean()) / np.sqrt(lrv_bartlett(a) / npre + lrv_bartlett(b) / npost)
        res = z - mf
        f[f"mf{w}_res_std_lr"] = lratio(res[npre:].std(), res[:npre].std())
    # piecewise vs common trend
    t = np.arange(n) / n
    D = np.zeros(n); D[npre:] = 1
    tt = (np.arange(n) - npre) / n * D
    X0 = np.column_stack([np.ones(n), t]); X1 = np.column_stack([np.ones(n), t, D]); X2 = np.column_stack([np.ones(n), t, D, tt])
    _, _, s0 = ols(X0, z); b1, _, s1 = ols(X1, z); b2, _, s2 = ols(X2, z)
    f["trend_F_level"] = (s0 - s1) / (s1 / (n - 3) + EPS)
    f["trend_F_full"] = ((s0 - s2) / 2) / (s2 / (n - 4) + EPS)
    f["trend_bic_gain"] = n * np.log((s0 + EPS) / (s2 + EPS)) - 2 * np.log(n)
    f["trend_level_coef"] = b1[2]; f["trend_slope_change"] = b2[3]
    _, _, sp_ = ols(X0[:npre], z[:npre]); _, _, sq_ = ols(X0[npre:], z[npre:])
    f["detr_resid_std_lr"] = 0.5 * lratio(sq_ / npost, sp_ / npre)
    # HAC mean-shift t
    lp, lq = lrv_bartlett(pre), lrv_bartlett(post)
    f["hac_t"] = (mq - mp) / np.sqrt(lp / npre + lq / npost)
    f["abs_hac_t"] = abs(f["hac_t"])
    f["lrv_lr"] = lratio(lq, lp)
    # AR prewhitened
    for p in (1, 2):
        e, beta, ep, eq = c.ar(p)
        f[f"ar{p}_e_mean_t"] = (eq.mean() - ep.mean()) / np.sqrt(ep.var() / len(ep) + eq.var() / len(eq) + EPS)
        f[f"ar{p}_e_std_lr"] = lratio(eq.std(), ep.std())
        f[f"ar{p}_e_mad_lr"] = lratio(np.median(np.abs(eq - np.median(eq))), np.median(np.abs(ep - np.median(ep))))
        f[f"ar{p}_e_ks"] = ks_stat(ep, eq)[0]
        f[f"ar{p}_e_kurt_diff"] = kurt(eq) - kurt(ep)
        f[f"ar{p}_e_cusum"] = np.abs(np.cumsum(eq - ep.mean())).max() / (ep.std() * np.sqrt(len(eq)) + EPS)
        f[f"ar{p}_phi1"] = beta[1]
        # Chow test
        Xa, ya = ar_design(z[:npre], p); Xb, yb = ar_design(z[npre:], p); Xc, yc = ar_design(z, p)
        _, _, ssa = ols(Xa, ya); bb, _, ssb = ols(Xb, yb); _, _, ssc = ols(Xc, yc)
        k = p + 1
        f[f"chow{p}_logF"] = np.log1p(max(((ssc - ssa - ssb) / k) / ((ssa + ssb) / (len(yc) - 2 * k) + EPS), 0))
        f[f"ar{p}_phi1_post"] = bb[1]; f[f"ar{p}_phi1_diff"] = bb[1] - beta[1]
        if p == 1:
            f["ar1_mu_pre"] = beta[0] / (1 - beta[1]) if abs(1 - beta[1]) > 0.02 else 0.0
            f["ar1_mu_post"] = bb[0] / (1 - bb[1]) if abs(1 - bb[1]) > 0.02 else 0.0
    # GLS trend shift (AR(1) quasi-differencing with pre phi)
    phi = float(np.clip(c.ar(1)[1][1], -0.98, 0.98))
    Xs = np.column_stack([np.ones(n), t, D, tt])
    ys = z[1:] - phi * z[:-1]; Xq = Xs[1:] - phi * Xs[:-1]
    bg, rg, sg = ols(Xq, ys)
    try:
        cov = np.linalg.inv(Xq.T @ Xq) * sg / (len(ys) - 4)
        f["gls_level_t"] = bg[2] / np.sqrt(cov[2, 2] + EPS); f["gls_slope_t"] = bg[3] / np.sqrt(cov[3, 3] + EPS)
    except np.linalg.LinAlgError:
        f["gls_level_t"] = 0.0; f["gls_slope_t"] = 0.0
    f["abs_gls_level_t"] = abs(f["gls_level_t"])
    # robust contrasts
    hp, hs = huber_loc(pre); hq, hs2 = huber_loc(post)
    f["huber_shift"] = hq - hp; f["huber_t"] = (hq - hp) / np.sqrt(hs ** 2 / npre + hs2 ** 2 / npost + EPS)
    tp, ts = t_loc(pre); tq, ts2 = t_loc(post)
    f["tloc_shift"] = tq - tp; f["tloc_t"] = (tq - tp) / np.sqrt(ts ** 2 / npre + ts2 ** 2 / npost + EPS)
    f["tscale_lr"] = lratio(ts2, ts)
    # local boundary windows
    for w in (10, 25, 50):
        a = pre[-w:]; b = post[:w]
        f[f"win{w}_shift"] = b.mean() - a.mean()
        f[f"win{w}_t"] = f[f"win{w}_shift"] / np.sqrt(a.var() / w + b.var() / w + EPS)
        f[f"win{w}_std_lr"] = lratio(b.std(), a.std())
        f[f"win{w}_vs_pre_shift"] = b.mean() - mp
    # ACF changes
    for k in (1, 2, 3, 5, 10, 20):
        a1, a2 = acf(pre, k), acf(post, k)
        f[f"acf{k}_pre"] = a1; f[f"acf{k}_post"] = a2; f[f"acf{k}_diff"] = a2 - a1
    ap, aq = np.abs(pre - np.median(pre)), np.abs(post - np.median(post))
    f["absacf1_pre"], f["absacf1_post"] = acf(ap, 1), acf(aq, 1)
    f["absacf1_diff"] = f["absacf1_post"] - f["absacf1_pre"]
    # differences
    d1p, d1q = np.diff(pre), np.diff(post)
    d2p, d2q = np.diff(d1p), np.diff(d1q)
    f["d1_std_lr"] = lratio(d1q.std(), d1p.std()); f["d2_std_lr"] = lratio(d2q.std(), d2p.std())
    f["d1_mad_lr"] = lratio(np.median(np.abs(d1q)), np.median(np.abs(d1p)))
    f["d1_mean_shift"] = d1q.mean() - d1p.mean()
    f["d1_acf1_pre"], f["d1_acf1_post"] = acf(d1p, 1), acf(d1q, 1)
    f["d1_acf1_diff"] = f["d1_acf1_post"] - f["d1_acf1_pre"]
    ep_ = d1p.var() / (pre.var() + EPS); eq_ = d1q.var() / (post.var() + EPS)
    f["denergy_pre"], f["denergy_post"], f["denergy_lr"] = ep_, eq_, lratio(eq_, ep_)
    f["d1_kurt_diff"] = kurt(d1q) - kurt(d1p)
    f["d1_maxabs_lr"] = lratio(np.abs(d1q).max(), np.abs(d1p).max())
    f["jump_at_boundary"] = (z[npre] - z[npre - 1]) / (np.median(np.abs(d1p)) * 1.4826 + 1e-6)
    return f


def _band_powers(seg):
    s = seg - seg.mean()
    P = np.abs(np.fft.rfft(s, NFFT)) ** 2 / len(s)
    return P


BANDS = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.35, 0.5001])


def fam_shape(c):
    f = {}
    pre, post, z, npre = c.pre, c.post, c.z, c.npre
    npost = len(post); n = len(z)
    # QQ regression
    g = np.linspace(0.02, 0.98, 49)
    qp, qq = np.quantile(pre, g), np.quantile(post, g)
    A = np.column_stack([np.ones_like(qp), qp])
    b, r, ssr = ols(A, qq)
    f["qq_int"], f["qq_slope"] = b
    f["qq_r2"] = 1 - ssr / (((qq - qq.mean()) ** 2).sum() + EPS)
    f["qq_maxdev"] = np.abs(r).max()
    f["qq_tail_up"] = (qq[-3:] - qp[-3:]).mean() - (qq[23] - qp[23])
    f["qq_tail_dn"] = (qq[:3] - qp[:3]).mean() - (qq[23] - qp[23])
    # Hill, L-moments
    hp, hq = hill(pre - np.median(pre)), hill(post - np.median(post))
    f["hill_pre"], f["hill_post"], f["hill_diff"] = hp, hq, hq - hp
    l2p, t3p, t4p = lmom(pre); l2q, t3q, t4q = lmom(post)
    f["l2_lr"] = lratio(l2q, l2p); f["t3_pre"], f["t3_post"], f["t3_diff"] = t3p, t3q, t3q - t3p
    f["t4_pre"], f["t4_post"], f["t4_diff"] = t4p, t4q, t4q - t4p
    ad, cvm, mw = ad_cvm_mw(pre, post)
    f["ad"], f["cvm"], f["mw"], f["abs_mw"] = ad, cvm, mw, abs(mw - 0.5)
    ad2, cvm2, mw2 = ad_cvm_mw(np.abs(pre), np.abs(post))
    f["ad_abs"], f["mw_abs"] = ad2, mw2
    # periodograms on a common 4096 grid
    Pp, Pq = _band_powers(pre), _band_powers(post)
    fr = np.fft.rfftfreq(NFFT)
    idx = np.digitize(fr, BANDS) - 1
    bp = np.bincount(idx, Pp, minlength=len(BANDS) - 1)[:len(BANDS) - 1] / np.maximum(np.bincount(idx, minlength=len(BANDS) - 1)[:len(BANDS) - 1], 1)
    bq = np.bincount(idx, Pq, minlength=len(BANDS) - 1)[:len(BANDS) - 1] / np.maximum(np.bincount(idx, minlength=len(BANDS) - 1)[:len(BANDS) - 1], 1)
    sharep, shareq = bp / (bp.sum() + EPS), bq / (bq.sum() + EPS)
    for i in range(len(bp)):
        f[f"band{i}_lr"] = lratio(bq[i], bp[i])
        f[f"band{i}_share_pre"] = sharep[i]
        f[f"band{i}_share_diff"] = shareq[i] - sharep[i]
    lpp, lpq = np.log(Pp[1:] + 1e-6), np.log(Pq[1:] + 1e-6)
    k = 32
    sm = lambda v: np.convolve(v, np.ones(k) / k, mode="valid")
    f["spec_l1"] = np.abs(sm(lpq) - sm(lpp)).mean()
    f["spec_flat_pre"] = np.exp(lpp.mean()) / (Pp[1:].mean() + EPS)
    f["spec_flat_post"] = np.exp(lpq.mean()) / (Pq[1:].mean() + EPS)
    lo = int(NFFT * 2 / npre) + 1
    ipk = lo + int(np.argmax(Pp[lo:]))
    fpk = fr[ipk]
    f["dom_freq_pre"] = fpk
    f["dom_pow_share_pre"] = Pp[ipk] / (Pp[1:].sum() + EPS) * 50
    f["dom_pow_post_at_pre"] = lratio(Pq[max(ipk - 2, 0):ipk + 3].max(), Pp[ipk])
    iq = lo + int(np.argmax(Pq[lo:]))
    f["dom_freq_post"] = fr[iq]; f["dom_freq_diff"] = abs(fr[iq] - fpk)
    f["dom_pow_share_post"] = Pq[iq] / (Pq[1:].sum() + EPS) * 50
    # cycle fit at pre-dominant frequency: common vs separate cycle, controlling seg means/trends
    tt = np.arange(n)
    D = np.zeros(n); D[npre:] = 1
    tn = tt / n
    cs, sn = np.cos(2 * np.pi * fpk * tt), np.sin(2 * np.pi * fpk * tt)
    Xb = np.column_stack([np.ones(n), D, tn, D * tn])
    X0 = np.column_stack([Xb, cs, sn]); X1 = np.column_stack([X0, D * cs, D * sn])
    _, _, s_base = ols(Xb, z)
    b0, r0, s0 = ols(X0, z); b1, r1, s1 = ols(X1, z)
    f["cyc_F_sep"] = np.log1p(max(((s0 - s1) / 2) / (s1 / (n - 8) + EPS), 0))
    f["cyc_R2_gain"] = (s_base - s0) / (s_base + EPS)
    ampp = np.hypot(b1[4], b1[5]); ampq = np.hypot(b1[4] + b1[6], b1[5] + b1[7])
    f["cyc_amp_pre"] = ampp; f["cyc_amp_lr"] = lratio(ampq, ampp)
    ph = np.arctan2(b1[5], b1[4]) - np.arctan2(b1[5] + b1[7], b1[4] + b1[6])
    f["cyc_phase_diff"] = abs(np.angle(np.exp(1j * ph)))
    f["cycres_std_lr"] = lratio(r0[npre:].std(), r0[:npre].std())
    f["cycres_mean_shift"] = r0[npre:].mean() - r0[:npre].mean()
    f["cycres_acf1_pre"], f["cycres_acf1_post"] = acf(r0[:npre], 1), acf(r0[npre:], 1)
    f["cycres_acf1_diff"] = f["cycres_acf1_post"] - f["cycres_acf1_pre"]
    # ARCH-LM on AR(2) innovations
    e, _, ep, eq = c.ar(2)
    a1, a2 = arch_lm(ep), arch_lm(eq)
    f["archlm_pre"], f["archlm_post"] = a1 / len(ep), a2 / len(eq)
    f["archlm_diff"] = f["archlm_post"] - f["archlm_pre"]
    # permutation entropy
    for m in (3, 4):
        a, b = perm_entropy2(pre, m), perm_entropy2(post, m)
        f[f"pe{m}_pre"], f[f"pe{m}_post"], f[f"pe{m}_diff"] = a, b, b - a
    return f


def _pit_feats(f, pfx, ref, x):
    srt = np.sort(ref)
    u = pit(srt, x)
    pz = ndtri(u)
    n = len(x)
    f[f"{pfx}_mean"] = pz.mean(); f[f"{pfx}_std"] = pz.std()
    f[f"{pfx}_meanz"] = pz.mean() * np.sqrt(n)
    f[f"{pfx}_absmean"] = np.abs(pz).mean()
    f[f"{pfx}_skew"] = skew(pz)
    h = np.histogram(u, bins=10, range=(0, 1))[0] / n
    for i in range(10):
        f[f"{pfx}_dec{i}"] = h[i]
    f[f"{pfx}_chi2"] = ((h - 0.1) ** 2).sum() * 10 * n
    f[f"{pfx}_acf1"] = acf(pz, 1)
    us = np.sort(u)
    f[f"{pfx}_ks"] = max(np.abs(us - np.arange(1, n + 1) / n).max(), np.abs(us - np.arange(n) / n).max())
    f[f"{pfx}_run_hi"] = longest_run(u > 0.9); f[f"{pfx}_run_lo"] = longest_run(u < 0.1)
    f[f"{pfx}_run_hi_rel"] = f[f"{pfx}_run_hi"] / n
    f[f"{pfx}_tails"] = (u > 0.95).mean() + (u < 0.05).mean()
    th = np.array_split(pz, 3)
    for i, t in enumerate(th):
        f[f"{pfx}_t{i}_mean"] = t.mean()
    f[f"{pfx}_sqmean"] = (pz ** 2).mean()
    return u, pz


def fam_pit(c):
    f = {}
    pre, post = c.pre, c.post
    _pit_feats(f, "pit", pre, post)
    for p in (1, 2):
        _, _, ep, eq = c.ar(p)
        _pit_feats(f, f"pit_ar{p}", ep, eq)
    _, _, ep, eq = c.ar(2)
    u = pit(np.sort(np.abs(ep)), np.abs(eq))
    pz = ndtri(u)
    f["pit_absar2_mean"] = pz.mean(); f["pit_absar2_hi"] = (u > 0.9).mean(); f["pit_absar2_acf1"] = acf(pz, 1)
    d1p, d1q = np.diff(pre), np.diff(post)
    u = pit(np.sort(d1p), d1q); pz = ndtri(u)
    f["pit_d1_std"] = pz.std(); f["pit_d1_tails"] = (u > 0.95).mean() + (u < 0.05).mean()
    return f


def fam_vol(c):
    f = {}
    npre = c.npre
    e, _, ep, eq = c.ar(2)
    ee = np.concatenate([ep, eq]); nb = len(ep)
    for w in (16, 48):
        if nb <= 2 * w or len(eq) <= 2 * w:
            rs = rolling_std(ee, min(w, 8)); ww = min(w, 8)
        else:
            rs = rolling_std(ee, w); ww = w
        rp, rq = rs[:nb - ww + 1], rs[nb:]
        if len(rq) < 3:
            rq = rs[-3:]
        ref = np.median(rp) + 1e-6
        f[f"rv{w}_post_mean_lr"] = np.log(rq.mean() / ref + 1e-6)
        f[f"rv{w}_post_max_lr"] = np.log(rq.max() / ref + 1e-6)
        f[f"rv{w}_pre_max_lr"] = np.log(rp.max() / ref + 1e-6)
        f[f"rv{w}_post_min_lr"] = np.log(rq.min() / ref + 1e-6)
        th = np.array_split(rq, 3)
        f[f"rv{w}_post_last_lr"] = np.log(th[2].mean() / ref + 1e-6)
        f[f"rv{w}_post_first_lr"] = np.log(th[0].mean() / ref + 1e-6)
        f[f"rv{w}_frac_hi"] = (rq > np.quantile(rp, 0.95)).mean()
        f[f"rv{w}_frac_lo"] = (rq < np.quantile(rp, 0.05)).mean()
        f[f"rv{w}_cv_pre"] = rp.std() / ref
    for k in (1, 2, 3):
        a, b = acf(ep ** 2, k), acf(eq ** 2, k)
        f[f"sqacf{k}_pre"], f[f"sqacf{k}_post"], f[f"sqacf{k}_diff"] = a, b, b - a
    # ARCH-like persistence: sum of squared-residual acf lags 1..10
    f["sqacf_sum10_pre"] = sum(acf(ep ** 2, k) for k in range(1, 11))
    f["sqacf_sum10_post"] = sum(acf(eq ** 2, k) for k in range(1, 11))
    # EWMA variance dynamics
    for lam in (0.94, 0.8):
        v = lfilter([1 - lam], [1, -lam], ee ** 2, zi=[lam * np.mean(ep[:20] ** 2)])[0]
        vp, vq = v[:nb], v[nb:]
        ref = np.median(vp) + 1e-6
        f[f"ewma{int(lam*100)}_post_mean_lr"] = np.log(vq.mean() / ref + 1e-6)
        f[f"ewma{int(lam*100)}_end_lr"] = np.log(vq[-1] / ref + 1e-6)
        f[f"ewma{int(lam*100)}_post_q90_lr"] = np.log(np.quantile(vq, 0.9) / (np.quantile(vp, 0.9) + 1e-6) + 1e-6)
    # local-level model (SES) fit on pre, run on full
    z = c.z; pre = c.pre
    best = None
    for a in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        lv = lfilter([a], [1, -(1 - a)], pre, zi=[(1 - a) * pre[0]])[0]
        err = pre[1:] - lv[:-1]
        sse = err @ err
        if best is None or sse < best[0]:
            best = (sse, a)
    a = best[1]
    f["ll_alpha"] = a
    lv = lfilter([a], [1, -(1 - a)], z, zi=[(1 - a) * z[0]])[0]
    err = z[1:] - lv[:-1]
    errp, errq = err[:npre - 1], err[npre - 1:]
    s = max(1.4826 * np.median(np.abs(errp - np.median(errp))), 0.25 * errp.std(), 1e-6)
    zq = errq / s; zp = errp / s
    f["ll_post_mean"] = zq.mean(); f["ll_post_absmean_lr"] = lratio(np.abs(zq).mean(), np.abs(zp).mean())
    f["ll_post_rms_lr"] = lratio(np.sqrt((zq ** 2).mean()), np.sqrt((zp ** 2).mean()))
    cs = np.cumsum(zq - zp.mean())
    f["ll_cusum_max"] = np.abs(cs).max() / np.sqrt(len(zq))
    f["ll_level_end_shift"] = (lv[-1] - lv[npre - 1])
    f["ll_level_post_mean_shift"] = lv[npre:].mean() - lv[:npre].mean()
    f["ll_level_post_range"] = lv[npre:].max() - lv[npre:].min()
    f["ll_level_pre_range"] = lv[:npre].max() - lv[:npre].min()
    f["ll_err_acf1_post"] = acf(zq, 1); f["ll_err_acf1_pre"] = acf(zp, 1)
    return f


def fam_tier2(c):
    """profile (pre-only) + placebo + persistence."""
    f = {}
    pre, post, npre = c.pre, c.post, c.npre
    npost = len(post)
    # profile: pre-only
    def vr(x, q):
        d = np.diff(x); dq = x[q:] - x[:-q]
        return dq.var() / (q * d.var() + EPS)
    for q in (2, 4, 8):
        f[f"vr{q}_pre"] = vr(pre, q); f[f"vr{q}_post"] = vr(post, q)
    # ADF-like t stat on pre / post
    for nm, x in (("pre", pre), ("post", post)):
        dx = np.diff(x); X = np.column_stack([np.ones(len(dx)), x[:-1]])
        b, r, ssr = ols(X, dx)
        try:
            se = np.sqrt(ssr / (len(dx) - 2) * np.linalg.inv(X.T @ X)[1, 1])
            f[f"adf_t_{nm}"] = b[1] / (se + EPS)
        except np.linalg.LinAlgError:
            f[f"adf_t_{nm}"] = 0.0
        S = np.cumsum(x - x.mean())
        f[f"kpss_{nm}"] = (S @ S) / (len(x) ** 2 * lrv_bartlett(x))
        # Hurst via R/S on blocks
        rs = []; ns = []
        for m in (8, 16, 32, 64):
            if len(x) < 2 * m: continue
            B = x[:len(x) // m * m].reshape(-1, m)
            Y = np.cumsum(B - B.mean(1, keepdims=True), 1)
            R = Y.max(1) - Y.min(1); S_ = B.std(1) + EPS
            rs.append(np.log((R / S_).mean())); ns.append(np.log(m))
        f[f"hurst_{nm}"] = np.polyfit(ns, rs, 1)[0] if len(ns) >= 2 else 0.5
    f["hurst_diff"] = f["hurst_post"] - f["hurst_pre"]
    f["kpss_diff"] = f["kpss_post"] - f["kpss_pre"]
    # placebo: same statistics on pre halves and post halves
    def stats(a, b):
        sa, sb = a.std() + EPS, b.std() + EPS
        t = (b.mean() - a.mean()) / np.sqrt(lrv_bartlett(a) / len(a) + lrv_bartlett(b) / len(b))
        return t, ks_stat(a, b)[0], np.log(sb / sa)
    h = npre // 2
    tp, kp, vp = stats(pre[:h], pre[h:])
    h2 = npost // 2
    tq, kq, vq = stats(post[:h2], post[h2:])
    ta, ka, va = stats(pre, post)
    f["plc_pre_t"], f["plc_pre_ks"], f["plc_pre_v"] = tp, kp, vp
    f["plc_post_t"], f["plc_post_ks"], f["plc_post_v"] = tq, kq, vq
    f["plc_t_ratio"] = abs(ta) / (abs(tp) + 1.0)
    f["plc_ks_ratio"] = ka / (kp + 0.02)
    f["plc_v_ratio"] = abs(va) / (abs(vp) + 0.05)
    # persistence: post second half vs pre, block consistency
    t2, k2, v2 = stats(pre, post[h2:])
    f["pers_late_t"], f["pers_late_ks"], f["pers_late_v"] = t2, k2, v2
    nb = 8
    blocks = np.array_split(post, nb)
    se = np.sqrt(lrv_bartlett(pre) / max(len(blocks[-1]), 1))
    bm = np.array([b.mean() for b in blocks]) - pre.mean()
    f["pers_frac_blocks_up"] = (bm > 2 * se).mean(); f["pers_frac_blocks_dn"] = (bm < -2 * se).mean()
    f["pers_sign_consist"] = abs(np.sign(bm).mean())
    bs = np.array([np.log(b.std() + EPS) for b in blocks]) - np.log(pre.std() + EPS)
    f["pers_vol_blocks_up"] = (bs > 0.3).mean(); f["pers_vol_blocks_dn"] = (bs < -0.3).mean()
    f["pers_vol_min"] = bs.min(); f["pers_vol_max"] = bs.max()
    f["pers_mean_min_abs"] = np.sign(bm[0]) * np.abs(bm).min() if np.all(np.sign(bm) == np.sign(bm[0])) else 0.0
    return f


FAMILIES = {"wide": fam_wide, "shape": fam_shape, "pit": fam_pit, "vol": fam_vol, "tier2": fam_tier2}


def extract_one(x, npre, fams):
    c = Ctx(x, npre)
    out = {}
    for fm in fams:
        d = FAMILIES[fm](c)
        out.update(d)
    return out
