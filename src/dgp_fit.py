"""v4 P1a: per-segment candidate-model battery (conditional ML, BIC selection) for generator recovery.
Each segment is fitted in pre-scaled units z = (x - med_pre)/s_pre (affine-invariant).
Mean models (Gaussian conditional ML on the common sample t >= 2): iid, trend, ar1, ar2, ar1t, ar2t, arma11 (CSS),
rw (random walk = AR(1) with phi = 1, plus drift), ll (local level = MA(1) on first differences, CSS).
Innovation families on the best mean model's residuals: gauss, laplace, uniform, student-t, skew-normal, 2-gauss mixture,
GARCH(1,1)-gauss.  Everything recorded per segment.
  python src/dgp_fit.py <n_sample> -> work/v3/dgp/fits_{pre,post}.parquet (stratified dev subsample, seed 4401)"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, json, warnings
from pathlib import Path
import numpy as np, pandas as pd

LOG2PI = np.log(2 * np.pi)
MEAN_MODELS = ("iid", "trend", "ar1", "ar2", "ar1t", "ar2t", "arma11", "rw", "ll")


def _ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, e


def _gll(e):
    m = len(e); s2 = max(e @ e / m, 1e-300)
    return -0.5 * m * (LOG2PI + np.log(s2) + 1.0), s2


def _css_ma(u, theta):
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, theta], u)


def fit_mean(z):
    from scipy.optimize import minimize_scalar, minimize
    n = len(z); t = np.arange(n) / n
    y, z1, z2, tt, one = z[2:], z[1:-1], z[:-2], t[2:], np.ones(n - 2)
    out = {}
    designs = {"iid": [one], "trend": [one, tt], "ar1": [one, z1], "ar2": [one, z1, z2], "ar1t": [one, tt, z1], "ar2t": [one, tt, z1, z2]}
    for k, cols in designs.items():
        b, e = _ols(np.column_stack(cols), y)
        ll, s2 = _gll(e); out[k] = dict(b=b, e=e, ll=ll, k=len(cols) + 1, s2=s2)
    # random walk with drift: dz_t = c + eps
    d = y - z1; e = d - d.mean(); ll, s2 = _gll(e); out["rw"] = dict(b=np.array([d.mean()]), e=e, ll=ll, k=2, s2=s2)
    # local level: dz_t = eta_t + eps_t - eps_{t-1}  <=>  MA(1) on dz with theta in (-1, 0]
    dc = d - d.mean()
    r = minimize_scalar(lambda th: _css_ma(dc, th) @ _css_ma(dc, th), bounds=(-0.995, 0.0), method="bounded")
    e = _css_ma(dc, r.x); ll, s2 = _gll(e); out["ll"] = dict(b=np.array([d.mean(), r.x]), e=e, ll=ll, k=3, s2=s2)
    # ARMA(1,1) CSS, started from AR(1)
    c0, p0 = out["ar1"]["b"]
    f = lambda q: (lambda u: u @ u)(_css_ma(y - q[0] - q[1] * z1, q[2]))
    r = minimize(f, x0=[c0, p0, 0.0], method="L-BFGS-B", bounds=[(None, None), (-0.995, 0.995), (-0.995, 0.995)])
    e = _css_ma(y - r.x[0] - r.x[1] * z1, r.x[2]); ll, s2 = _gll(e); out["arma11"] = dict(b=r.x, e=e, ll=ll, k=4, s2=s2)
    m = n - 2
    for k in out:
        out[k]["bic"] = -2 * out[k]["ll"] + out[k]["k"] * np.log(m)
    return out


def _garch_nll(q, e):
    w, a, bb = q
    if w <= 0 or a < 0 or bb < 0 or a + bb >= 0.999:
        return 1e12
    from scipy.signal import lfilter
    # h_t = w + a e_{t-1}^2 + b h_{t-1}  (h_0 = sample var)
    x = np.r_[e.var(), e[:-1] ** 2]
    h = lfilter([1.0], [1.0, -bb], w + a * x)
    h[0] = e.var()
    h = np.maximum(h, 1e-12)
    return 0.5 * np.sum(LOG2PI + np.log(h) + e ** 2 / h)


def fit_family(e):
    from scipy import stats
    from scipy.optimize import minimize
    m = len(e); e = e - e.mean(); sd = e.std() + 1e-300; u = e / sd
    res = {}
    res["gauss"] = dict(ll=_gll(e)[0], k=1)
    b = np.mean(np.abs(e - np.median(e))); res["laplace"] = dict(ll=-m * np.log(2 * b) - m, k=2)
    res["uniform"] = dict(ll=-m * np.log(np.ptp(e) * (1 + 1.0 / m)), k=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, loc, sc = stats.t.fit(u, floc=0.0)
        res["t"] = dict(ll=float(np.sum(stats.t.logpdf(u, df, 0, sc))) - m * np.log(sd), k=2, nu=float(df))
        a, loc, sc = stats.skewnorm.fit(u)
        res["skewnorm"] = dict(ll=float(np.sum(stats.skewnorm.logpdf(u, a, loc, sc))) - m * np.log(sd), k=3, skew_a=float(a))
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(2, random_state=0, n_init=2).fit(u[:, None])
        w = g.weights_; mu = g.means_[:, 0]; s = np.sqrt(g.covariances_[:, 0, 0]); j = np.argsort(-w)
        res["gmm2"] = dict(ll=float(g.score(u[:, None]) * m) - m * np.log(sd), k=5, w_minor=float(w[j[1]]),
                           sd_ratio=float(s[j[1]] / s[j[0]]), mu_sep=float(abs(mu[j[1]] - mu[j[0]]) / s[j[0]]))
        r = minimize(_garch_nll, x0=[0.1 * e.var(), 0.1, 0.8], args=(e,), method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-4, fatol=1e-3))
        res["garch"] = dict(ll=-float(r.fun), k=3, g_alpha=float(r.x[1]), g_beta=float(r.x[2]))
    for k in res:
        res[k]["bic"] = -2 * res[k]["ll"] + res[k]["k"] * np.log(m)
    return res, dict(kurt=float(stats.kurtosis(u)), skew=float(stats.skew(u)))


def fit_seg(z):
    z = np.asarray(z, float); n = len(z)
    M = fit_mean(z)
    best = min(MEAN_MODELS, key=lambda k: M[k]["bic"])
    F, mom = fit_family(M[best]["e"])
    fam = min(F, key=lambda k: F[k]["bic"])
    a1 = M["ar1"]["b"][1]; c2, p1, p2 = M["ar2"]["b"]
    roots = np.roots([1, -p1, -p2]); rho = float(np.max(np.abs(roots)))
    cyc = bool(np.iscomplexobj(roots) and np.abs(roots.imag).max() > 1e-9)
    period = float(2 * np.pi / abs(np.angle(roots[0]))) if cyc else np.nan
    row = dict(n=n, mean_model=best, family=fam, mean=float(z.mean()), sd=float(z.std()),
               sig_iid=float(np.sqrt(M["iid"]["s2"])), sig_best=float(np.sqrt(M[best]["s2"])),
               ar1_phi=float(a1), ar2_phi1=float(p1), ar2_phi2=float(p2), ar2_rho=rho, ar2_cyclic=cyc, ar2_period=period,
               trend_slope=float(M["trend"]["b"][1]), arma_phi=float(M["arma11"]["b"][1]), arma_theta=float(M["arma11"]["b"][2]),
               ll_theta=float(M["ll"]["b"][1]), sig_ar1=float(np.sqrt(M["ar1"]["s2"])), sig_ar2=float(np.sqrt(M["ar2"]["s2"])),
               t_nu=F["t"]["nu"], skew_a=F["skewnorm"]["skew_a"], gmm_w=F["gmm2"]["w_minor"], gmm_sdr=F["gmm2"]["sd_ratio"],
               gmm_sep=F["gmm2"]["mu_sep"], g_alpha=F["garch"]["g_alpha"], g_beta=F["garch"]["g_beta"], **mom)
    for k in MEAN_MODELS:
        row[f"bic_{k}"] = float(M[k]["bic"])
    for k in F:
        row[f"fbic_{k}"] = float(F[k]["bic"])
    return row


def _work(items):
    out = []
    for sid, x, npre in items:
        x = np.asarray(x, float); pre = x[:npre]; med = np.median(pre)
        s = max(1.4826 * np.median(np.abs(pre - med)), 0.25 * pre.std())
        s = s if s > 1e-12 * (abs(med) + 1) else 1.0
        z = (x - med) / s
        r = {}
        for seg, zz in (("pre", z[:npre]), ("post", z[npre:])):
            try:
                d = fit_seg(zz)
            except Exception as ex:
                d = {"err": str(ex)[:80]}
            d["id"] = sid; r[seg] = d
        r["pre"]["raw_med"] = float(med); r["pre"]["raw_scale"] = float(s)
        out.append(r)
    return out


if __name__ == "__main__":
    from joblib import Parallel, delayed
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).parent))
    import extract, h2
    from sklearn.model_selection import train_test_split
    n_s = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    dev = h2.dev_ids(); y = h2.labels().loc[dev].values
    sub, _ = train_test_split(dev, train_size=n_s, stratify=y, random_state=4401)
    ids, ser = extract.load_series(ROOT / "train_series.parquet"); pos = {k: j for j, k in enumerate(ids)}
    items = [(i, ser[pos[i]][0], ser[pos[i]][1]) for i in sub]
    t0 = time.time()
    ch = [items[i:i + 50] for i in range(0, len(items), 50)]
    res = [r for part in Parallel(n_jobs=int(os.environ.get("FIT_JOBS", "11")))(delayed(_work)(c) for c in ch) for r in part]
    D = ROOT / "work" / "v3" / "dgp"; D.mkdir(parents=True, exist_ok=True)
    lab = h2.labels()
    for seg in ("pre", "post"):
        df = pd.DataFrame([r[seg] for r in res]).set_index("id"); df["label"] = lab.loc[df.index].values
        df.to_parquet(D / f"fits_{seg}.parquet")
    print(f"[dgp_fit] {len(res)} series x 2 segments in {time.time()-t0:.0f}s", flush=True)
