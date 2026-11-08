"""v6 S1: efficient-score and GLR evidence. ONE cached feature family 'glr' (column groups by prefix):
  sc_*  (1b) efficient score tests at "no break", from the best-fitting no-break model of the WHOLE series
  lr_*  (1c) generalized likelihood ratios: refit allowing a post-boundary change, 2*(logL1 - logL0)
  ma_*  (1d) the same statistics averaged over all candidate models with BIC weights + how ambiguous the model choice is
  cal_* (1e) finite-sample calibrated versions (z-score and log tail probability) from a simulated lookup table
  tf_*  (1f) stratum transforms: first differences, sinusoid extrapolated from the pre segment, GARCH-filtered residuals
  aux_* side data for S2 (one-step shift estimates and standard errors); never used as model features
Plain-language idea: fit ONE model to the whole series as if nothing changed. If something did change after the
boundary, the fitted model "wants" to move in that direction there; the score measures that pull, scaled by how much
pull pure chance would produce. Score tests are the most powerful tests for small changes.
All computations use the pre-boundary robust-scaled series c.z, so the family is affine-invariant; reversal TTA is
handled by the view layer. CLI: python src/feats7.py calib | sizecheck | profile"""
import os, sys, json, hashlib                                  # standard tools: environment, arguments, json, checksums
from pathlib import Path                                       # file-path handling
import numpy as np                                             # fast array maths
from scipy import optimize, special                            # 1-D/n-D optimisers and the log-gamma function
from scipy.signal import lfilter                               # fast linear recursions (used for ARMA and GARCH)
import feats                                                   # project feature helpers (robust scaling, Ctx)
from feats import FAMILIES                                     # registry of feature families used by extract.py

ROOT = Path(__file__).resolve().parent.parent                  # project folder
CALIB_PATH = __import__("paths").CONFIG / "calib_table.npz"     # where the 1e lookup table is stored
CALIB_MD5 = "49be79d93df703f3f2a2936a3b8d61a9"                 # checksum of the table, written in after it is built (B=2000, seed 20261107)
CAL_SEED = 20261107                                            # one fixed random seed for the calibration simulations
MEANS = ("iid", "ar1", "ar2", "arma11", "trend", "rw", "sine")  # candidate no-break mean models (the recovered set)
INNOVS = ("gauss", "t", "garch", "mix")                        # candidate noise ("innovation") distributions
LOC = ("lvl", "slp", "dep", "dep2", "s_amp", "s_frq", "s_phs")  # break directions that move the mean of the series
COORDS = LOC + ("scl", "tail", "skew")                         # plus break directions in noise scale, tail weight, skewness
LS = ("lvl", "scl")                                            # level + scale subspace (the classic break types)
DS = ("dep", "dep2", "tail", "skew")                           # dependence + shape subspace (the harder break types)
LOG2PI = np.log(2 * np.pi)                                     # constant used in the normal density
NU_GRID = np.exp(np.linspace(np.log(2.1), np.log(200.0), 25))  # candidate t tail parameters (2.1 = very heavy .. 200 = nearly normal)
CAL_STATS = ("LM_full", "LM_ls", "LM_ds", "zsq_lvl", "zsq_scl", "zsq_dep", "zsq_tail", "GLR_joint")  # statistics calibrated in 1e
CAL_P = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999])   # quantile levels stored per calibration cell
CAL_N = np.array([80, 160, 320, 640])                          # grid of pre / post segment lengths for the table
CAL_FAMS = ([(m, "gauss", 0.0) for m in ("iid", "ar2", "arma11", "trend", "rw", "sine")] +   # Gaussian-noise cells
            [("ar1", "gauss", p) for p in (0.3, 0.6, 0.9)] +   # AR(1) cells at three persistence levels
            [(m, "t", 0.0) for m in ("iid", "ar2", "arma11", "trend", "rw", "sine")] +     # Student-t-noise cells
            [("ar1", "t", p) for p in (0.3, 0.6, 0.9)] +       # AR(1) with t noise at three persistence levels
            [("iid", "garch", 0.0), ("iid", "mix", 0.0)])      # volatility-clustering and mixture-noise cells
_CAL = None                                                    # the lookup table, loaded once per process


# ----------------------------------------------------------------------------- small numeric helpers
def _wls(W, y, w=None):                                        # (weighted) least squares fit of y on the columns of W
    if w is None:                                              # plain least squares
        b = np.linalg.lstsq(W, y, rcond=None)[0]               # coefficients minimising the squared residuals
    else:                                                      # weighted version: rows with larger w count more
        sw = np.sqrt(w)                                        # square-root weights turn it into an ordinary fit
        b = np.linalg.lstsq(W * sw[:, None], y * sw, rcond=None)[0]   # weighted coefficients
    return b, y - W @ b                                        # coefficients and residuals


def _logt(r, nu):                                              # log density of a standard Student-t with nu degrees of freedom
    return special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(r * r / nu)


def _logmix(r, w, k):                                          # log density of a 2-part normal mixture: narrow (1-w) and k-times wider (w)
    return np.logaddexp(np.log1p(-w) - 0.5 * (LOG2PI + r * r), np.log(w) - np.log(k) - 0.5 * (LOG2PI + (r / k) ** 2))


def _garch_var(e2, om, a, b):                                  # GARCH(1,1) conditional variance path for squared residuals e2
    v0 = max(float(e2.mean()), 1e-12)                          # start from the overall average variance
    u = om + a * e2[:-1]                                       # today's variance input from yesterday's squared residual
    v = np.empty(len(e2)); v[0] = v0                           # output array, first value = start value
    v[1:] = lfilter([1.0], [1.0, -b], u, zi=[b * v0])[0]       # recursion v_t = om + a*e2_{t-1} + b*v_{t-1}, done in C
    return np.maximum(v, 1e-12)                                # never allow zero variance


def _garch_par(q):                                             # map unconstrained numbers to valid GARCH parameters
    P = 0.995 / (1 + np.exp(-q[1]))                            # persistence a+b kept below 1 (stationary)
    a = P / (1 + np.exp(-q[2]))                                # reaction to news a is a share of the persistence
    return np.exp(q[0]), a, P - a                              # (omega > 0, a, b)


def _polish(g, x0, lo, hi, h):                                 # refine a minimiser by solving derivative = 0 with 3 secant steps
    x1 = min(max(x0 + h, lo), hi); g0, g1 = g(x0), g(x1)       # two nearby points and their slopes
    for _ in range(3):                                         # a fixed number of steps keeps the result a smooth function of the data
        if g1 == g0:                                           # slopes identical: nothing more to gain
            break
        x0, x1 = x1, min(max(x1 - g1 * (x1 - x0) / (g1 - g0), lo), hi)   # secant step towards zero slope, kept inside the bounds
        g0, g1 = g1, g(x1)                                     # slope at the new point
    return x1


def _sine_freq(z):                                             # dominant angular frequency of the whole series (periods 4 .. n/2)
    n = len(z); t = np.arange(n)                               # length and time index
    zc = z - np.polyval(np.polyfit(t, z, 1), t)                # remove a straight-line trend so it does not look like a slow wave
    P = np.abs(np.fft.rfft(zc)) ** 2                           # periodogram: power at each Fourier frequency
    f = np.fft.rfftfreq(n)                                     # those frequencies in cycles per step
    ok = (f >= 2.0 / n) & (f <= 0.25)                          # only periods between 4 steps and half the series
    k = np.flatnonzero(ok)[np.argmax(P[ok])]                   # the strongest one
    w0 = 2 * np.pi * f[k]                                      # as an angular frequency

    def rss(w):                                                # misfit of a sinusoid at frequency w
        X = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t)])   # constant + cosine + sine columns
        return float(np.sum(_wls(X, z)[1] ** 2))               # sum of squared residuals
    w1 = float(optimize.minimize_scalar(rss, bounds=(w0 - np.pi / n, w0 + np.pi / n), method="bounded",
                                        options={"xatol": 1e-12}).x)   # refine between neighbouring Fourier frequencies

    def slope(w):                                              # exact derivative of the misfit w.r.t. frequency (amplitudes re-fitted)
        X = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t)]); b, e = _wls(X, z)   # fit at frequency w
        return float(np.sum(e * t * (-b[1] * np.sin(w * t) + b[2] * np.cos(w * t))))   # residuals times the frequency direction
    return _polish(slope, w1, w0 - np.pi / n, w0 + np.pi / n, 1e-7)   # pin the minimum down to round-off precision


# ----------------------------------------------------------------------------- no-break (H0) model fits
def _design(m, z, npre, om):                                   # regression form "y = W @ beta + noise" of mean model m
    n = len(z); tt = np.arange(2, n)                           # rows t = 2..n-1: the same sample for every model (fair BIC)
    one = np.ones(len(tt))                                     # constant column
    if m == "rw":                                              # random walk: model the step changes
        return z[tt] - z[tt - 1], one[:, None], tt             # y = change from the previous value, W = drift only
    W = {"iid": [one], "ar1": [one, z[tt - 1]], "ar2": [one, z[tt - 1], z[tt - 2]], "arma11": [one, z[tt - 1]],
         "trend": [one, (tt - n / 2) / n],                     # straight-line trend in scaled time
         "sine": [one, np.cos(om * tt), np.sin(om * tt)]}[m]   # sinusoid at the series' own frequency
    return z[tt], np.column_stack(W), tt                       # target, regressors, time index of each row


def _filt(th):                                                 # the ARMA(1,1) "undo the moving average" filter for parameter th
    return (lambda v: lfilter([1.0], [1.0, th], v, axis=0)) if th != 0.0 else (lambda v: v)   # identity when there is no MA part


def _fit_mean(m, z, npre, om):                                 # Gaussian least-squares fit of mean model m (shared by all noise types)
    y, W, tt = _design(m, z, npre, om)                         # regression form
    th = 0.0                                                   # moving-average parameter (only ARMA uses it)
    if m == "arma11":                                          # ARMA(1,1): search the MA parameter, least squares for the rest
        th = float(optimize.minimize_scalar(lambda q: float(np.sum(_wls(_filt(q)(W), _filt(q)(y))[1] ** 2)),
                                            bounds=(-0.9, 0.9), method="bounded", options={"xatol": 1e-11}).x)

        def slope(q):                                          # exact derivative of the misfit w.r.t. the MA parameter
            e_ = _wls(_filt(q)(W), _filt(q)(y))[1]             # residuals at MA parameter q (other coefficients re-fitted)
            return float(np.sum(e_ * _filt(q)(np.r_[0.0, e_[:-1]])))   # residuals times their MA direction
        th = _polish(slope, th, -0.9, 0.9, 1e-7)               # pin the minimum down to round-off precision
    F = _filt(th)                                              # the filter for this fit
    Wf, yf = F(W), F(y)                                        # filtered regressors and target (plain copies unless ARMA)
    b, e = _wls(Wf, yf)                                        # least-squares coefficients and residuals
    return dict(m=m, y=y, W=W, Wf=Wf, yf=yf, tt=tt, th=th, F=F, b=b, e=e, om=om, n=len(z), npre=npre, z=z)


def _fit_innov(base, innov):                                   # fit one noise distribution on top of a mean model
    f = dict(base); f["innov"] = innov                         # copy the mean-model fit and tag the noise type
    Wf, yf, b, e = base["Wf"], base["yf"], base["b"], base["e"]   # shorthand
    N = len(e); k = Wf.shape[1] + (base["m"] == "arma11") + (base["m"] == "sine")   # rows, mean parameters (+MA, +frequency)
    if innov == "gauss":                                       # normal noise
        s = np.sqrt(max(float(np.mean(e * e)), 1e-12))         # noise scale = root mean square residual
        r = e / s; ll = float(np.sum(-0.5 * (LOG2PI + r * r)) - N * np.log(s)); k += 1   # log-likelihood; +1 for the scale
        f.update(s=s, r=r, ll=ll, shape={})                    # store
    elif innov == "t":                                         # Student-t noise: big residuals are down-weighted
        nu, s2 = 8.0, max(float(np.mean(e * e)), 1e-12)        # start values
        for it in range(12):                                   # iteratively reweighted least squares (EM for the t model)
            w = (nu + 1) / (nu + e * e / s2)                   # weight of each point: outliers get small weights
            b, e = _wls(Wf, yf, w)                             # refit the mean with those weights
            s2 = max(float(np.mean(w * e * e)), 1e-12)         # update the scale
            if it % 4 == 3:                                    # every few rounds, update the tail parameter nu
                r = e / np.sqrt(s2)                            # standardised residuals
                nu = float(NU_GRID[np.argmax(_logt(r[None, :], NU_GRID[:, None]).sum(1))])   # best nu on a fixed grid (stable to round-off)
        s = np.sqrt(s2); r = e / s                             # final scale and standardised residuals
        ll = float(np.sum(_logt(r, nu)) - N * np.log(s)); k += 2   # log-likelihood; +scale +nu
        f.update(b=b, e=e, s=s, r=r, ll=ll, shape={"nu": nu})  # store
    elif innov == "mix":                                       # mixture of a narrow and a wide normal
        w_, kk, s2 = 0.1, 3.0, max(float(np.mean(e * e)), 1e-12)   # start: 10% of points from a 3x wider normal
        for _ in range(15):                                    # EM iterations
            r = e / np.sqrt(s2)                                # standardised residuals
            l1 = np.log1p(-w_) - 0.5 * r * r                   # log weight of "narrow" for each point
            l2 = np.log(w_) - np.log(kk) - 0.5 * (r / kk) ** 2  # log weight of "wide" for each point
            p = np.exp(l2 - np.logaddexp(l1, l2))              # probability each point came from the wide part
            lam = (1 - p) + p / kk ** 2                        # precision weight of each point
            b, e = _wls(Wf, yf, lam)                           # refit the mean with those weights
            s2 = max(float(np.mean(lam * e * e)), 1e-12)       # update the narrow scale
            w_ = float(np.clip(p.mean(), 0.005, 0.5))          # update the share of wide points
            kk = float(np.sqrt(np.clip(np.sum(p * e * e) / s2 / max(p.sum(), 1e-9), 2.25, 400.0)))   # update how much wider
        s = np.sqrt(s2); r = e / s                             # final scale and standardised residuals
        ll = float(np.sum(_logmix(r, w_, kk)) - N * np.log(s)); k += 3   # log-likelihood; +scale +share +width
        f.update(b=b, e=e, s=s, r=r, ll=ll, shape={"w": w_, "k": kk})   # store
    else:                                                      # GARCH(1,1): noise variance follows recent squared residuals
        e2 = e * e                                             # squared residuals of the least-squares mean fit (two-step)

        v_bar = max(float(e2.mean()), 1e-12)                   # average variance (variance targeting fixes omega from it)
        best_q, best_l = None, -np.inf                         # best grid point so far
        for P in (0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.96, 0.985):   # persistence a+b on a fixed grid (no optimiser: round-off cannot change the answer)
            for fa in (0.05, 0.15, 0.3, 0.5, 0.8):             # share of the persistence that reacts to news (a / (a+b))
                q = np.array([np.log(v_bar * (1 - P)), np.log(P / (0.995 - P)), np.log(fa / (1 - fa))])   # same parameters in unconstrained form
                v = _garch_var(e2, *_garch_par(q))             # conditional variances for this grid point
                l = -0.5 * float(np.sum(np.log(v) + e2 / v))   # Gaussian log-likelihood (constants dropped)
                if l > best_l:
                    best_q, best_l = q, l                      # keep the best
        q = best_q                                             # chosen GARCH parameters
        s = np.sqrt(_garch_var(e2, *_garch_par(q)))            # conditional standard deviations
        r = e / s; ll = float(np.sum(-0.5 * (LOG2PI + r * r) - np.log(s))); k += 3   # log-likelihood; +omega +a +b
        f.update(s=s, r=r, ll=ll, shape={"q": q})              # store
    f["k"] = k; f["bic"] = -2 * f["ll"] + k * np.log(N)        # parameter count and BIC (lower = better)
    return f


# ----------------------------------------------------------------------------- per-observation score pieces
def _psi(f):                                                   # "pull" of each residual on a location parameter, and r*pull
    r, innov = f["r"], f["innov"]                              # standardised residuals and noise type
    if innov == "t":                                           # Student-t: large residuals pull less (built-in outlier protection)
        nu = f["shape"]["nu"]; psi = (nu + 1) * r / (nu + r * r)
    elif innov == "mix":                                       # mixture: pull = weighted blend of the two normals
        w_, kk = f["shape"]["w"], f["shape"]["k"]              # mixture parameters
        l1 = np.log1p(-w_) - 0.5 * r * r; l2 = np.log(w_) - np.log(kk) - 0.5 * (r / kk) ** 2   # log weights of the two parts
        p = np.exp(l2 - np.logaddexp(l1, l2))                  # probability of the wide part
        psi = r * ((1 - p) + p / kk ** 2)                      # pull shrinks for points that look like they came from the wide part
    else:                                                      # Gaussian and GARCH (Gaussian given the variance)
        psi = r                                                # pull is the residual itself
    return psi, r * psi


def _dshape(f, which):                                         # per-point derivative of log density w.r.t. a shape parameter
    r, h = f["r"], 1e-4                                        # residuals and finite-difference step
    if f["innov"] == "t":                                      # t: derivative w.r.t. log(nu)
        nu = f["shape"]["nu"]
        return (_logt(r, nu * np.exp(h)) - _logt(r, nu * np.exp(-h))) / (2 * h)
    if f["innov"] == "mix":                                    # mixture: w.r.t. logit(share) or log(width)
        w_, kk = f["shape"]["w"], f["shape"]["k"]; lw = np.log(w_ / (1 - w_))   # current share on the logit scale
        if which == "w":                                       # share of wide points
            return (_logmix(r, 1 / (1 + np.exp(-(lw + h))), kk) - _logmix(r, 1 / (1 + np.exp(-(lw - h))), kk)) / (2 * h)
        return (_logmix(r, w_, kk * np.exp(h)) - _logmix(r, w_, kk * np.exp(-h))) / (2 * h)   # width of the wide part
    return (r ** 4 - 2 * r * r - 1) / 4                        # Gaussian: exact score for "heavier tails than normal" (1/nu at 0)


def _break_regs(f):                                            # how each mean-shifting break would move the fitted mean (row by row)
    m, z, tt, n, npre, b, om = f["m"], f["z"], f["tt"], f["n"], f["npre"], f["b"], f["om"]   # shorthand
    post = (tt >= npre).astype(float)                          # 1 after the boundary, 0 before
    p1, p2 = (tt - 1 >= npre).astype(float), (tt - 2 >= npre).astype(float)   # was the previous / second-previous point after it?
    g = {}                                                     # one column per break direction
    if m in ("ar1", "arma11"):                                 # AR(1)-type: a level shift enters partly through yesterday's value
        g["lvl"] = post - b[1] * p1
    elif m == "ar2":                                           # AR(2): through the last two values
        g["lvl"] = post - b[1] * p1 - b[2] * p2
    elif m == "rw":                                            # random walk: a level shift is one jump at the boundary
        g["lvl"] = (tt == npre).astype(float)
    else:                                                      # iid / trend / sine: the mean simply moves after the boundary
        g["lvl"] = post
    g["slp"] = post / n if m == "rw" else post * (tt - npre) / n   # a new trend starting at the boundary (random walk: new drift)
    if m in ("ar1", "arma11", "ar2"):                          # AR models: the AR coefficient itself changes
        mu = b[0] / (1 - b[1] - (b[2] if m == "ar2" else 0.0)) if abs(1 - b[1] - (b[2] if m == "ar2" else 0.0)) > 1e-3 else float(z.mean())   # process mean
        g["dep"] = post * (z[tt - 1] - mu)                     # change in lag-1 coefficient
        if m == "ar2":
            g["dep2"] = post * (z[tt - 2] - mu)                # change in lag-2 coefficient
    else:                                                      # other models: autocorrelation APPEARS in the residuals after the boundary
        g["dep"] = post * np.r_[0.0, f["e"][:-1]]              # yesterday's residual as a regressor
    if m == "sine":                                            # sinusoid: amplitude, frequency and phase can change
        A, B = b[1], b[2]; c, s = np.cos(om * tt), np.sin(om * tt)   # fitted cosine/sine weights and the waves
        g["s_amp"] = post * (A * c + B * s)                    # bigger/smaller wave after the boundary
        g["s_phs"] = post * (-A * s + B * c)                   # wave shifted in time after the boundary
        g["s_frq"] = post * om * (tt - npre) * (-A * s + B * c)   # wave speeds up / slows down after the boundary
    return {k: f["F"](v) for k, v in g.items()}, post          # ARMA: pass through the same filter as the data


def _nuis_cols(f):                                             # per-point scores of the no-break model's own parameters
    psi, rpsi = _psi(f); s = f["s"]                            # pulls and the noise scale
    cols = [psi * f["Wf"][:, j] / s for j in range(f["Wf"].shape[1])]   # one column per mean coefficient
    if f["m"] == "arma11":                                     # MA parameter of ARMA
        cols.append(psi * f["F"](np.r_[0.0, f["e"][:-1]]) / s)
    if f["m"] == "sine":                                       # estimated frequency of the sinusoid
        b, om, tt = f["b"], f["om"], f["tt"]; cols.append(psi * tt * (-b[1] * np.sin(om * tt) + b[2] * np.cos(om * tt)) / s)
    if f["innov"] == "garch":                                  # GARCH: three variance parameters (numerical derivative)
        q, e2 = f["shape"]["q"], f["e"] ** 2                   # parameters and squared residuals
        for j in range(3):                                     # for omega, persistence, and share
            dq = np.zeros(3); dq[j] = 1e-4                     # small step in one parameter
            dl = 0.25 * (np.log(_garch_var(e2, *_garch_par(q + dq))) - np.log(_garch_var(e2, *_garch_par(q - dq)))) / 1e-4   # d log(sd)
            cols.append((rpsi - 1) * dl)                       # score of that parameter
    else:
        cols.append(rpsi - 1)                                  # overall noise scale
    if f["innov"] == "t":
        cols.append(_dshape(f, "nu"))                          # tail parameter nu
    if f["innov"] == "mix":
        cols += [_dshape(f, "w"), _dshape(f, "k")]             # share and width of the wide part
    return np.column_stack(cols)


def _score_stats(f):                                           # efficient score tests for every break direction (1b)
    g, post = _break_regs(f)                                   # mean-moving break regressors
    psi, rpsi = _psi(f); r, s = f["r"], f["s"]                 # pulls, residuals, scale
    D = {c: psi * v / s for c, v in g.items()}                 # per-point score of each mean-moving break
    D["scl"] = (rpsi - 1) * post                               # noise gets bigger/smaller after the boundary
    D["tail"] = _dshape(f, "w" if f["innov"] == "mix" else "nu") * post   # tails get heavier/lighter after the boundary
    D["skew"] = np.sign(r) * rpsi * post                       # noise becomes lopsided after the boundary
    names = [c for c in COORDS if c in D]                      # break directions available for this model
    T = _nuis_cols(f); Dm = np.column_stack([D[c] for c in names])   # nuisance scores and break scores as matrices
    p = T.shape[1]; G = np.hstack([T, Dm])                     # all per-point scores side by side
    S = G.sum(0); I = G.T @ G                                  # total scores and their information (outer-product estimate)
    Itt, Itd, Idd = I[:p, :p], I[:p, p:], I[p:, p:]            # blocks: nuisance-nuisance, nuisance-break, break-break
    A = np.linalg.solve(Itt + 1e-9 * np.trace(Itt) / p * np.eye(p), Itd)   # how much each break score is explained by nuisance scores
    Ip = Idd - Itd.T @ A                                       # information left after removing the nuisance part
    Se = S[p:] - A.T @ S[:p]                                   # efficient (nuisance-corrected) break scores
    dg = np.maximum(np.diag(Ip), 1e-12)                        # variance of each efficient score
    st = {"z": dict(zip(names, Se / np.sqrt(dg))), "d": dict(zip(names, Se / dg)), "se": dict(zip(names, 1 / np.sqrt(dg)))}   # z, one-step shift, its SE

    def lm(sub):                                               # combined score statistic over a group of directions
        ix = [names.index(c) for c in sub if c in names]       # which directions exist here
        if not ix:
            return 0.0
        M = Ip[np.ix_(ix, ix)]; v = Se[ix]                     # their information and scores
        return float(v @ np.linalg.solve(M + 1e-9 * np.trace(M) / len(ix) * np.eye(len(ix)), v))   # score' * inverse-information * score
    st["LM_full"], st["LM_ls"], st["LM_ds"] = lm(names), lm(LS), lm(DS)   # all directions / level+scale / dependence+shape
    return st


# ----------------------------------------------------------------------------- likelihood-ratio (H1) refits, 1c
def _refit_ll(Wf, yf, innov, nu, segpost=None):                # log-likelihood after refitting the mean (and maybe separate scales)
    N = len(yf); w = np.ones(N); b, e = _wls(Wf, yf)           # start from plain least squares
    seg = segpost if segpost is not None else np.zeros(N, bool)   # which rows get their own (post-boundary) scale
    for _ in range(8 if (innov == "t" or segpost is not None) else 1):   # a few reweighting rounds when needed
        s2 = np.where(seg, max(float(np.mean(e[seg] ** 2 * w[seg])), 1e-12) if seg.any() else 1.0,
                      max(float(np.mean(e[~seg] ** 2 * w[~seg])), 1e-12))   # scale before / after the boundary
        w = (nu + 1) / (nu + e * e / s2) if innov == "t" else np.ones(N)   # t: down-weight outliers (nu held at its no-break value)
        b, e = _wls(Wf, yf, w / s2)                            # refit the mean with precision weights
    s2 = np.where(seg, max(float(np.mean(e[seg] ** 2 * w[seg])), 1e-12) if seg.any() else 1.0, max(float(np.mean(e[~seg] ** 2 * w[~seg])), 1e-12))
    r = e / np.sqrt(s2)                                        # standardised residuals
    lp = _logt(r, nu) if innov == "t" else -0.5 * (LOG2PI + r * r)   # log density of each point
    return float(np.sum(lp - 0.5 * np.log(s2)))                # total log-likelihood


def _glr(f):                                                   # generalized likelihood ratios for Gaussian / t models
    g, post = _break_regs(f); innov = f["innov"]; nu = f["shape"].get("nu", 0.0)   # break regressors, noise type, nu
    l0 = _refit_ll(f["Wf"], f["yf"], innov, nu)                # no-break log-likelihood under the same refit procedure
    groups = {"lvl": ["lvl"], "slp": ["slp"], "dep": [c for c in ("dep", "dep2") if c in g], "sin": [c for c in ("s_amp", "s_phs") if c in g]}
    out = {}
    for name, cs in groups.items():                            # one refit per break type, adding its regressors
        if cs:
            out[name] = max(0.0, 2 * (_refit_ll(np.column_stack([f["Wf"]] + [g[c] for c in cs]), f["yf"], innov, nu) - l0))
        else:
            out[name] = 0.0                                    # break type does not exist for this model
    pb = post.astype(bool)                                     # rows after the boundary
    out["scl"] = max(0.0, 2 * (_refit_ll(f["Wf"], f["yf"], innov, nu, pb) - l0))   # separate noise scale after the boundary
    allc = [c for c in LOC if c in g]                          # every mean-moving direction available
    out["joint"] = max(0.0, 2 * (_refit_ll(np.column_stack([f["Wf"]] + [g[c] for c in allc]), f["yf"], innov, nu, pb) - l0))   # everything at once
    out["df"] = float(len(allc) + 1)                           # number of extra parameters in the joint refit
    N = len(f["yf"])
    out["aic"], out["bic"] = out["joint"] - 2 * out["df"], out["joint"] - out["df"] * np.log(N)   # penalised versions
    return out


# ----------------------------------------------------------------------------- 1e calibration lookup
def _cal():                                                    # load the calibration table once
    global _CAL
    if _CAL is None:
        if CALIB_MD5 != "unset":                               # once frozen, refuse a table that differs from the one recorded
            assert hashlib.md5(CALIB_PATH.read_bytes()).hexdigest() == CALIB_MD5, "calibration table changed"
        d = np.load(CALIB_PATH); _CAL = {k: d[k] for k in d.files}   # arrays: summaries[cell, n_pre, n_post, stat, (mean, sd, quantiles)]
    return _CAL


def _cell_summary(m, innov, phi, npre, npost):                 # interpolated null summary for one model at these lengths
    C = _cal(); fams = [tuple(x) for x in C["fams"].tolist()]  # table and its list of (model, noise, parameter) cells
    if innov in ("garch", "mix") and (m, innov, "0.0") not in fams:   # GARCH/mixture cells exist only for iid: fall back to Gaussian
        innov = "gauss"
    if m == "ar1":                                             # AR(1): interpolate between persistence levels .3/.6/.9
        ph = float(np.clip(phi, 0.3, 0.9)); grid = (0.3, 0.6, 0.9)
        j = min(int(np.searchsorted(grid, ph, "right")) - 1, 1); wt = (ph - grid[j]) / (grid[j + 1] - grid[j])   # bracket and weight
        idx = [(fams.index(("ar1", innov, str(grid[j]))), 1 - wt), (fams.index(("ar1", innov, str(grid[j + 1]))), wt)]
    else:
        idx = [(fams.index((m, innov, "0.0")), 1.0)]           # single cell for this model
    ln = np.log(CAL_N)                                         # length grid on the log scale
    out = 0.0
    for ci, cw in idx:                                         # blend the (one or two) parameter cells
        a = np.clip(np.log(npre), ln[0], ln[-1]); bq = np.clip(np.log(npost), ln[0], ln[-1])   # lengths, clamped to the grid
        i = min(int(np.searchsorted(ln, a, "right")) - 1, len(ln) - 2); j = min(int(np.searchsorted(ln, bq, "right")) - 1, len(ln) - 2)
        u = (a - ln[i]) / (ln[i + 1] - ln[i]); v = (bq - ln[j]) / (ln[j + 1] - ln[j])   # position inside the grid square
        T = C["summ"][ci]                                      # this cell's summaries over the length grid
        out = out + cw * ((1 - u) * (1 - v) * T[i, j] + u * (1 - v) * T[i + 1, j] + (1 - u) * v * T[i, j + 1] + u * v * T[i + 1, j + 1])   # bilinear blend
    return out                                                 # [stat, (mean, sd, q...)]


def _calibrate(x, row):                                        # calibrated z-score and log tail probability of value x
    mean, sd, q = row[0], max(row[1], 1e-9), np.maximum.accumulate(row[2:]) + np.arange(len(CAL_P)) * 1e-9   # summaries (quantiles forced increasing)
    logS = np.log1p(-CAL_P)                                    # log of the tail probability at each stored quantile
    if x <= q[0]:
        lp = logS[0]                                           # below the lowest stored quantile: tail prob ~ 0.9
    elif x >= q[-1]:                                           # beyond the highest: continue the exponential tail
        lp = logS[-1] + (logS[-1] - logS[-2]) / max(q[-1] - q[-2], 1e-9) * (x - q[-1])
    else:
        lp = float(np.interp(x, q, logS))                      # in between: interpolate
    return (x - mean) / sd, max(lp, -50.0)                     # z-score, log tail probability (floored)


# ----------------------------------------------------------------------------- 1f stratum transforms
def _sine_extrap(z, npre):                                     # sinusoid fitted on the pre segment, extended into the post segment
    zp, n = z[:npre], len(z); om = _sine_freq(zp)              # pre part and its own refined frequency
    X = lambda t: np.column_stack([np.ones(len(t)), np.cos(om * t), np.sin(om * t)])   # constant + wave columns at times t
    tp, tq = np.arange(npre), np.arange(npre, n)               # pre / post time indices
    b, ep = _wls(X(tp), zp)                                    # pre-segment fit and residuals
    eq = z[npre:] - X(tq) @ b                                  # post residuals around the extrapolated wave
    sp = max(float(np.std(ep)), 1e-9)                          # pre residual noise level
    bq = _wls(X(tq), z[npre:])[0]                              # refit the wave on the post segment (same frequency)
    ac = lambda v: float(np.corrcoef(v[:-1], v[1:])[0, 1]) if np.std(v) > 1e-12 else 0.0   # lag-1 autocorrelation
    amp_p, amp_q = np.hypot(b[1], b[2]), np.hypot(bq[1], bq[2])   # pre and post wave amplitudes
    return {"tf_s_lvl": float(eq.mean() / (sp / np.sqrt(len(eq)))),   # post residual mean in standard errors
            "tf_s_scl": float(np.log(max(np.std(eq), 1e-9) / sp)),     # post / pre residual spread (log ratio)
            "tf_s_rms": float(np.log(max(np.mean(eq ** 2), 1e-18) / sp ** 2)),   # post misfit incl. level (log ratio)
            "tf_s_dep": ac(eq) - ac(ep),                       # change in residual autocorrelation
            "tf_s_amp": float(np.log((amp_q + 1e-6) / (amp_p + 1e-6))),   # change in wave amplitude (log ratio)
            "tf_s_phs": float(np.sin(np.arctan2(bq[2], bq[1]) - np.arctan2(b[2], b[1])))}   # change in wave phase


# ----------------------------------------------------------------------------- family
def _all_fits(z, npre):                                        # fit all 7 x 4 no-break models
    om = _sine_freq(z); fits = {}                              # shared sinusoid frequency; results by (model, noise)
    for m in MEANS:
        base = _fit_mean(m, z, npre, om)                       # least-squares mean fit
        for i in INNOVS:
            fits[(m, i)] = _fit_innov(base, i)                 # add each noise distribution
    return fits


def family_stats(z, npre, m, innov):                           # statistics of ONE model (used by the calibration simulator)
    om = _sine_freq(z) if m == "sine" else 0.0                 # frequency only needed for the sinusoid model
    f = _fit_innov(_fit_mean(m, z, npre, om), innov)           # fit it
    st = _score_stats(f)                                       # score tests
    out = {k: st[k] for k in ("LM_full", "LM_ls", "LM_ds")}   # combined statistics
    out.update({f"zsq_{c}": st["z"].get(c, 0.0) ** 2 for c in ("lvl", "scl", "dep", "tail")})   # squared single-direction scores
    out["GLR_joint"] = _glr(f)["joint"] if innov in ("gauss", "t") else np.nan   # joint likelihood ratio (Gaussian / t only)
    return out, f


def fam_glr(c):
    z, npre = c.z, c.npre; n = len(z)                          # robust-scaled series and boundary
    if n < 40 or npre < 12 or n - npre < 12 or np.std(z) < 1e-9:   # too short or constant: no features (filled with 0 later)
        return {}
    fits = _all_fits(z, npre)                                  # all candidate no-break models
    keys = list(fits)                                          # (model, noise) pairs
    bic = np.array([fits[k]["bic"] for k in keys])             # their BIC values
    wts = np.exp(-(bic - bic.min()) / 2); wts /= wts.sum()     # BIC weights = approximate posterior model probabilities
    best = keys[int(np.argmin(bic))]                           # best model overall
    ST = {k: _score_stats(fits[k]) for k in keys}              # score tests under every model
    gkeys = [k for k in keys if k[1] in ("gauss", "t")]        # models for which likelihood ratios are computed
    GL = {k: _glr(fits[k]) for k in gkeys}                     # likelihood ratios under those models
    gbest = min(gkeys, key=lambda k: fits[k]["bic"])           # best Gaussian / t model
    o = {}
    sb = ST[best]                                              # 1b: score tests under the best model
    for cc in COORDS:
        zc = sb["z"].get(cc, 0.0); o[f"sc_z_{cc}"] = zc; o[f"sc_zsq_{cc}"] = zc * zc   # signed and squared (|z| splits identically to z^2)
    for k in ("LM_full", "LM_ls", "LM_ds"):
        o[f"sc_{k}"] = sb[k]                                   # combined score statistics
    for k, v in GL[gbest].items():
        o[f"lr_{k}"] = v                                       # 1c: likelihood ratios under the best Gaussian / t model
    for cc in COORDS:                                          # 1d: BIC-weighted averages over models that have this direction
        has = np.array([cc in ST[k]["z"] for k in keys]); ww = wts * has
        zz = np.array([ST[k]["z"].get(cc, 0.0) for k in keys])
        o[f"ma_z_{cc}"] = float(np.sum(ww * zz) / ww.sum()) if ww.sum() > 0 else 0.0
        o[f"ma_zsq_{cc}"] = float(np.sum(ww * zz * zz) / ww.sum()) if ww.sum() > 0 else 0.0
    for k in ("LM_full", "LM_ls", "LM_ds"):
        o[f"ma_{k}"] = float(sum(wts[j] * ST[keys[j]][k] for j in range(len(keys))))
    gw = np.array([wts[keys.index(k)] for k in gkeys]); gw = gw / gw.sum() if gw.sum() > 0 else np.ones(len(gkeys)) / len(gkeys)
    for k in ("lvl", "slp", "dep", "sin", "scl", "joint"):
        o[f"ma_lr_{k}"] = float(sum(gw[j] * GL[gkeys[j]][k] for j in range(len(gkeys))))
    o["ma_entropy"] = float(-np.sum(wts * np.log(wts + 1e-300)))   # how spread the model weights are (0 = one clear winner)
    o["ma_topw"] = float(wts.max())                            # weight of the best model
    o["ma_best_mean"] = float(MEANS.index(best[0])); o["ma_best_innov"] = float(INNOVS.index(best[1]))   # which model won
    phi = fits[(best[0], best[1])]["b"][1] if best[0] in ("ar1", "arma11") else 0.0   # AR(1) persistence for the table lookup
    row = _cell_summary(best[0], best[1], phi, npre, n - npre)   # 1e: null summaries for the best model at these lengths
    vals = {"LM_full": sb["LM_full"], "LM_ls": sb["LM_ls"], "LM_ds": sb["LM_ds"],
            **{f"zsq_{cc}": sb["z"].get(cc, 0.0) ** 2 for cc in ("lvl", "scl", "dep", "tail")}}
    for j, k in enumerate(CAL_STATS[:-1]):
        o[f"cal_z_{k}"], o[f"cal_lp_{k}"] = _calibrate(vals[k], row[j])
    gphi = fits[gbest]["b"][1] if gbest[0] in ("ar1", "arma11") else 0.0
    o["cal_z_GLR_joint"], o["cal_lp_GLR_joint"] = _calibrate(GL[gbest]["joint"], _cell_summary(gbest[0], gbest[1], gphi, npre, n - npre)[-1])
    d = np.diff(z); dbest, dbic = None, np.inf                 # 1f: first differences (random-walk / near-unit-root view)
    for m in ("iid", "ar1"):
        bse = _fit_mean(m, d, npre - 1, 0.0)
        for i in ("gauss", "t"):
            fd = _fit_innov(bse, i)
            if fd["bic"] < dbic:
                dbest, dbic = fd, fd["bic"]                    # keep the better model of the differences
    sd_ = _score_stats(dbest)
    for cc in ("lvl", "slp", "dep", "scl", "tail", "skew"):
        o[f"tf_d_z_{cc}"] = sd_["z"].get(cc, 0.0)
    o["tf_d_LM"] = sd_["LM_full"]
    b2 = fits[("ar2", "gauss")]["b"]                           # AR(2) coefficients of the whole series
    o["tf_rho"] = float(np.max(np.abs(np.roots([1.0, -b2[1], -b2[2]]))))   # spectral radius (near 1 = near unit root)
    o["tf_nur"] = float(o["tf_rho"] > 0.98 or best[0] == "rw")   # flag: random-walk-like series
    o.update(_sine_extrap(z, npre))                            # sinusoid extrapolated from the pre segment
    sg = ST[(best[0], "garch")]                                # GARCH-filtered: tests on volatility-standardised residuals
    for cc in ("lvl", "scl", "dep", "tail", "skew"):
        o[f"tf_g_z_{cc}"] = sg["z"].get(cc, 0.0)
    o["tf_g_LM"] = sg["LM_full"]
    pq = _garch_par(fits[(best[0], "garch")]["shape"]["q"]); o["tf_g_pers"] = float(pq[1] + pq[2])   # volatility persistence a+b
    for cc in ("lvl", "slp", "dep", "scl", "skew", "s_amp", "s_frq"):   # aux (S2 only): one-step shift estimates and SEs
        o[f"aux_d_{cc}"] = sb["d"].get(cc, 0.0); o[f"aux_se_{cc}"] = sb["se"].get(cc, 0.0)
    st_t = ST[(best[0], "t")]; st_m = ST[(best[0], "mix")]     # tail shift from the t model, mixture-share shift from the mixture model
    o["aux_d_tail"], o["aux_se_tail"] = st_t["d"]["tail"], st_t["se"]["tail"]
    o["aux_d_mixw"], o["aux_se_mixw"] = st_m["d"]["tail"], st_m["se"]["tail"]
    o["aux_nu"] = fits[(best[0], "t")]["shape"]["nu"]          # fitted tail parameter
    return {k: float(np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)) for k, v in o.items()}   # always finite


fam_glr._hash_module = True                                    # cache key covers the whole module (helpers included)
FAMILIES.update({"glr": fam_glr})                              # register with the feature extractor


# ----------------------------------------------------------------------------- calibration builder (offline)
def _simulate(m, innov, par, npre, npost, rng):                # one no-break series from a known model
    n, burn = npre + npost, 200                                # length and warm-up
    N = n + burn
    if innov == "gauss":
        e = rng.standard_normal(N)                             # normal noise
    elif innov == "t":
        e = rng.standard_t(6, N) / np.sqrt(1.5)                # t(6) noise rescaled to unit variance
    elif innov == "mix":
        e = rng.standard_normal(N) * np.where(rng.random(N) < 0.1, 3.0, 1.0)   # 10% of points 3x wider
    else:                                                      # GARCH(1,1) noise, omega .2, a .18, b .62
        u = rng.standard_normal(N); e = np.empty(N); v = 1.0
        for t in range(N):
            e[t] = np.sqrt(v) * u[t]; v = 0.2 + 0.18 * e[t] ** 2 + 0.62 * v
    t = np.arange(n)
    if m == "iid":
        x = e[burn:]
    elif m == "ar1":
        x = lfilter([1.0], [1.0, -par], e)[burn:]              # AR(1) with persistence par
    elif m == "ar2":
        x = lfilter([1.0], [1.0, -1.2, 0.5], e)[burn:]         # quasi-cyclic AR(2)
    elif m == "arma11":
        x = lfilter([1.0, 0.3], [1.0, -0.6], e)[burn:]         # ARMA(1,1)
    elif m == "trend":
        x = 2.0 * (t - n / 2) / n + e[burn:]                   # straight-line trend
    elif m == "rw":
        x = np.cumsum(e[burn:])                                # random walk
    else:
        x = 1.5 * np.sin(2 * np.pi * t / 20 + rng.uniform(0, 2 * np.pi)) + e[burn:]   # sinusoid, period 20
    return x


def _cell_job(ci, cell, i, j, B, seed):                        # simulate B no-break series for one table cell and one length pair
    m, innov, par = cell; a, bq = int(CAL_N[i]), int(CAL_N[j])  # model and the (pre, post) lengths
    rng = np.random.default_rng([seed, ci, i, j])              # independent, reproducible random stream per cell and length pair
    V = np.zeros((B, len(CAL_STATS)))                          # statistics of each simulated series
    for b in range(B):
        x = _simulate(m, innov, par, a, bq, rng)               # a series with NO break
        st, _ = family_stats(feats.robust_scale(x, a), a, m, innov)   # same scaling + statistics as the real features
        V[b] = [st[k] for k in CAL_STATS]
    out = np.full((len(CAL_STATS), 2 + len(CAL_P)), np.nan)    # summaries per statistic (NaN = undefined for this model)
    for s in range(len(CAL_STATS)):
        v = V[:, s]
        if not np.all(np.isnan(v)):
            out[s] = np.r_[np.nanmean(v), np.nanstd(v), np.nanquantile(v, CAL_P)]   # mean, SD, quantiles under "no break"
    return ci, i, j, out


if __name__ == "__main__":
    from joblib import Parallel, delayed                       # run cells in parallel worker processes
    import time
    cmd = sys.argv[1]
    if cmd == "calib":                                         # build the lookup table
        B = int(os.environ.get("CAL_B", "2000")); t0 = time.time()
        jobs = [(ci, c, i, j) for ci, c in enumerate(CAL_FAMS) for i in range(len(CAL_N)) for j in range(len(CAL_N))]   # one job per cell and length pair
        res = Parallel(n_jobs=int(os.environ.get("FEAT_JOBS", "11")), batch_size=1)(delayed(_cell_job)(ci, c, i, j, B, CAL_SEED) for ci, c, i, j in jobs)
        summ = np.full((len(CAL_FAMS), len(CAL_N), len(CAL_N), len(CAL_STATS), 2 + len(CAL_P)), np.nan)   # [cell, n_pre, n_post, stat, summaries]
        for ci, i, j, o in res:
            summ[ci, i, j] = o                                 # place each job's summaries
        gi = {m: CAL_FAMS.index((m, "gauss", 0.0)) for m in ("iid",)}   # GLR is undefined for GARCH / mixture: use the iid Gaussian cell
        for ci, (m, innov, _) in enumerate(CAL_FAMS):
            if innov in ("garch", "mix"):
                summ[ci, :, :, -1] = summ[gi["iid"], :, :, -1]
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CALIB_PATH, summ=summ, fams=np.array([(m, i, str(p)) for m, i, p in CAL_FAMS]), n=CAL_N, p=CAL_P, stats=np.array(CAL_STATS))
        print(f"[calib] {len(CAL_FAMS)} cells x {len(CAL_N)**2} length pairs x B={B} in {time.time()-t0:.0f}s; md5 {hashlib.md5(CALIB_PATH.read_bytes()).hexdigest()}")
    elif cmd == "sizecheck":                                   # do calibrated tests reject 5% of no-break series at the 5% level?
        B = int(os.environ.get("SC_B", "1000")); rows = []
        checks = [("iid", "gauss", 0.0, 113, 453), ("iid", "t", 0.0, 226, 226), ("ar1", "gauss", 0.45, 453, 113), ("ar1", "t", 0.75, 113, 113),
                  ("ar2", "gauss", 0.0, 226, 453), ("trend", "gauss", 0.0, 453, 226), ("rw", "gauss", 0.0, 226, 226), ("sine", "gauss", 0.0, 113, 226),
                  ("iid", "garch", 0.0, 226, 226), ("arma11", "t", 0.0, 453, 453)]   # OFF-grid lengths / parameters (tests interpolation)

        def job(k, m, innov, par, a, bq):                      # rejection rates for one off-grid cell
            rng = np.random.default_rng([CAL_SEED + 1, k]); rej = np.zeros(len(CAL_STATS)); cnt = np.zeros(len(CAL_STATS))
            for _ in range(B):
                st, f = family_stats(feats.robust_scale(_simulate(m, innov, par, a, bq, rng), a), a, m, innov)
                phi = f["b"][1] if m in ("ar1", "arma11") else 0.0
                row = _cell_summary(m, innov, phi, a, bq)      # interpolated null summaries
                for s, kk in enumerate(CAL_STATS):
                    if np.isfinite(st[kk]):
                        cnt[s] += 1; rej[s] += np.exp(_calibrate(st[kk], row[s])[1]) < 0.05   # calibrated p-value below 5%?
            return dict(cell=f"{m}/{innov}/{par}/{a}/{bq}", **{kk: round(float(rej[s] / max(cnt[s], 1)), 4) for s, kk in enumerate(CAL_STATS)})
        rows = Parallel(n_jobs=10)(delayed(job)(k, *c) for k, c in enumerate(checks))
        worst = max(((r["cell"], kk, r[kk]) for r in rows for kk in CAL_STATS if kk in r), key=lambda x: abs(x[2] - 0.05))
        out = {"B": B, "rows": rows, "worst": worst}
        (__import__("paths").CACHE / "calib_sizecheck.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    elif cmd == "profile":                                     # time the family on 500 real series
        import extract, h2
        ids, ser = extract.load_series(__import__("paths").DATA / "train_series.parquet")
        sel = np.random.default_rng(0).choice(len(ids), 500, replace=False)
        t0 = time.time()
        rows = Parallel(n_jobs=11)(delayed(lambda x, npre: fam_glr(feats.Ctx(x, npre)))(ser[i][0], ser[i][1]) for i in sel)
        dt = time.time() - t0
        print(f"[profile] 500 series in {dt:.1f}s with 11 workers; {len(rows[0])} columns; projected train+test x orig+rev (48,000): {dt/500*48000/60:.1f} min")
