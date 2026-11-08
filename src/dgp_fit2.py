"""v4 P1e iteration 2: sinusoid + AR(1)-noise model per segment (refined periodogram frequency, harmonic regression,
Gaussian conditional ML), BIC-compared with the best P1a mean model; also outlier counts.
  python src/dgp_fit2.py -> work/v3/dgp/fits2_{pre,post}.parquet"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

LOG2PI = np.log(2 * np.pi)


def fit_sine(z):
    n = len(z); t = np.arange(n, dtype=float)
    zc = z - z.mean()
    F = np.abs(np.fft.rfft(zc, 8 * n)) ** 2; F[:8] = 0          # 8x zero-padded periodogram, skip ~DC
    k = int(np.argmax(F)); f0 = k / (8 * n)
    best = None
    for f in f0 + np.linspace(-0.5, 0.5, 11) / (8 * n):
        X = np.column_stack([np.ones(n), np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
        b, *_ = np.linalg.lstsq(X, z, rcond=None); r = z - X @ b; ss = r @ r
        if best is None or ss < best[0]:
            best = (ss, f, b, r)
    ss, f, b, r = best
    # AR(1) on the harmonic residual (conditional, t >= 2 to match P1a's common sample)
    y, x1 = r[2:], r[1:-1]
    phi = float(x1 @ y / max(x1 @ x1, 1e-12)); e = y - phi * x1; m = len(e); s2 = max(e @ e / m, 1e-300)
    ll = -0.5 * m * (LOG2PI + np.log(s2) + 1)
    bic = -2 * ll + 6 * np.log(m)                                     # c, a, b, f, phi, sigma
    amp = float(np.hypot(b[1], b[2]))
    return dict(sine_bic=float(bic), sine_amp=amp, sine_period=float(1 / f) if f > 0 else np.nan, sine_phase=float(np.arctan2(b[2], b[1])),
                sine_phi=phi, sine_sig=float(np.sqrt(s2)), sine_amp_sig=amp / float(np.sqrt(s2)), sine_r2=float(1 - ss / max(zc @ zc, 1e-12)),
                sine_resid_kurt=float(((e - e.mean()) ** 4).mean() / max(e.var() ** 2, 1e-300) - 3))


def _work(items):
    out = []
    for sid, x, npre in items:
        x = np.asarray(x, float); pre = x[:npre]; med = np.median(pre)
        s = max(1.4826 * np.median(np.abs(pre - med)), 0.25 * pre.std()); s = s if s > 1e-12 * (abs(med) + 1) else 1.0
        z = (x - med) / s; r = {}
        for seg, zz in (("pre", z[:npre]), ("post", z[npre:])):
            d = fit_sine(zz); d["id"] = sid
            mu, sd = np.median(zz), 1.4826 * np.median(np.abs(zz - np.median(zz))) + 1e-12
            d["n_out5"] = int((np.abs(zz - mu) / sd > 5).sum()); d["n_out8"] = int((np.abs(zz - mu) / sd > 8).sum())
            r[seg] = d
        out.append(r)
    return out


if __name__ == "__main__":
    from joblib import Parallel, delayed
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(Path(__file__).parent))
    import extract
    D = ROOT / "work" / "v3" / "dgp"
    P = pd.read_parquet(D / "fits_pre.parquet")
    ids, ser = extract.load_series(ROOT / "train_series.parquet"); pos = {k: j for j, k in enumerate(ids)}
    items = [(i, ser[pos[i]][0], ser[pos[i]][1]) for i in P.index]
    t0 = time.time()
    ch = [items[i:i + 100] for i in range(0, len(items), 100)]
    res = [r for part in Parallel(n_jobs=11)(delayed(_work)(c) for c in ch) for r in part]
    for seg in ("pre", "post"):
        pd.DataFrame([r[seg] for r in res]).set_index("id").to_parquet(D / f"fits2_{seg}.parquet")
    print(f"[dgp_fit2] {len(res)} in {time.time()-t0:.0f}s", flush=True)
