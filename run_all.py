"""One command for the KU-SBB v6 submission. Raw data is read from the folder named in config/settings.json
("data_dir", default ../data); everything else stays inside this folder unless --out names a scratch work folder.
    python run_all.py --verify-only        saved models -> must reproduce submission.csv byte for byte from the raw TEST
                                           series (test features are computed and cached first), then src/infer.py on
                                           50 raw test series must agree with it
    python run_all.py [--out DIR]          full rebuild from raw data: train + test features, 10-seed refit on all 24,000
                                           labelled series, candidate predictions, robustness suite, and an exact
                                           comparison of the candidate with the shipped submission.csv. Everything is
                                           written to DIR (default: cache/); the shipped models/ and submission.csv are
                                           NOT touched unless --ship is also given
    python run_all.py --reduced [--out DIR] the same complete pipeline on a small stratified training subsample with 2
                                           seeds (features, folds, cross-validated training, refit, prediction, output),
                                           as a fast end-to-end proof; writes only to DIR"""
import subprocess, sys, os                                     # run the steps as separate Python processes
from pathlib import Path                                       # file paths
R = Path(__file__).resolve().parent                            # this SUBMISSION folder
os.chdir(R)                                                    # all relative paths below are inside it
if "--out" in sys.argv:                                        # a scratch work folder for caches, models and candidate files
    os.environ["KUSBB_WORK_DIR"] = str(Path(sys.argv[sys.argv.index("--out") + 1]).resolve())   # read by src/paths.py in every step
sys.path.insert(0, str(R / "src"))                             # so the location settings in src/paths.py can be read here
import paths                                                   # the single place that knows where the raw data lives
FAMS1 = "wide,shape,pit,vol,battery,distfam,dgplr,null3,glr"   # feature families that need only the raw series
FAMS_ALL = FAMS1 + ",dgplr2"                                   # dgplr2 also needs the frozen prior in config/
PINNED = {"numpy": "2.3.5", "pandas": "2.3.3", "scipy": "1.16.3", "scikit-learn": "1.8.0", "lightgbm": "4.7.0",
          "joblib": "1.5.3", "pyarrow": "26.0.0"}               # the versions the byte-identical reproduction was verified on


def run(*a):                                                   # run one step and stop on the first failure
    print(">>", " ".join(a), flush=True)                       # show what is being run
    r = subprocess.run([sys.executable, "-W", "ignore", *a])   # same Python interpreter, warnings hidden
    if r.returncode:                                           # a failed step ends the run with a one-line reason
        sys.exit(f"[run_all] step failed (exit code {r.returncode}): {' '.join(a)[:120]}")


# --- failure mode 1: package versions differ from the pinned set (warn and continue) ----------------------------------
from importlib.metadata import version, PackageNotFoundError   # installed package versions
for pkg, want in PINNED.items():
    try:
        have = version(pkg)                                    # the installed version
    except PackageNotFoundError:
        sys.exit(f"[run_all] required package not installed: {pkg} (pip install -r requirements.txt)")
    if have != want:
        print(f"[run_all] WARNING: {pkg} {have} installed, {want} pinned - results should rank identically, but the last "
              f"digits of predictions may differ, so byte-identity is not guaranteed (see REPORT.md, section 4, 'Version lock')", flush=True)
if sys.version_info[:2] != (3, 13):
    print(f"[run_all] WARNING: Python {sys.version.split()[0]} in use, verified on 3.13.14 (see REPORT.md, section 4, 'Version lock')", flush=True)

# --- failure mode 2: competition data not where config/settings.json says -----------------------------------------
need = ["test_series.parquet", "sample_submission.csv"] + ([] if "--verify-only" in sys.argv else ["train_series.parquet", "train_labels.csv"])
missing = [f for f in need if not (paths.DATA / f).exists()]   # which of the required competition files are absent
if missing:
    sys.exit(f"[run_all] competition data not found: {', '.join(missing)} missing in {paths.DATA} "
             f"- put the competition files there or set \"data_dir\" in config/settings.json")
if "--verify-only" in sys.argv:                                # cheap check, no training: only the 4,000 test series are needed
    run("src/extract.py", "--fams", FAMS_ALL, "--files", "test_series.parquet")   # original-view test features (cached)
    run("src/views.py", "--views", "rev", "--fams", FAMS_ALL, "--stems", "test_series")   # time-reversed-view test features
    run("src/final6.py", "verify")                            # saved models must reproduce submission.csv exactly
    run("-c", "import sys;sys.path.insert(0,'src');import pandas as pd, paths;from infer import predict_frame;"
              "te=pd.read_parquet(paths.DATA/'test_series.parquet');ids=te.id.drop_duplicates().iloc[:50];p=predict_frame(te[te.id.isin(ids)]);"
              "s=pd.read_csv('submission.csv',dtype={'id':str},float_precision='round_trip').set_index('id').break_probability;"
              "d=float((p-s.loc[p.index]).abs().max());print('[verify] infer.py on 50 raw test series: max |diff| vs submission',d);"
              "sys.exit(0 if d<1e-12 else '[verify] infer.py disagrees with submission.csv by '+str(d)+' (see REPORT.md, section 4, Version lock)')")
    sys.exit(0)                                                # verification done
if "--reduced" in sys.argv:                                    # fast end-to-end proof of the complete pipeline
    run("src/final6.py", "reduced", os.environ.get("KUSBB_REDUCED_N", "600"), os.environ.get("KUSBB_REDUCED_TEST", "100"))
    sys.exit(0)
if not (paths.CACHE / "dev_ids.csv").exists():                 # the fixed 80/20 dev / holdout split (seed 20261103), needed by s2_deconv.py
    run("-c", "import sys;sys.path.insert(0,'src');import paths, pandas as pd;from sklearn.model_selection import StratifiedShuffleSplit as S;"
              "l=pd.read_csv(paths.DATA/'train_labels.csv',dtype={'id':str});d,h=next(S(1,test_size=0.2,random_state=20261103).split(l.id,l.label));"
              "l.iloc[h][['id']].to_csv(paths.CACHE/'holdout_ids.csv',index=False);l.iloc[d][['id']].to_csv(paths.CACHE/'dev_ids.csv',index=False)")
if not (R / "config/calib_table.npz").exists():                # glr calibration lookup table (fixed seed; feats7.py checks its md5)
    run("src/feats7.py", "calib")
run("src/extract.py", "--fams", FAMS1)                         # original-view features, train and test
if not (R / "config/dgplr2_priors.json").exists():             # deconvolved dgplr2 prior from training positives (feats8.py checks its md5)
    run("src/s2_deconv.py", "fit")
run("src/extract.py", "--fams", "dgplr2")                      # dgplr2 needs the prior above
run("src/views.py", "--views", "rev", "--fams", FAMS_ALL)      # time-reversed-view features, train and test
ship = "--ship" in sys.argv                                    # only an explicit --ship may overwrite the shipped artifacts
run("src/final6.py", "refit", "v6", "10", *(["--ship"] if ship else []))   # train on all 24,000 series with 10 seeds
md = paths.MODELS if ship else paths.CACHE / "models_v6"       # where the refit saved its boosters
cand = R / "submission.csv" if ship else paths.CACHE / "candidate_v6.csv"   # and its predictions
os.environ["ROBUST_SUB"] = str(cand); os.environ["ROBUST_OUT"] = str(paths.CACHE / "robustness.json")   # robustness suite checks the NEW models
run("src/robust3.py", str(md))                                 # full robustness suite on the refitted models
run("-c", "import hashlib;from decimal import Decimal as D;"   # exact comparison of the rebuilt predictions with the shipped file
          f"a=open(r'{cand}').read().splitlines()[1:];b=open('submission.csv').read().splitlines()[1:];"
          "A={l.split(',')[0]:D(l.split(',')[1]) for l in a};B={l.split(',')[0]:D(l.split(',')[1]) for l in b};"
          "d=max(abs(A[k]-B[k]) for k in B);n=sum(A[k]!=B[k] for k in B);"
          f"print('[rebuild] candidate md5',hashlib.md5(open(r'{cand}','rb').read()).hexdigest(),'| shipped md5',hashlib.md5(open('submission.csv','rb').read()).hexdigest());"
          "print('[rebuild] values differing:',n,'of',len(B),'| max |diff|:',d)")
