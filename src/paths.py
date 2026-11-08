"""Where every file lives. This is the ONLY place that decides locations.
The raw competition data folder is the single setting "data_dir" in config/settings.json (default "../data", i.e. a
folder named data next to this SUBMISSION folder). A relative value is read relative to this SUBMISSION folder; an
absolute value is used as given. Everything else is read from, or written to, this folder:
config/ (frozen settings), models/ (trained boosters), cache/ (feature tables, created on first run; a different
work folder can be chosen with run_all.py --out, which sets the environment variable KUSBB_WORK_DIR)."""
import json                                                    # reads the settings file
from pathlib import Path                                       # file-path handling
ROOT = Path(__file__).resolve().parent.parent                  # the SUBMISSION folder (the parent of src/)
CONFIG = ROOT / "config"                                       # frozen settings: column list, LightGBM parameters, generator config, tables
_SETTINGS = json.loads((CONFIG / "settings.json").read_text())  # the one user-editable setting (where the raw data is)
_D = Path(_SETTINGS["data_dir"])                               # the configured data folder, as written in the settings file
DATA = _D if _D.is_absolute() else (ROOT / _D).resolve()       # relative values are taken relative to the SUBMISSION folder
MODELS = ROOT / "models"                                       # the 10 trained LightGBM boosters and their metadata
import os                                                      # environment variables
CACHE = Path(os.environ.get("KUSBB_WORK_DIR", ROOT / "cache"))  # feature tables computed from the raw data (not shipped; rebuilt on demand); run_all.py --out points this at a scratch folder
(CACHE / "views").mkdir(parents=True, exist_ok=True)           # create cache/ and cache/views/ on first use so the extractors can write there
