"""Central place for hard-coded parameters: file paths and model settings.

Paths are built relative to the repository root, so the code works no matter
which directory it is run from or which machine it is run on.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# --- Train / validation split -------------------------------------------
VALID_SIZE = 0.2
RANDOM_STATE = 42

# --- Model ---------------------------------------------------------------
PREDICTORS = ["Sex", "Pclass"]
TARGET = "Survived"
RF_PARAMS = {
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "criterion": "gini",
    "n_estimators": 100,
    "verbose": False,
}

# --- Plotting ------------------------------------------------------------
COLOR_LIST = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]