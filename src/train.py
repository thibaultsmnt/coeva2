import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import Pickler
from utils import Datafilter
from joblib import dump

from sklearn.preprocessing import MinMaxScaler
from coeva2 import venus_constraints

# ----- PARAMETERS

OUTPUT_DIR = "../out/target_model"
DATASET_PATH = "../data/lcld/lcld_venus.csv"

# ----- CONSTANT

MODEL_FILE = "/model.joblib"
INITIAL_STATES = "/X_attack_candidate.npy"
SCALER = "/scaler.pickle"

PARAMS = {
    "n_estimators": 125,
    "min_samples_split": 6,
    "min_samples_leaf": 2,
    "max_depth": 10,
    "bootstrap": True,
}
THRESHOLD = 0.24
VERBOSE = 2


def train(dataset_path=DATASET_PATH, output_dir=OUTPUT_DIR, model_file=MODEL_FILE,
    initial_states=INITIAL_STATES,training_parameters=PARAMS,scaler_file=SCALER, threshold=THRESHOLD, verbose=VERBOSE):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ----- GET DATA

    data = pd.read_csv(dataset_path)
    y = data.pop("charged_off").to_numpy()
    X = data.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # ----- DEFINE, TRAIN AND SAVE CLASSIFIER

    model = RandomForestClassifier(**training_parameters, verbose=2, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)
    dump(model, output_dir + model_file)

    # ----- SAVE X correctly rejected loans

    X_test, y_test, y_pred = Datafilter.filter_correct_prediction(X_test, y_test, y_pred)
    X_test, y_test, y_pred = Datafilter.filter_by_target_class(X_test, y_test, y_pred, 1)
    X_test = X_test[np.random.permutation(X_test.shape[0])]

    # ----- REMOVE CONSTRAINTS VIOLATED
    constraints = venus_constraints.evaluate(X_test)
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    X_test = X_test[(1 - constraints_violated).astype(bool)]
    np.save(output_dir + initial_states, X_test)

    # ----- Create and save min max scaler

    scaler = MinMaxScaler()
    scaler.fit(X)
    Pickler.save_to_file(scaler, output_dir + scaler_file)


if __name__ == '__main__':
    train()	