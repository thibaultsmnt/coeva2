import warnings
import random


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import logging
from joblib import dump
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.meta import AdaptiveRandomForest

from utils import Pickler
from online_attack import venus_constraints
from utils import Datafilter

logging.basicConfig(level=logging.DEBUG)

# ----- PARAMETERS

output_dir = "../out/target_online_model2"
dataset_path = "../data/lcld/lcld_venus_sorted.csv"
seed = 0

# ----- CONSTANT

model_file_prefix = "/model"
model_file_extension = ".joblib"
X_attack_candidate_file = "/X_attack_candidate_{}.npy"
scaler_file = "/scaler.pickle"
mcc_file = "/mcc.csv"

model_parameters = {
    "n_estimators": 125,
    "min_samples_split": 6,
    "min_samples_leaf": 2,
    "max_depth": 10,
    "bootstrap": True,
}
threshold = 0.24


Path(output_dir).mkdir(parents=True, exist_ok=True)

# Set random seed
random.seed(seed)
np.random.seed(seed)

# ----- GET DATA

data = pd.read_csv(dataset_path)
data = data.sample(frac=0.1)

# ----- DEFINE, TRAIN AND SAVE CLASSIFIER

model = AdaptiveRandomForest()

months = data["issue_d"].unique()
months = np.sort(months)
classes = np.array([0, 1])

mccs = []
train_months = months[: (len(months) - 1)]


def get_data_by_month(a_month):
    month_df = data[data["issue_d"] == a_month]
    a_y = month_df.pop("charged_off").to_numpy()
    a_X = month_df.to_numpy()
    return a_X, a_y


for index, month in np.ndenumerate(train_months):

    # Training

    X_train, y_train = get_data_by_month(month)
    logging.debug("Fitting month {} ({}).".format(month, len(y_train)))
    model.partial_fit(X_train, y_train, classes=classes)
    dump(
        model,
        "{}{}_{}{}".format(output_dir, model_file_prefix, month, model_file_extension),
    )

    # Testing

    X_test, y_test = get_data_by_month(months[index[0] + 1])
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)

    # Evaluating

    mcc = matthews_corrcoef(y_test, y_pred)
    logging.debug("Mcc month +1: {}.".format(mcc))
    mccs.append(mcc)

    # Save target

    X_test, y_test, y_pred = Datafilter.filter_correct_prediction(
        X_test, y_test, y_pred
    )
    X_test, _, _ = Datafilter.filter_by_target_class(X_test, y_test, y_pred, 1)
    X_test = X_test[np.random.permutation(X_test.shape[0])]
    constraints = venus_constraints.evaluate(X_test)
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    X_test = X_test[(1 - constraints_violated).astype(bool)]
    np.save(output_dir + X_attack_candidate_file.format(month), X_test)


mccs = np.array(mccs)
month_mcc = np.transpose(np.array([train_months, mccs]))
month_mcc = pd.DataFrame(month_mcc, columns=["month", "mcc"])
month_mcc.to_csv(output_dir + mcc_file)

# ----- Create and save min max scaler
y = data.pop("charged_off").to_numpy()
X = data.to_numpy()
scaler = MinMaxScaler()
scaler.fit(X)
Pickler.save_to_file(scaler, output_dir + scaler_file)
