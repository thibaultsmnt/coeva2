import copy
import multiprocessing
import warnings
import random
import time

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

import logging
from joblib import dump, load, Parallel, delayed
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

output_dir = "../out/target_online_model_parallel"
dataset_path = "../data/lcld/lcld_venus_sorted.csv"
seed = 0
date_col = "issue_d"
offset = 0
n_jobs = -1

# ----- CONSTANT

model_file_prefix = "/model_"
model_file_extension = ".joblib"
y_pred_file_prefix = "/y_pred_"
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
# data = data.sample(frac=0.01)
date_index = data.columns.get_loc(date_col)

# ----- DEFINE, TRAIN AND SAVE CLASSIFIER

model = AdaptiveRandomForest()

months = data["issue_d"].unique()
months = np.sort(months)

mccs = []
if offset > 0:
    model = load(
        "{}{}{}.joblib".format(output_dir, model_file_prefix, train_months[offset - 1])
    )
    months = months[offset:]

train_months = months[: len(months) - 1]


y = data.pop("charged_off").to_numpy()
X = data.to_numpy()
classes = np.unique(y)

if n_jobs == -1:
    n_jobs = multiprocessing.cpu_count()

for index, month in enumerate(train_months):

    # Training
    t0 = time.clock()

    train_index = X[:, date_index] == month
    X_train = X[train_index]
    y_train = y[train_index]

    logging.debug("Fitting month {} ({}).".format(month, y_train.shape[0]))

    model.partial_fit(X_train, y_train, classes=classes)
    dump(
        model,
        "{}{}{}{}".format(output_dir, model_file_prefix, month, model_file_extension),
    )

    logging.debug("Time: {}s.".format(time.clock() - t0))

    # Testing with the entire dataset
    t0 = time.clock()
    logging.debug("Testing dataset.")

    splits = np.array_split(X, n_jobs)
    model_copies = [copy.deepcopy(model) for i in range(n_jobs)]
    y_pred_proba = Parallel(n_jobs=n_jobs)(
        delayed(model_copies[i].predict_proba)(x) for i, x in enumerate(splits)
    )
    y_pred_proba = np.concatenate(y_pred_proba)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)
    np.save("{}{}{}.npy".format(output_dir, y_pred_file_prefix, month), y_pred)

    logging.debug("Time: {}s.".format(time.clock() - t0))

    # Evaluating with month +1
    t0 = time.clock()
    logging.debug("Evaluating t+1.")

    test_index = X[:, date_index] == months[index + 1]
    y_test = y[test_index]
    y_test_pred = y_pred[test_index]
    mcc = matthews_corrcoef(y_test, y_test_pred)
    logging.debug("Mcc month +1: {}.".format(mcc))
    mccs.append(mcc)

    logging.debug("Time: {}s.".format(time.clock() - t0))


mccs = np.array(mccs)
month_mcc = np.transpose(np.array([train_months, mccs]))
month_mcc = pd.DataFrame(month_mcc, columns=["month", "mcc"])
month_mcc.to_csv(output_dir + mcc_file)
