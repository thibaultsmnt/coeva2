import warnings

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

logging.basicConfig(level=logging.DEBUG)

# ----- PARAMETERS

output_dir = "../out/target_online_model"
dataset_path = "../data/lcld/lcld_venus_sorted.csv"

# ----- CONSTANT

model_file_prefix = "/model"
model_file_extension = ".joblib"
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

# ----- GET DATA

data = pd.read_csv(dataset_path)
data = data.sample(frac=0.01)

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
    X, y = get_data_by_month(month)
    logging.debug("Fitting month {} ({}).".format(month, len(y)))
    model.partial_fit(X, y, classes=classes)
    dump(
        model,
        "{}{}_{}{}".format(output_dir, model_file_prefix, month, model_file_extension),
    )

    X_1, y_1 = get_data_by_month(months[index[0] + 1])
    y_pred_proba = model.predict_proba(X_1)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)
    mcc = matthews_corrcoef(y_1, y_pred)
    logging.debug("Mcc month +1: {}.".format(mcc))
    mccs.append(mcc)


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
