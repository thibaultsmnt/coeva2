import numpy as np
import pandas as pd
from joblib import load
from utils import Pickler
from utils import Datafilter

# ----- PARAMETERS
from sklearn.preprocessing import MinMaxScaler

output_dir = "../out/target_online_model2"
dataset_path = "../data/lcld/lcld_venus_sorted.csv"
threshold = 0.24

# ----- CONSTANT

model_file_prefix = "/model"
model_file_extension = ".joblib"
X_attack_candidate_file = "/X_attack_candidate_{}.npy"
scaler_file = "/scaler.pickle"
mcc_file = "/mcc.csv"

data = pd.read_csv(dataset_path)

months = data["issue_d"].unique()

train_months = months[: (len(months) - 1)]


def get_data_by_month(a_month):
    month_df = data[data["issue_d"] == a_month]
    a_y = month_df.pop("charged_off").to_numpy()
    a_X = month_df.to_numpy()
    return a_X, a_y


for index, month in np.ndenumerate(train_months):

    # Load model
    model = load(
        "{}{}_{}{}".format(output_dir, model_file_prefix, month, model_file_extension)
    )

    # Testing

    X_test, y_test = get_data_by_month(months[index[0] + 1])
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)

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

# ----- Create and save min max scaler
y = data.pop("charged_off").to_numpy()
X = data.to_numpy()
scaler = MinMaxScaler()
scaler.fit(X)
Pickler.save_to_file(scaler, output_dir + scaler_file)
