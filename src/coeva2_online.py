import copy

import pandas as pd
import numpy as np
from joblib import load
import logging

from pymoo.optimize import minimize
from utils import Pickler
from online_attack.venus_encoder import VenusEncoder
from online_attack.venus_attack_generator import create_attack

logging.basicConfig(level=logging.DEBUG)

# ----- PARAMETERS

input_dir = "../out/target_online_model2"
dataset_path = "../data/lcld/lcld_venus_sorted.csv"
starting_date = 201501
max_time = 2
max_input_frac = 0.01
threshold = 0.24

# ----- CONSTANT

model_file = "/model_{}.joblib"
X_attack_candidate_file = "/X_attack_candidate_{}.npy"
scaler_file = "/scaler.pickle"


def get_data_by_month(a_month):
    month_df = data[data["issue_d"] == a_month]
    a_y = month_df.pop("charged_off").to_numpy()
    a_X = month_df.to_numpy()
    return a_X, a_y


data = pd.read_csv(dataset_path)
months = data["issue_d"].unique()
start_index = np.where(months == starting_date)[0][0]
months = months[start_index : start_index + max_time]
start_month = months[0]

# Load models
squares = [i * i for i in range(10)]
models = [load(input_dir + model_file.format(i)) for i in months]
model = models[0]
scaler = Pickler.load_from_file(input_dir + scaler_file)
encoder = VenusEncoder()

potential_targets = np.load(input_dir + X_attack_candidate_file.format(start_month))
target = None
target_index = 0
target_accepted = False
while not target_accepted:
    target = potential_targets[target_index]
    presume_accepted = True
    for model in models:
        y_pred_proba = model.predict_proba([target])
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)[0]
        if not y_pred:
            presume_accepted = False
            break

    target_accepted = presume_accepted
    target_index += 1

crafted_input = []

success_attack = False

for index, month in np.ndenumerate(months):
    X, y = get_data_by_month(month)
    logging.info(
        "--- Crafting attacks for month {} ({}/{}".format(
            month, index[0] + 1, len(months)
        )
    )
    max_input = int(len(y) * max_input_frac)
    for i in range(max_input):
        logging.info("--- Crafting attacks {}/{}".format(i + 1, max_input))
        model = copy.deepcopy(model)
        scaler = copy.deepcopy(scaler)
        encoder = copy.deepcopy(encoder)
        problem, algorithm, termination = create_attack(
            X[i], model, scaler, encoder, 1, 50, 100, target, threshold
        )
        result = minimize(
            problem, algorithm, termination, verbose=0, save_history=False,
        )
        print(result.CV)
        print(result.F)
