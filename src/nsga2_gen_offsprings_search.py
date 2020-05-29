import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load

from utils import Pickler

from nsga2.venus_attack import attack
from nsga2 import venus_constraints
from nsga2.venus_encoder import VenusEncoder

logging.basicConfig(level=logging.DEBUG)

# ----- PARAMETERS

input_dir = "../out/target_model"
output_dir = "../out/venus_attacks/nsga2_gen_offspring_search"
seed = 0
threshold = 0.24
n_jobs = -1
budget = 200000
list_n_offsprings = [10, 20, 40, 80, 160, 320, 640]

n_initial_state = 1000
n_repetition = 1
offset = 6

# ----- CONSTANTS

model_file = "/model.joblib"
X_attack_candidate_file = "/X_attack_candidate.npy"
scaler_file = "/scaler.pickle"
out_columns = [
    "n_gen",
    "pop_size",
    "n_offsprings",
    "objective_1",
    "objective_2",
    "objective_3",
    "objective_4",
]

# ----- Load and create necessary objects

list_n_generation = [int(budget / i) for i in list_n_offsprings]
list_pop_size = [2 * i for i in list_n_offsprings]

model = load(input_dir + model_file)
model.set_params(verbose=0, n_jobs=1)
X_initial_states = np.load(input_dir + X_attack_candidate_file)
scaler = Pickler.load_from_file(input_dir + scaler_file)
encoder = VenusEncoder()
X_initial_states = X_initial_states[:n_initial_state]

# ----- Check if constraints are satisfied
constraints = venus_constraints.evaluate(X_initial_states)
constraints_violated = (constraints > 0).sum()
if constraints_violated > 0:
    logging.error("Constraints violated {} time(s).".format(constraints_violated))
    exit(1)

# Set random seed
random.seed(seed)
np.random.seed(seed)

Path(output_dir).mkdir(parents=True, exist_ok=True)

# Copy the initial states n_repetition times
X_initial_states = np.repeat(X_initial_states, n_repetition, axis=0)

if __name__ == "__main__":

    for i in range(offset, len(list_n_offsprings)):
        n_generation = list_n_generation[i]
        pop_size = list_pop_size[i]
        n_offsprings = list_n_offsprings[i]

        logging.info(
            "Parameters: {} {} {} ({}/{})".format(
                n_generation, pop_size, n_offsprings, i + 1, len(list_n_offsprings)
            )
        )

        # Initial state loop (threaded)
        parameter_objectives = Parallel(n_jobs=n_jobs)(
            delayed(attack)(
                index,
                initial_state,
                model,
                scaler,
                encoder,
                n_generation,
                n_offsprings,
                pop_size,
                threshold,
            )
            for index, initial_state in enumerate(X_initial_states)
        )

        # Calculate success rate
        parameter_objectives = np.array(parameter_objectives)
        success_rate = np.apply_along_axis(
            lambda x: x.sum() / x.shape[0], 0, parameter_objectives
        )

        # Save results
        result = np.concatenate(
            (np.array([n_generation, pop_size, n_offsprings]), success_rate)
        )
        results_df = pd.DataFrame(result.reshape(1, -1), columns=out_columns)
        results_df.to_csv(
            output_dir + "/parameters_{}.csv".format(offset + i), index=False
        )
