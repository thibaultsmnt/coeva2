import copy, json, sys

import pandas as pd
import numpy as np
from joblib import load
import logging
from pathlib import Path
from pymoo.optimize import minimize
from datetime import datetime

sys.path.append("./")
from src.coeva2.venus_encoder import VenusEncoder
from src.coeva2.venus_attack_generator import init_attack
from src.coeva2.problem_definition import ProblemConstraints, ProblemEvaluation

from src.utils.in_out import json_to_file, save_to_file, pickle_from_file


def run(config_file):

    parameters= {}
    with open(config_file) as f:
        parameters = json.load(f)

    if parameters.get("experiment_path", None) is None:
        raise KeyError()

    
    experiment_id =  int(datetime.timestamp(datetime.now()))

    dataset_path = parameters.get("dataset_path")
    dataset_features = parameters.get("dataset_features")
    experiment_path = parameters.get("experiment_path")
    scaler_file = parameters.get("scaler_file")
    model_file = parameters.get("model_file")
    threshold = parameters.get("model_threshold")
    ga_parameters = parameters.get("ga_parameters")
    dataset_constraints = parameters.get("dataset_constraints")
    max_states = parameters.get("max_states",0)

    data = pd.read_csv(dataset_path)
    model = load("{}/{}".format(experiment_path,model_file))
    scaler = pickle_from_file("{}/{}".format(experiment_path,scaler_file))
    encoder = VenusEncoder(dataset_features, len(ga_parameters.get("gene_types")))
    problem_constraints = ProblemConstraints()

    y = data.pop("charged_off").to_numpy()
    X  = data.to_numpy()

    output_dir = "{}/states/{}".format(experiment_path,experiment_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if ga_parameters.get("record_history"):
        history_dir = "{}/history/{}".format(experiment_path,experiment_id)
        Path(history_dir).mkdir(parents=True, exist_ok=True)
    
    for i,state in enumerate(X):
        print("state {}".format(i))

        if max_states>0 and i > max_states:
            break

        initial_state = copy.copy(state)
        problem, algorithm, termination = init_attack(
                state, model, scaler, encoder, problem_constraints, **ga_parameters
        )
        result = minimize(
            problem, algorithm, termination, verbose=2, save_history=False,
        )

        evaluation = ProblemEvaluation(result, encoder, initial_state, threshold, model)
        objectives = evaluation.calculate_objectives()
        save_to_file(objectives,"{}/s{}.npy".format(output_dir,i))

        if ga_parameters.get("record_history"):
            json_to_file(problem.history, "{}/s{}.json".format(history_dir,i))
                
    return experiment_id

        