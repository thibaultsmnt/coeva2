import copy, json, sys

import pandas as pd
import numpy as np
from joblib import load
import logging
from pathlib import Path
from pymoo.optimize import minimize
from datetime import datetime

sys.path.append("./")
from src.utils import Pickler
from src.coeva2.venus_encoder import VenusEncoder
from src.coeva2.venus_attack_generator import init_attack
from src.coeva2.problem_definition import ProblemConstraints, ProblemEvaluation



def run(config_file="./configurations/config1.json"):

    parameters= {}
    with open(config_file) as f:
        parameters = json.load(f)

    if parameters.get("experiment_path", None) is None:
        raise KeyError()

    
    seed =  int(datetime.timestamp(datetime.now()))

    dataset_path = parameters.get("dataset_path")
    dataset_features = parameters.get("dataset_features")
    experiment_path = parameters.get("experiment_path")
    scaler_file = parameters.get("scaler_file")
    model_file = parameters.get("model_file")
    threshold = parameters.get("model_threshold")
    ga_parameters = parameters.get("ga_parameters")
    dataset_constraints = parameters.get("dataset_constraints")

    data = pd.read_csv(dataset_path)
    model = load("{}/{}".format(experiment_path,model_file))
    scaler = Pickler.load_from_file("{}/{}".format(experiment_path,scaler_file))
    encoder = VenusEncoder(dataset_features, len(ga_parameters.get("gene_types")))
    problem_constraints = ProblemConstraints()

    y = data.pop("charged_off").to_numpy()
    X  = data.to_numpy()

    output_dir = "{}/states/{}".format(experiment_path,seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if ga_parameters.get("record_history"):
        history_dir = "{}/history/{}".format(experiment_path,seed)
        Path(history_dir).mkdir(parents=True, exist_ok=True)

    group_states = []
    step = 50
    
    
    for i,state in enumerate(X):

        initial_state = copy.copy(state)
        problem, algorithm, termination = init_attack(
                state, model, scaler, encoder, problem_constraints, **ga_parameters
        )
        result = minimize(
            problem, algorithm, termination, verbose=0, save_history=False,
        )

        evaluation = ProblemEvaluation(result, encoder, initial_state, threshold, model)
        objectives = evaluation.calculate_objectives()

        print(objectives)
        group_states.append(objectives)

        if i%step ==0 & i>0:
            np.save("{}/s{}_{}".format(output_dir,i),np.array(group_states))
            group_states = []

        
        if ga_parameters.get("record_history"):
            with open("{}/s{}_{}".format(output_dir,i),np.array(group_states), 'w') as outfile:
                json.dump(data, problem.history)
                

        