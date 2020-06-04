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
from src.utils.rf_attacks import RFAttack

from src.coeva2.problem_definition import ProblemConstraints

def run(config_file, experiment_id=None,nb_estimators=10, nb_iterations=10):

    parameters= {}
    with open(config_file) as f:
        parameters = json.load(f)

    if parameters.get("experiment_path", None) is None:
        raise KeyError()

    experiment_id =  int(datetime.timestamp(datetime.now())) if experiment_id is None else experiment_id
    dataset_path = parameters.get("dataset_path")
    experiment_path = parameters.get("experiment_path")
    scaler_file = parameters.get("scaler_file")
    model_file = parameters.get("model_file")
    threshold = parameters.get("model_threshold")

    dataset_constraints = parameters.get("dataset_constraints")
    max_states = parameters.get("max_states",0)

    print("running Papernot with config {} experiment {} id {}".format(config_file, experiment_path, experiment_id))

    data = pd.read_csv(dataset_path)
    model = load("{}/{}".format(experiment_path,model_file))
    scaler = pickle_from_file("{}/{}".format(experiment_path,scaler_file))
    problem_constraints = ProblemConstraints()

    y = data.pop("charged_off").to_numpy()
    X  = data.to_numpy()


    attack = RFAttack(model, nb_estimators=nb_estimators, nb_iterations=nb_iterations, threshold=threshold)
    adv, rf_success_rate, l_2, l_inf = attack.generate(X[:max_states],y[:max_states])

    problem_constraints = ProblemConstraints()
    constraints = problem_constraints.evaluate(adv)
    constraints_violated = constraints > 0
    constraints_violated = constraints_violated.sum(axis=1).astype(bool)
    respectsConstraints = 1 - constraints_violated

    isMisclassified = np.array(model.predict_proba(adv)[:, 1] < threshold).astype(
            np.int64
        )
    isBigAmount = (adv[:, 0] >= 10000).astype(np.int64)

    o3 = respectsConstraints * isMisclassified
    o4 = o3 * isBigAmount
    objectives = np.array([respectsConstraints, isMisclassified, o3, o4])
    objectives = objectives.sum(axis=1)
    objectives = (objectives > 0).astype(np.int64)
    

    return adv, objectives

