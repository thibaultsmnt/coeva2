from pymoo.optimize import minimize

from .venus_attack_generator import create_attack
import numpy as np
import logging
import copy
import datetime
import time


def attack(
    index,
    initial_state,
    weight,
    model,
    scaler,
    encoder,
    n_generation,
    n_offsprings,
    pop_size,
    threshold,
):

    # Logging

    logging.debug("Attack #{}".format(index))
    print("{}: Attack #{}".format(datetime.datetime.now(), index))

    # Copying shared resources

    t0 = time.clock()

    weight = copy.deepcopy(weight)
    model = copy.deepcopy(model)
    scaler = copy.deepcopy(scaler)
    encoder = copy.deepcopy(encoder)
    print("Copy process time {}".format(time.clock()))

    # Create attack

    problem, algorithm, termination = create_attack(
        initial_state,
        weight,
        model,
        scaler,
        encoder,
        n_generation,
        n_offsprings,
        pop_size,
    )

    print("Create attack process time {}".format(time.clock()))

    # Execute attack

    result = minimize(problem, algorithm, termination, verbose=0, save_history=False,)
    print("Execute attack process time {}".format(time.clock()))

    # Calculate objectives

    objectives = calculate_objectives(
        result, pop_size, encoder, initial_state, threshold, model
    )
    print("Calculate objectives process time {}".format(time.clock()))
    
    return objectives



def calculate_objectives(result, pop_size, encoder, initial_state, threshold, model):

    respectsConstraints = np.zeros(pop_size)
    isMisclassified = np.zeros(pop_size)
    isBigAmount = np.zeros(pop_size)
    for i, individual in enumerate(result.pop):
        respectsConstraints[i] = (individual.CV[0] == 0).astype(np.int64)
        X = np.array(individual.X).astype(np.float64)
        x_ml = encoder.from_genetic_to_ml(initial_state, np.array([X])).astype(
            "float64"
        )
        isMisclassified[i] = np.array(
            model.predict_proba(x_ml)[:, 1] < threshold
        ).astype(np.int64)[0]
        isBigAmount[i] = (X[0] >= 10000).astype(np.int64)

    o3 = respectsConstraints * isMisclassified
    o4 = o3 * isBigAmount
    objectives = np.array([respectsConstraints, isMisclassified, o3, o4])
    objectives = objectives.sum(axis=1)
    objectives = (objectives > 0).astype(np.int64)

    return objectives
