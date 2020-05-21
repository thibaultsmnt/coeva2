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
    print("Copy process time {}".format(time.clock() - t0))

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

    print("Create attack process time {}".format(time.clock() - t0))

    # Execute attack

    result = minimize(problem, algorithm, termination, verbose=0, save_history=False,)
    print("Execute attack process time {}".format(time.clock() - t0))

    # Calculate objectives

    objectives = calculate_objectives(
        result, pop_size, encoder, initial_state, threshold, model
    )
    print("Calculate objectives process time {}".format(time.clock() - t0))

    return objectives


def calculate_objectives(result, pop_size, encoder, initial_state, threshold, model):

    CVs = np.array(list(map(lambda x: x.CV[0], result.pop)))
    Xs = np.array(list(map(lambda x: x.X, result.pop))).astype(np.float64)
    respectsConstraints = (CVs == 0).astype(np.int64)
    Xs_ml = encoder.from_genetic_to_ml(initial_state, Xs).astype(np.float64)
    isMisclassified = np.array(model.predict_proba(Xs_ml)[:, 1] < threshold).astype(
        np.int64
    )
    isBigAmount = (Xs[:, 0] >= 10000).astype(np.int64)

    o3 = respectsConstraints * isMisclassified
    o4 = o3 * isBigAmount
    objectives = np.array([respectsConstraints, isMisclassified, o3, o4])
    objectives = objectives.sum(axis=1)
    objectives = (objectives > 0).astype(np.int64)

    return objectives
