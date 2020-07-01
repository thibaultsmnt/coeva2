from pymoo.optimize import minimize

from .venus_attack_generator import create_attack
import numpy as np
import copy
from .venus_constraints import evaluate


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
    # Copying shared resources

    weight = copy.deepcopy(weight)
    model = copy.deepcopy(model)
    scaler = copy.deepcopy(scaler)
    encoder = copy.deepcopy(encoder)

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

    # Execute attack

    result = minimize(problem, algorithm, termination, verbose=0, save_history=False,)

    # Calculate objectives

    objectives = calculate_objectives(result, encoder, initial_state, threshold, model)

    return objectives


def calculate_objectives(result, encoder, initial_state, threshold, model):

    CVs = np.array(list(map(lambda x: x.CV[0], result.pop)))
    Xs = np.array(list(map(lambda x: x.X, result.pop))).astype(np.float64)
    Xs_ml = encoder.from_genetic_to_ml(initial_state, Xs).astype(np.float64)
    respectsConstraints = (evaluate(Xs_ml, encoder).mean(axis=1) <= 0).astype(np.int64)
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
