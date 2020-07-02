from pymoo.optimize import minimize

from .venus_attack_generator import create_attack
import numpy as np
import copy


def attack(
    index, initial_state, model, scaler, encoder, n_gen, pop_size, n_offsprings,
):
    # Copying shared resources

    model = copy.deepcopy(model)
    scaler = copy.deepcopy(scaler)
    encoder = copy.deepcopy(encoder)

    # Create attack

    problem, algorithm, termination = create_attack(
        initial_state, model, scaler, encoder, n_gen, pop_size, n_offsprings,
    )

    # Execute attack

    result = minimize(problem, algorithm, termination, verbose=0, save_history=False,)

    # Calculate objectives

    return result
