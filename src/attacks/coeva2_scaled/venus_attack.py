from pymoo.optimize import minimize

from .venus_attack_generator import create_attack
import numpy as np
import copy
from src.attacks.venus_constraints import evaluate


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

    return result
