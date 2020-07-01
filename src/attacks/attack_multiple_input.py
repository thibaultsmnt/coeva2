from joblib import Parallel, delayed

from attacks.coeva2_scaled.venus_attack import attack as coeva2_attack
from attacks.nsga2_scaled.venus_attack import attack as nsga2_attack


def attack(
    model,
    scaler,
    encoder,
    n_gen,
    pop_size,
    n_offspring,
    X_initial_states,
    weight=None,
    attack_type="coeva2",
):
    if attack_type == "coeva2":
        attack = coeva2_attack
    elif attack == "nsga2":
        attack = nsga2_attack

    results = Parallel(n_jobs=-1)(
        delayed(attack)(
            index,
            initial_state,
            model,
            scaler,
            encoder,
            n_gen,
            pop_size,
            n_offspring,
            weight,
        )
        for index, initial_state in enumerate(X_initial_states)
    )

    return results
