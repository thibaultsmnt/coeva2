from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling,
    MixedVariableMutation,
    MixedVariableCrossover,
)
from .venus_coeva2_all_problem import VenusProblem


def create_attack(
    initial_state, model, scaler, encoder, n_generation, n_offsprings, pop_size
):

    problem = VenusProblem(initial_state, model, encoder, scaler)

    type_mask = [
        "real",
        "int",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "real",
        "int",
        "real",
        "real",
        "int",
    ]

    sampling = MixedVariableSampling(
        type_mask,
        {"real": get_sampling("real_random"), "int": get_sampling("int_random")},
    )

    # Default parameters for crossover (prob=0.9, eta=30)
    crossover = MixedVariableCrossover(
        type_mask,
        {
            "real": get_crossover("real_sbx", prob=0.9, eta=30),
            "int": get_crossover("int_sbx", prob=0.9, eta=30),
        },
    )

    # Default parameters for mutation (eta=20)
    mutation = MixedVariableMutation(
        type_mask,
        {
            "real": get_mutation("real_pm", eta=20),
            "int": get_mutation("int_pm", eta=20),
        },
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
        return_least_infeasible=True,
    )

    termination = get_termination("n_gen", n_generation)

    return problem, algorithm, termination
