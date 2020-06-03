from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.operators.mixed_variable_operator import (
    MixedVariableSampling,
    MixedVariableMutation,
    MixedVariableCrossover,
)
from .venus_coeva2_all_problem import VenusProblem

def init_attack(
    initial_state, model, scaler, encoder, problem_constraints, gene_types, objectives_weight, n_generation, n_offsprings, pop_size, record_history=False
):

    problem = VenusProblem(initial_state, objectives_weight, model, encoder, scaler, problem_constraints, len(gene_types), record_history)

    type_mask = gene_types

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

    algorithm = GA(
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
