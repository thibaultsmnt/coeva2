from pymoo.model.problem import Problem
import numpy as np
from . import venus_constraints


class VenusProblem(Problem):
    def __init__(self, initial_state, weight, model, encoder, scaler):
        min_max = encoder.get_min_max_genetic()
        self.weight = weight
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.original = initial_state
        self.original_mm = self.scaler.transform([initial_state])[0]
        super().__init__(n_var=15, n_obj=1, n_constr=10, xl=min_max[0], xu=min_max[1])

    def _evaluate(self, x, out, *args, **kwargs):

        # ----- PARAMETERS

        alpha = self.weight[0]
        beta = self.weight[1]
        gamma = self.weight[2]
        delta = self.weight[3]

        x_ml = self.encoder.from_genetic_to_ml(self.original, x).astype("float64")
        x_ml_mm = self.scaler.transform(x_ml)

        # f1 Maximize probability of target
        LOG_ALPHA = 0.00000001
        f1 = np.log(self.model.predict_proba(x_ml)[:, 1] + LOG_ALPHA)

        # f2 Minimize perturbation
        l2_distance = np.linalg.norm(x_ml_mm[:, 1:] - self.original_mm[1:], axis=1)
        f2 = l2_distance

        # f3 Maximize amount
        f3 = 1 / x_ml[:, 0]

        # f4 Domain constraints

        constraints = venus_constraints.evaluate(x_ml)

        out["F"] = alpha * f1 + beta * f2 + gamma * f3
        out["G"] = constraints * delta
