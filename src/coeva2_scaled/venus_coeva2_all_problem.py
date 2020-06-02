import copy

from pymoo.model.problem import Problem
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        self.LOG_ALPHA = 0.00000001
        self.AMOUNT_BETA = 0.00000001
        f1_scaler = MinMaxScaler(feature_range=(0, 1))
        f1_scaler.fit([[np.log(self.LOG_ALPHA), np.log(1)]])
        self.f1_scaler = f1_scaler
        f2_scaler = MinMaxScaler(feature_range=(0, 1))
        f2_scaler.fit([[0, np.sqrt(15)]])
        self.f2_scaler = f2_scaler
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
        f1 = self.model.predict_proba(x_ml)[:, 1]
        f1[f1 < self.LOG_ALPHA] = self.LOG_ALPHA
        f1 = np.log(f1)
        f1 = self.f1_scaler.transform(f1)

        # f2 Minimize perturbation
        l2_distance = np.linalg.norm(x_ml_mm[:, 1:] - self.original_mm[1:], axis=1)
        f2 = l2_distance
        f2 = self.f2_scaler.transform(f2)

        # f3 Maximize amount (f3 don't need scaler if amount > 1)
        amount = copy.deepcopy(x_ml[:, 0])
        amount[amount <= 0] = self.AMOUNT_BETA
        f3 = 1 / amount

        # f4 Domain constraints

        constraints = venus_constraints.evaluate(x_ml, self.encoder)

        out["F"] = alpha * f1 + beta * f2 + gamma * f3
        out["G"] = constraints * delta
