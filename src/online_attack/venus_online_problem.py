from pymoo.model.problem import Problem
import numpy as np
from . import venus_constraints
import copy
import time
import logging


def _f1(X_ml, model, target, threshold):
    LOG_ALPHA = 0.00000001
    init_target_proba = model.predict_proba([target])[0, 1]
    y_pred_proba = model.predict_proba([X_ml])
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(np.int64)
    y_poisoned = 1 - y_pred
    model.partial_fit(np.stack((X_ml,) * 10), np.stack((y_poisoned[0],) * 10))
    target_proba_delta = -1 * (model.predict_proba([target])[0, 1] - init_target_proba)
    return target_proba_delta


class VenusProblem(Problem):
    def __init__(self, initial_state, model, encoder, scaler, target, threshold):
        min_max = encoder.get_min_max_genetic()
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.original = initial_state
        self.original_mm = self.scaler.transform([initial_state])[0]
        self.target = target
        self.threshold = threshold
        super().__init__(n_var=15, n_obj=2, n_constr=10, xl=min_max[0], xu=min_max[1])

    def _evaluate(self, x, out, *args, **kwargs):

        # ----- PARAMETERS

        x_ml = self.encoder.from_genetic_to_ml(self.original, x).astype("float64")
        x_ml_mm = self.scaler.transform(x_ml)

        # f1 Maximize probability of target
        f1 = np.array(
            list(map(lambda z: _f1(z, self.model, self.target, self.threshold), x_ml))
        )

        # f2 Minimize perturbation
        l2_distance = np.linalg.norm(x_ml_mm[:, 1:] - self.original_mm[1:], axis=1)
        f2 = l2_distance

        # f4 Domain constraints

        constraints = venus_constraints.evaluate(x_ml)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = constraints
