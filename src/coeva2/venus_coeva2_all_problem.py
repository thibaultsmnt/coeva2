from pymoo.model.problem import Problem
import numpy as np

class VenusProblem(Problem):

    FEATURE_TO_MAXIMIZE = 0

    def __init__(self, initial_state, objectives_weight, model, encoder, scaler, problem_constraints, nb_genes, record_history=False):
        min_max = encoder.get_min_max_genetic()
        self.weight = objectives_weight
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.original = initial_state
        self.original_mm = self.scaler.transform([initial_state])[0]
        self.problem_constraints = problem_constraints

        self.record_history = record_history
        self.history = {"X": [], "X_ml": [], "F": [], "G": []}

        super().__init__(n_var=nb_genes, n_obj=1, n_constr=problem_constraints.nb_constraints, xl=min_max[0], xu=min_max[1])

    def _evaluate(self, x, out, *args, **kwargs):


        x_ml = self.encoder.from_genetic_to_ml(self.original, x).astype("float64")
        x_ml_mm = self.scaler.transform(x_ml)

        # f1 Maximize probability of target
        LOG_ALPHA = 0.00000001
        f1 = np.log(self.model.predict_proba(x_ml)[:, 1] + LOG_ALPHA)

        # f2 Minimize perturbation
        l2_distance = np.linalg.norm(x_ml_mm[:, 1:] - self.original_mm[1:], axis=1)
        f2 = l2_distance

        # f3 Maximize amount
        AMOUNT_BETA = 0.00000001
        f3 = 1 / (x_ml[:, VenusProblem.FEATURE_TO_MAXIMIZE] + AMOUNT_BETA)

        objectives = [f1, f2, f3]

        # R Random objective
        if len(self.weight)==5:
            R = np.random.rand()
            objectives.append(R)

        # f4 Domain constraints

        constraints = self.problem_constraints.evaluate(x_ml)

        out["F"] = sum(i[0] * i[1] for i in zip(objectives, self.weight[:-1]))
        out["G"] = constraints * self.weight[-1]

        if self.record_history:
            self.history["X"].append(x.tolist())
            self.history["X_ml"].append(x_ml.tolist())
            self.history["F"].append(out["F"].tolist())
            self.history["G"].append(out["G"].tolist())
