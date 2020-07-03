from joblib import load


def run(MODEL_PATH):
    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)
