from joblib import load

from utils import Pickler, in_out

config = in_out.get_parameters()


def run(
    MODEL_PATH,
    ATTACK_RESULTS_PATH,
    TRAIN_TEST_DATA_DIR=config["dirs"]["train_test_data"],
):
    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)

    Pickler.load_from_file(ATTACK_RESULTS_PATH)

    adversarial = []
