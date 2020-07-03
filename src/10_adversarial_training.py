from joblib import load, dump

from attacks.objectives import adversarial_training
from attacks.venus_encoder import VenusEncoder
from utils import Pickler, in_out
import numpy as np
from sklearn.base import clone

config = in_out.get_parameters()


def run(
    MODEL_PATH=config["paths"]["model"],
    RESISTANT_MODEL_PATH=config["paths"]["resistant_model"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    TRAIN_TEST_DATA_DIR=config["dirs"]["train_test_data"],
    THRESHOLD=config["threshold"],
):
    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)
    encoder = VenusEncoder()

    attack_results = Pickler.load_from_file(ATTACK_RESULTS_PATH)
    X_adv = adversarial_training(attack_results, encoder, THRESHOLD, model)
    y_adv = np.zeros(X_adv.shape[0]) + 1

    X_train = np.load("{}/X_train.npy".format(TRAIN_TEST_DATA_DIR))
    y_train = np.load("{}/y_train.npy".format(TRAIN_TEST_DATA_DIR))

    resistant_model = clone(model)
    resistant_model.set_params(verbose=2, n_jobs=-1)
    resistant_model.fit(
        np.concatenate([X_train, X_adv]), np.concatenate([y_train, y_adv])
    )
    resistant_model.set_params(verbose=0, n_jobs=1)

    dump(resistant_model, RESISTANT_MODEL_PATH)


if __name__ == "__main__":
    run()
