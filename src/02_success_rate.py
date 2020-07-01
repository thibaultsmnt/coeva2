import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from pathlib import Path
from joblib import load
import pandas as pd

from attacks.objectives import calculate_success_rates
from attacks.venus_encoder import VenusEncoder
from utils import Pickler, in_out

config = in_out.get_parameters()


def run(
    MODEL_PATH=config["paths"]["model"],
    ATTACK_RESULTS_PATH=config["paths"]["attack_results"],
    OBJECTIVES_PATH=config["paths"]["objectives"],
    THRESHOLD=config["threshold"],
):
    Path(OBJECTIVES_PATH).parent.mkdir(parents=True, exist_ok=True)

    efficient_results = Pickler.load_from_file(ATTACK_RESULTS_PATH)
    encoder = VenusEncoder()
    model = load(MODEL_PATH)
    model.set_params(verbose=0, n_jobs=1)
    success_rates = calculate_success_rates(
        efficient_results, encoder, THRESHOLD, model
    )

    columns = ["o{}".format(i + 1) for i in range(success_rates.shape[0])]
    success_rate_df = pd.DataFrame(success_rates.reshape([1, -1]), columns=columns,)
    success_rate_df.to_csv(OBJECTIVES_PATH, index=False)


if __name__ == "__main__":
    run()
