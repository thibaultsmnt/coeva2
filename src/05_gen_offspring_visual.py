import pandas as pd
import matplotlib.pyplot as plt

from utils import in_out

config = in_out.get_parameters()

N_OBJECTIVES = 4


def run(OBJECTIVES_PATH=config["paths"]["objectives"]):
    df = pd.read_csv(OBJECTIVES_PATH)
    df.sort_values(by=["n_offsprings"], inplace=True)
    fig, ax = plt.subplots()
    for i in range(N_OBJECTIVES):
        df.plot(ax=ax, x="n_offsprings", y="o{}".format(i + 1))

    plt.show()


if __name__ == "__main__":
    run()
