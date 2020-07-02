import pandas as pd
from utils import in_out
import matplotlib.pyplot as plt

config = in_out.get_parameters()

# ----- CONSTANT
n_objectives = 4
objectives_col = ["o{}".format(i + 1) for i in range(4)]


def plot_single(ax, data, style):
    for i in range(n_objectives):
        plt.plot(
            i + 1, data[objectives_col[i]], style, axes=ax,
        )


def plot_multiple(ax, data):
    data.boxplot(ax=ax, column=objectives_col)


def run(METHODS=config["methods"]):
    fig, ax = plt.subplots()
    for method in METHODS:
        data = pd.read_csv(method["path"])
        if len(data) > 1:
            plot_multiple(ax, data)
        else:
            plot_single(ax, data, method["style"])

    plt.show()


if __name__ == "__main__":
    run()
