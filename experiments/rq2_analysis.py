import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.in_out import load_from_dir

def run_success_rates(config_file, experiment_id, max_states = None):

    parameters= {}
    with open(config_file) as f:
        parameters = json.load(f)

    if parameters.get("experiment_path", None) is None:
        raise KeyError()

    experiment_path = parameters.get("experiment_path")
    states_path = "{}/states/{}".format(experiment_path,experiment_id)
    

    states = load_from_dir(states_path)
    if max_states is not None:
        states = states[:max_states]

    objectives =  np.array(states).sum(axis=0)/len(states)
    df = pd.DataFrame(data=[objectives], columns=["respectsConstraints", "isMisclassified", "o3", "o4"])
    df["run_id"] = experiment_id
    df = df[["run_id","respectsConstraints", "isMisclassified", "o3", "o4" ]]

    return df

    


def run_history_analysis(config_file, experiment_id):

    parameters= {}
    with open(config_file) as f:
        parameters = json.load(f)

    if parameters.get("experiment_path", None) is None:
        raise KeyError()

    experiment_path = parameters.get("experiment_path")
    history_dir = "{}/history/{}".format(experiment_path,experiment_id)

    file_id = "s0.json"
    with open("{}/{}".format(history_dir,file_id)) as f:
        hist = json.load(f)

        F= np.array(hist.get("F"))
        G= np.array(hist.get("G"))

        print(F.shape,G.shape)

    scales = ['linear', 'log', 'log', 'log']
    y_labels = ['Prediction', 'L2 Perturbation', 'Overdraft', 'Constraint violation']
    fig, axs = plt.subplots(gen_data.shape[2], 1)
    x = range(gen_data.shape[1])
    for i, ax in enumerate(axs):
        ax.fill_between(x, gen_data_min_max_avg[0, :, i], gen_data_min_max_avg[1, :, i])
        ax.plot(x, gen_data_min_max_avg[2, :, i], color='red', linewidth=0.5)
        ax.set_yscale(scales[i])
        ax.set_xlabel('Generation')
        ax.set_ylabel(y_labels[i])


