import pandas as pd
import glob
import numpy as np

# ----- PARAMETERS

input_dirs = [
    "../out/venus_attacks/coeva2_all_random_fitness",
    # "../out/venus_attacks/coeva2_all_random_fitness_scaled",
    "../out/venus_attacks/coeva2_all_random_fitness_scaled_tol",
    "../out/venus_attacks/coeva2_original",
    "../out/venus_attacks/nsga2_all",
    "../out/venus_attacks/nsga2_all_scaled",
]
output_file = "../out/venus_attacks/bests.csv"
n_ojectives = 4

pd.set_option("display.max_columns", 500)
objective_cols = ["objective_{}".format(i + 1) for i in range(n_ojectives)]

bests = []
for input_dir in input_dirs:
    all_files = glob.glob(input_dir + "/*.csv")
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.sort_values(by=["objective_4"], ascending=[0])
    bests.append(df.head(1)[objective_cols].to_numpy())

bests = np.concatenate(bests)
data = {"name": input_dirs}
for i in range(n_ojectives):
    data.update({"objective_{}".format(i + 1): bests[:, i]})

bests = pd.DataFrame(data)
bests = bests.sort_values(by=["objective_4"], ascending=[0])
bests.to_csv(output_file)
print(bests)
