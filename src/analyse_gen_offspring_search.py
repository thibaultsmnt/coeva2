import pandas as pd
import glob
import matplotlib.pyplot as plt

# ----- PARAMETERS

input_dir = "../out/venus_attacks/nsga2_gen_offsprings_search"
n_objective = 4


all_files = glob.glob(input_dir + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

fig, ax = plt.subplots()
for i in range(n_objective):
    df.plot(ax=ax, x="n_offsprings", y="objective_{}".format(i + 1))

plt.show()
