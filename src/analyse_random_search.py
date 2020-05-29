import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_columns", 500)

# ----- PARAMETERS

input_dir = "../out/venus_attacks/coeva2_all_random_fitness"
nsga2_file = "../out/venus_attacks/nsga2_all/nsga2.csv"
n_objective = 4
n_best = 10


objectives = ["objective_{}".format(i + 1) for i in range(n_objective)]

all_files = glob.glob(input_dir + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

original_parameter = df[df["weight_4"] == 1000]

nsga2 = pd.read_csv(nsga2_file)

print("----- Comparing {} weight parameters.".format(len(df)))


fig, ax = plt.subplots()
df.boxplot(column=objectives)
for i in range(n_objective):
    plt.plot(i + 1, original_parameter[objectives[i]], "r.")
    plt.plot(i + 1, nsga2[objectives[i]], "b.")

plt.show()

df.plot(y=objectives, use_index=True)
plt.show()

df = df.sort_values(by=["objective_3"], ascending=[0])
print("--- {} best parameters and results.".format(n_best))
print(df.head(n_best))
print("--- Original parameters and results")
print(original_parameter)


# Print the index of the objective from smaller to bigger
ordering = []
for index, row in df.iterrows():
    row_np = row.to_numpy()
    weights = row_np[:4]
    ordering.append(np.argsort(weights))

ordering = np.array(ordering)
# For the 10 best elements, check which index is the more often the smallest

ordering = ordering[:n_best]
for i in range(4):
    print("The {}th smallest weight is:".format(i))
    (unique, counts) = np.unique(ordering[:, i], return_counts=True)
    print((unique, counts))

# Best order seems to be 1 3 2 0, or b d c a
# Best order seems to be 1 3 0 2, or b d a c
