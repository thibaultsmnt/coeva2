import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

path = '../out/venus_attacks/coeva2_all_random_fitness_1_be'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

df.boxplot(column=['objective_3'])

df.plot(y=['objective_1','objective_2','objective_3', 'objective_4'], use_index=True)
plt.show()

df = df.sort_values(by=['objective_3'], ascending=[0])

# Print the index of the objective from smaller to bigger
ordering = []
for index, row in df.iterrows():
    row_np = row.to_numpy()
    weights = row_np[:4]
    ordering.append(np.argsort(weights))

# For the 10 best elements, check which index is the more often the smallest
ordering = np.array(ordering)
ordering = ordering[:10]
for i in range(4):
    print('Frequency in columns {}'.format(i))
    (unique, counts) = np.unique(ordering[:,i], return_counts=True)
    print((unique, counts))

# Best order seems to be 1 3 2 0, or b d c a
