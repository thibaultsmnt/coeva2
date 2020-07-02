import numpy as np
from coeva2.venus_encoder import VenusEncoder


def _date_feature_to_month(feature):
    return np.floor(feature / 100) * 12 + (feature % 100)


encoder = VenusEncoder()
min_f = encoder.features["min"]
max_f = encoder.features["max"]

print("--- g41")
print(0)
a = (
    np.ceil(
        100
        * (max_f[0] * (max_f[2] / 1200) * (1 + max_f[2] / 1200) ** max_f[1])
        / ((1 + min_f[2] / 1200) ** min_f[1] - 1)
    )
    / 100
)
print(-(min_f[3] - a))
print(max_f[3])

print("--- g42")
print("{}".format(min_f[10] - max_f[14]))
print("{}".format(max_f[10] - min_f[14]))

print("--- g43")
print("{}".format(min_f[16] - max_f[11]))
print("{}".format(max_f[16] - min_f[11]))

print("--- g44")
print("{}".format(0))  # min f: y=|((36-x) (60-x))| on [36, 60]
print("{}".format(144))  # max f: y=|((36-x) (60-x))| on [36, 60]

print("--- g45")
print("0")
print("{}".format(-(min_f[20] - max_f[0] / min_f[6])))
print("{}".format(max_f[20] - min_f[0] / max_f[6]))

print("--- g46")
print("0")
print("{}".format(-(min_f[21] - max_f[10] / min_f[14])))
print("{}".format(max_f[21] - min_f[10] / max_f[14]))


print("--- g47")
print("0")
print(
    "{}".format(
        -(
            min_f[22]
            - (_date_feature_to_month(max_f[7]) - _date_feature_to_month(min_f[9]))
        )
    )
)
print(
    "{}".format(
        (
            max_f[22]
            - (_date_feature_to_month(min_f[7]) - _date_feature_to_month(max_f[9]))
        )
    )
)

print("--- g48")
print("0")
print("{}".format(-(min_f[23] - max_f[11] / min_f[22])))
print("{}".format(max_f[23] - min_f[11] / max_f[22]))

print("--- g49")
print("0")
print("{}".format(-(min_f[24] - max_f[16] / min_f[22])))
print("{}".format(max_f[24] - min_f[16] / max_f[22]))

print("--- g410")
print("0")
print("2")
