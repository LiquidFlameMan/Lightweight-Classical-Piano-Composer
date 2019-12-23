import glob
from math import floor

import matplotlib.pyplot as plt

weight = []
for file in glob.glob("weight_new/*.hdf5"):
    weight.append(file.split("-")[3])
weight = sorted(weight)[::-1]
plt.plot(range(1, len(weight) - 5, 5),
         [weight[5 * i] for i in range(1, floor(len(weight) / 5.0))], 'ro')
plt.gca().invert_yaxis()
plt.savefig('example.png')
plt.show()
