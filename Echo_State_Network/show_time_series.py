import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


data = np.loadtxt("Lorentz.txt")
data = data.T
plt.figure()

plt.scatter(data[0,:50000], data[2,:50000],  marker = "o", s = 0.05)

ax = Axes3D(plt.figure())
ax.plot(data[0,:50000], data[1,:50000], data[2,:50000], color = "magenta", lw = 0.1)
plt.show()