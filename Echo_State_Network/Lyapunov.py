import nolds
import numpy as np

data = np.loadtxt("Henon.txt").T[0,:10000]

lyapunov_max = nolds.lyap_r(data = data)

print("maximun lyapunov exponent of Henon is ")
print(lyapunov_max)