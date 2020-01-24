import nolds
import numpy as np

data = np.loadtxt("Henon.txt").T[0,:60000]

lyapunov_max = nolds.lyap_r(data = data,emb_dim = 6)

print("maximun lyapunov exponent of Henon is ")
print(lyapunov_max)