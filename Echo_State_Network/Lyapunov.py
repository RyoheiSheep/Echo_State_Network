import nolds
import numpy as np

data = np.loadtxt("Henon_reservoir_states_node100.txt").T[1,:500000]

lyapunov_max = nolds.lyap_r(data = data, emb_dim = 10)

np.savetxt("Henon_reservoir_maximum_lyapunov_emb10_node1.txt", lyapunov_max)
print("maximun lyapunov exponent of Henon is ")
print(lyapunov_max)