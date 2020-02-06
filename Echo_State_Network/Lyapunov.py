import nolds
import numpy as np

data = np.loadtxt("Rossler_reservoir_states_node100_no1.txt")[:40000]
lyapunov_max = np.zeros((5,1))
for i in range(4):
    
    lyapunov_max[i] = np.array([i+1, nolds.lyap_r(data = data[:10000*(i+1)], emb_dim =5)])
    
np.savetxt("Rossler_reservoir_maximum_lyapunov_emb4_node1.txt", lyapunov_max)
print("maximun lyapunov exponent of Rossler is ")
print(lyapunov_max)