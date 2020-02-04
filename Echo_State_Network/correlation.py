import numpy as np
import nolds



states = np.loadtxt("Rossler_reservoir_states_node200.txt",dtype='float32')
dimension = 5
correlation_dimension = np.zeros((dimension,2))
for i in range(1,dimension,1):
    data = states.T[1,:1000* (10**i)*5]
    correlation_dimension[i-1] = np.array([[nolds.corr_dim(data = data, emb_dim = i),i]])
    print("correlation dimension at dimension" +str(i) + "is" + str(correlation_dimension))
    print("#############################################")

np.savetxt("Rossler_correlation_dimension.txt", correlation_dimension)