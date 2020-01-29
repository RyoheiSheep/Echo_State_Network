###############Takens.py################

import numpy as np
from sklearn import metrics


class Takens:

	'''
	constant
	'''
	tau_max = 100


	'''
	initializer
	'''
	def __init__(self, data):
		self.data           = data
		self.tau, self.nmi  = self.__search_tau()


	'''
	reconstruct data by using searched tau
	'''
	def reconstruct(self):
		_data1 = self.data[:-2]
		_data2 = np.roll(self.data, -1 * self.tau)[:-2]
		_data3 = np.roll(self.data, -2 * self.tau)[:-2]
		return _data1, _data2, _data3


	'''
	find tau to use Takens' Embedding Theorem
	'''
	def __search_tau(self):

		# Create a discrete signal from the continunous dynamics
		hist, bin_edges = np.histogram(self.data, bins=200, density=True)
		bin_indices = np.digitize(self.data, bin_edges)
		data_discrete = self.data[bin_indices]

		# find usable time delay via mutual information
		before     = 1
		nmi        = []
		res        = None

		for tau in range(1, self.tau_max):
			unlagged = data_discrete[:-tau]
			lagged = np.roll(data_discrete, -tau)[:-tau]
			nmi.append(metrics.normalized_mutual_info_score(unlagged, lagged))

			if res is None and len(nmi) > 1 and nmi[-2] < nmi[-1]:
				res = tau - 1

		if res is None:
			res = 50

		return res, nmi


data = np.loadtxt("reservoir_states_node200.txt")
embed = Takens(data = data.T[1])
print("delayed time is ")
print(embed.tau)

