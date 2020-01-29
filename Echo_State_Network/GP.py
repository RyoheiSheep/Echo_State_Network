
import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import time
######### get states from txt file  ###############
states = np.loadtxt("reservoir_states_node200.txt")
########## choose a node for calculation ###########
states = states.T[1,:1000]

############ set parameters for delayed coordination ######
tau = 16
dimension = 10

###########  set variavles for G-P method  ###############
D2 = np.zeros((dimension,1))
r = 0.5
summation =0
length = len(states)

############# make a delayed coordination #################
x = np.zeros((dimension, length- dimension*tau))
for i in range(dimension):
    x[i] = np.array([states[i*tau:length- (dimension-i)*tau ]])
x = x.T       

##################################################

######################  x.shape =  ( length of states x dimension)
############### calculation of GP method  ####################
# @jit 
def Cm(x,D2):
    summation = 0
    for k in range(dimension):   ########### 各埋め込み次元に対して相関次元を計算するためのループ
        v = x[:,:k+1]            ####### 各埋め込み次元に対して状態を切り出し
        
        for i in range(v.shape[0]):      ######### sum under the condition condition s.t. i<j
            for j in range(0,i,1):
                summation+= np.heaviside(r - np.linalg.norm(v[i]-v[j]),0)
        
        D2[k] = 2/v.shape[0]/(v.shape[0]-1)*summation
    return D2        
start_time = time.time()
D2 = Cm(x = x, D2 = D2)
print("processing time is")
print(time.time()-start_time)
print("###############")
print("D2 is ")
print(D2)
        