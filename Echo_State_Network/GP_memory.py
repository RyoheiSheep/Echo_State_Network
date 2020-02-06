import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import time
from tqdm import tqdm

######### get states from txt file  ###############
states = np.loadtxt("Rossler_reservoir_states_node100_no1.txt")[:50000]
# states = np.loadtxt("sinewave.txt")

########## choose a node for calculation ###########

############ set parameters for delayed coordination ######
tau = 16
dimension = 10
############# make a delayed coordination #################
length = len(states)
x = np.zeros((dimension, length- dimension*tau))
for i in range(dimension):
    x[i] = np.array([states[i*tau:length- (dimension-i)*tau ]])
x = x.T       
print("shape of x")
print(x.shape)

###########  set variavles for G-P method  ###############
D2 = np.zeros((dimension,1))
num_split = 10
hajime = -15.0
owari = -1.2
r_list = np.zeros(num_split)
kukan = owari - hajime 
katamuki  = kukan/((num_split-1)**2)
for i in range(num_split):
    r_list[i] = hajime + katamuki *i*i
r_list = np.exp(r_list)
##################################################
#########calculate distance of  all pair points########
# for i in range(1, len(x), 1):
#     j = 0
#     while(j<i):
#         d = np.linalg.norm(x[i]-x[j])
        
######################  x.shape =  ( length of states x dimension)
############### calculation of GP method  ####################
@jit#('void(f8[:],f8[:,:],i8)')
def Corr_dim(r_list, x, dimension):
    Cm = np.zeros(len(r_list))
    D2 = np.zeros((dimension,2))
    plt.figure()
    plt.xlabel("log(r)")
    plt.ylabel("log(Cm(r))")
    global hajime, owari
    plt.xlim([hajime,owari])
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "maroon", "magenta", "purple", "crimson", "orangered","darkorange", "darkcyan", "olive" ]
    size = len(x)
    dist = np.zeros((size, size))
    
    for k in tqdm(range(dimension)):   ########### 各埋め込み次元に対して相関次元を計算するためのループ
        v = x[:10000+int(30000/dimension)*k,:k+1]            ####### 各埋め込み次元に対して状態を切り出し
        time = 0
        for low in v:
            dist[time] = np.sum(np.square(low -v), axis = 1)
            time+=1
        l=0
        for r in tqdm(r_list):                          ######相関積分を計算
            count = 0
            for low in dist:
                count+= np.heaviside(r - low,0)
            Cm[l]= np.sum(count)*2/size/(size-1)
            l+=1
        #########calculate distance of  all pair points########
        # step = 0
        # for i in range(1, len(v), 1):
        #     for j in range(i):
        #         d[step] = np.linalg.norm(v[i]-v[j])     ##すべての2点間の距離を計算してリストdに格納#####
        # l=0        
        # for r in r_list:                          ######相関積分を計算
        #     count = np.heaviside(r - d,0)
        #     Cm[l]= np.sum(count)*2/len(v)/(len(v)-1)
        #     l+=1
        # plt.figure()
        plt.plot(np.log(r_list), np.log(Cm), lw = 1, color = colorlist[k], label = "m" +str(k))
        # plt.scatter(np.log(r_list), np.log(Cm), marker = "o", s = 0.5, color = 'blue')
        cut_length = 5
        A = np.array([np.log(r_list)[cut_length:cut_length+4], np.ones(4)])
        A = A.T
        a= np.linalg.lstsq(A,np.log(Cm)[cut_length:cut_length+4])[0][0]   ####最小2乗法で傾きを計算
        D2[k] = np.array([[k+1, a]])
    np.savetxt("Rossler_resevoir_corr_dim_no1.txt", D2)
    plt.figure()
    plt.scatter(D2.T[0], D2.T[1], marker = 'o', s = 1, color='blue')
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Corrrelation Dimension")
    plt.yticks([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7])
start_time = time.time()
# D2 = Cm(r= r, x = x, D2 = D2)
Corr_dim(r_list = r_list, x = x, dimension = dimension )
print("processing time is")
print(time.time()-start_time)
print("###############")

plt.show()
# print("D2 is ")
# print(D2)
# print("x is ")
# print(x)
# print("state")
# print(states)