#####################################################
####Echo State Network trained by adapted filter ####
#####################################################

import numpy as np 
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from Functions import get_reservoir_states, train, one_step_predict, RMSE



start_time = time.time()
#################################################################
####definition of parameters of Echo State Network###############
#################################################################

num_input_node = 3
num_reservoir_node =200
num_output_node = 3

##################################################################
bias = None
if bias is None:
    num_reservoir_node = num_reservoir_node
else:
    num_reservoir_node = num_reservoir_node + 1
##################################################################
#############  parameters of training and data####################
##################################################################

T = 600
dt = 0.001
num_time_step = int(T/dt)

##################################################################
################  Parameters of Training and Prediction ##########
##################################################################

wash_time = 100
length_train = 50000
length_prediction = 10000
# length_freerun_prediction  = 500
    
RATIO_TRAIN = 0.6
LEAK_RATE=0.02

#**************************************************************************************************************#

######################################################################
############### preparation of data #################################
#####################################################################
teacher_data = np.loadtxt("Lorentz.txt")        ######学習用データ
# teacher_input_init = np.array([np.loadtxt("logistic.txt")]).T
teacher_input = teacher_data[wash_time :-1]
teacher_output = teacher_data[wash_time +1:]

train_data_input = teacher_input[:length_train]           #学習用入力
train_data_output = teacher_output[:length_train]      #学習用出力

test_data_input = teacher_input[length_train: length_train + length_prediction]
test_data_output= teacher_output[length_train: length_train + length_prediction]



#****************************************************************************************************************************************#

############################################################################################################################
####################  Set Weight ramdomly ##################################################################################
############################################################################################################################

#################  Initialize  All Weights   ######################################
weight_input_init = (np.random.normal(0, 1, num_input_node * num_reservoir_node).reshape((num_input_node, num_reservoir_node)) * 2 - 1) * 0.1
weight_reservoir_init = np.zeros((num_reservoir_node, num_reservoir_node))
weight_output_init = np.zeros((num_reservoir_node, num_output_node))

############### Set Reservoir Weight  #################################
matrix_nonzero1 = np.random.randint(0,2,(num_reservoir_node,num_reservoir_node))
matrix_nonzero2 = np.random.randint(0,2,(num_reservoir_node,num_reservoir_node))
weight_normal = np.random.normal(0, 1, num_reservoir_node * num_reservoir_node).reshape(num_reservoir_node, num_reservoir_node)
weights = matrix_nonzero1 * matrix_nonzero2 * weight_normal
spectral_radius = np.max(np.abs(linalg.eigvals(weights)))
weight_reservoir = weights / spectral_radius *0.997

##############  Set  Input Weight       #################################
matrix_nonzero = np.random.randint(0,2,(num_input_node, num_reservoir_node))
weight_input_init = weight_input_init * matrix_nonzero

#************************************************************************************************************************************************#

##############################################################################################################################
######################  Training #############################################################################################
##############################################################################################################################

######################  Get State  ##################################
initial_state = np.zeros((len(teacher_data), num_reservoir_node))
states = get_reservoir_states(inputs = teacher_data, states_init = initial_state, 
                                weight_input= weight_input_init, weight_reservoir=weight_reservoir_init, 
                                leak_rate = LEAK_RATE)
# np.savetxt('states.txt', states)
# states = np.loadtxt("states.txt")
#####################  Train  Output Weight   #########################

weight_output =  train(num_reservoir_node = num_reservoir_node, num_output_node = num_output_node, 
                        train_data_output = train_data_output,
                        states = states, weight_output_initial = weight_output_init,
                        wash_time = wash_time, LAMBDA = 0.00221)

#**********************************************************************************************************************************************#
############################################################################################################################### 
####################   Prediction #############################################################################################
###############################################################################################################################

################# Free Run Predicton######################################

# free_run_predict_result = free_run_predict(states = states, num_output_node = num_output_node, 
#                                             weight_output = weight_output, length_train = 101,
#                                             weight_input= weight_input_init, 
#                                             weight_reservoir=weight_reservoir, length_freerun=length_freerun)

#################  One Step Prediction  ##################################
one_step_predict_train = one_step_predict(num_output_node= num_output_node, 
                                            states=states[wash_time:], weight_output = weight_output, 
                                            length_prediction= length_train)

one_step_predict_test = one_step_predict(num_output_node= num_output_node, states=states[wash_time+length_train:], 
                                        weight_output = weight_output, 
                                        length_prediction= length_prediction)

print("length of one step predict test")
print(len(one_step_predict_test))
print("length of test data out put")
print(len(test_data_output))    
rmse_training = RMSE(one_step_predict_train, train_data_output)
rmse_test = RMSE(one_step_predict_test, test_data_output)
print("####################################")
print("RMSE Training")
print(rmse_training)
print("************************************")
print("RMSE Test")
print(rmse_test)
print("####################################")

################################################
process_time = time.time() - start_time#########
print("process_time is ")#######################
print(process_time)#############################
################################################

#*********************************************************************************************************************************************************#

################################################################################################################################
#################### Graphical Part    #########################################################################################
################################################################################################################################

#################### Free Run Prediction #############################################   
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot(free_run_predict_result.T[0,:100], free_run_predict_result.T[1,:100], free_run_predict_result.T[2,:100], color = "red", linewidth = 1)
# ax.set_title(" Free-Running Prediction")

#################  One Step Prediction  ###############################################

fig1 = plt.figure()
ax1 = Axes3D(fig1)
ax1.plot(one_step_predict_train.T[0,:], one_step_predict_train.T[1,:], one_step_predict_train.T[2,:], color = "blue", linewidth = 1 )
ax1.set_title(" One Step Prediction")


################# Comparison of Free Running and One Step  ##############################

# ax.plot(teacher_input_init.T[0,length_train:], teacher_input_init.T[1,length_train:], teacher_input_init.T[2,length_train:], color = "magenta", linewidth = 0.1)
# ax.set_title("real time series")
# print('one step predict')
# print(one_step_predict_result)
plt.figure()
# plt.plot(free_run_predict_result.T[0,:100], lw = 1.0, color ="red", label ="x of free runnning prediction")
plt.plot(one_step_predict_test.T[0,:100],lw = 1.0,  color = "blue", label ="one step prediction")
plt.plot(test_data_output.T[0][:100],lw = 1.0,  color = "red", label ="real time series")
plt.title('x of one step prediction and real time series')
plt.legend()

###################    Teacher Data     #############################################

# ax.set_xlim([-1.0, 1.0])
# ax.set_ylim([-1.0, 1.0])
# ax.set_zlim([-1.0, 1.0])

# ax.plot(states.T[0,:],states.T[1,:],states.T[2,:], color = "red", linewidth = 1)


############## state[t],   state[t + delta],   state[t + 2delta]########################
fig2= plt.figure()
ax2 = Axes3D(fig2)
ax2.plot(states.T[0,wash_time: 10000], states.T[0,wash_time+15: 10015], states.T[0,wash_time+30: 10030], color = "red", linewidth = 0.1 )
ax2.set_title("states delayed axis")

#######################################################################################

plt.show()
#******************************************************************************************************************************#

print("OK!")