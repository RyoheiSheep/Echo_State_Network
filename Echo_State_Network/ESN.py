#####################################################
####Echo State Network trained by adapted filter ####
#####################################################

import numpy as np 
from input_generator import InputGenerator 
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import time

    ###################################################################
    ##############  Get Reservoir State   #############################
    ###################################################################

# @jit
def get_reservoir_states(inputs, states_init, weight_input, weight_reservoir, leak_rate):
    states = states_init
    for i in range(len(inputs)-1):
        current_state = np.array([states[i]])
        states[i+1] = (1-leak_rate)*np.array(current_state) + leak_rate *np.tanh(np.array([inputs[i]]) @ weight_input + current_state @ weight_reservoir)
    # print(states)
    print("shape of states")
    print(states.shape)
    return states
    ######################################################################
    #####################   Train  #######################################
    ######################################################################
# @jit
def train(num_reservoir_node, num_output_node, train_data, states, teacher_output, weight_output_initial, LAMBDA = 0.00221):
    
    ################# Set Length of Training #########
    
    Length_Training = len(teacher_output)
    
    #############Initialize Variables#################
    
    re = 0
    kp = np.zeros((num_reservoir_node,1))
    e = np.zeros((1,num_output_node))
    P = 1/LAMBDA * np.identity(num_reservoir_node)
    weight_output = weight_output_initial
    #*******************************************************************#
    ###########  Training Using Recursive Least Square ##################
    
    for i in range(Length_Training):
        
        re = 1 + np.dot(np.dot(np.array([states[i+100]]), P), np.array([states[i+100]]).T)
        kp = np.dot(P,np.array([states[i+100]]).T )/ re
        e = np.array([teacher_output[i+100]]) -  np.dot(np.array([states[i+100]]),weight_output)
        weight_output = weight_output + np.dot(kp, e)
        P = P  -( np.dot(P,np.dot (np.array([states[i+100]]).T,np.dot(np.array([states[i+100]]), P))) / re)
        
    return weight_output

######################################################################
##############   prediction against the teacher output  ##############
######################################################################
# @jit
def one_step_predict(num_output_node, states, weight_output,  length_train, length_prediction):
    one_step_predict_result = np.zeros((length_prediction,num_output_node))
        
    # print("length of states")
    # print(len(states))
    # print("length of teacher data")
    # print(len(teacher_input))
    one_step_predict_result[0] = np.zeros((1,num_output_node))
    for i in range(1, length_prediction -1,1):
        one_step_predict_result[i] =  np.array([states[i]]) @ weight_output
    return one_step_predict_result

########################################################################
#############  Free Running Prediction #################################
########################################################################

# @jit
def free_run_predict(states, num_output_node, weight_output, length_train, weight_input, weight_reservoir, length_freerun):
    l = length_train - 1
    state_free_run = np.array([states[l]])
    free_run_predict_result = np.zeros((length_freerun, num_output_node))
    # print("freerun predict")
    # print(free_run_predict)
    for i in range(length_freerun):
        # print(" 1state_free_run @ weight_output")
        # print( state_free_run @ weight_output )
        # print("state type 8888888")
        # print(state_free_run.shape)
        # print("2free run predict")
        # print(free_run_predict)
        free_run_predict_result[i] = state_free_run @ weight_output
        # print("free run predict -1")
        # print(np.array([free_run_predict[-1]]).shape)
        # print("3np.array([free_run_predict[-1]]) @ weight_input + np.array([state_free_run]) @ weight_reservoir")
        # print((np.array(free_run_predict[-1]) @ weight_input + np.array(state_free_run) @ weight_reservoir).shape)
        state_free_run = np.tanh(np.array(free_run_predict_result[i]) @ weight_input + np.array(state_free_run) @ weight_reservoir)
    return free_run_predict_result

##########################################################################
###############  Calculation of Eroor ####################################
##########################################################################

def RMSE(outputs, teacher):
    error_vector = outputs - teacher
    summation = np.sum(np.square(error_vector))/len(error_vector)
    RMSE_ERROR = np.sqrt(summation)
    return RMSE_ERROR



def main():
    start_time = time.time()
    ######################################################
    ####definition of parameters of Echo State Network####
    ######################################################

    num_input_node = 3
    num_reservoir_node =200
    num_output_node = 3

    
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
    length_one_step_prediction = 10000
    length_freerun_prediction  = 500
        
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
    states = get_reservoir_states(inputs = teacher_input_init, states_init = states_init, weight_input= weight_input_init, weight_reservoir=weight_reservoir_init, leak_rate = LEAK_RATE)
    
    #####################  Train  Output Weight   #########################
    
    weight_output =  train(num_reservoir_node = num_reservoir_node, num_output_node = num_output_node, train_data = train_data_init, states = states, teacher_output = teacher_output_init, weight_output_initial = weight_output_init)
   
    #**********************************************************************************************************************************************#
   
    ############################################################################################################################### 
    ####################   Prediction #############################################################################################
    ###############################################################################################################################
    
    ################# Free Run Predicton######################################
    
    free_run_predict_result = free_run_predict(states = states, num_output_node = num_output_node, weight_output = weight_output, length_train = 101, weight_input= weight_input_init, weight_reservoir=weight_reservoir, length_freerun=length_freerun)
    
    #################  One Step Prediction  ##################################
    one_step_predict_result = one_step_predict(num_output_node= num_output_node, states=states, weight_output = weight_output,  length_train = length_train , length_prediction= length_prediction)
    
    training_prediction = one_step_predict(num_output_node= num_output_node, states=states, weight_output = weight_output, length_train = 101, length_prediction= 50001)
        
    rmse_training = RMSE(one_step_prediction_result, train_data_output)
    rmse_one_step = RMSE(one_step_predict_result, test_data_output)
    print("####################################")
    print("rmse_training")
    print(rmse_training)
    print("************************************")
    print("rmse_prediction")
    print(rmse_one_step)
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
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(free_run_predict_result.T[0,:100], free_run_predict_result.T[1,:100], free_run_predict_result.T[2,:100], color = "red", linewidth = 1)
    ax.set_title(" Free-Running Prediction")
    
    #################  One Step Prediction  ###############################################
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(one_step_predict_result.T[0,:], one_step_predict_result.T[1,:], one_step_predict_result.T[2,:], color = "blue", linewidth = 1 )
    ax.set_title(" One Step Prediction")
    
    
    
    ################# Comparison of Free Running and One Step  ##############################
    
    # ax.plot(teacher_input_init.T[0,length_train:], teacher_input_init.T[1,length_train:], teacher_input_init.T[2,length_train:], color = "magenta", linewidth = 0.1)
    # ax.set_title("real time series")
    # print('one step predict')
    # print(one_step_predict_result)
    plt.figure()
    plt.plot(free_run_predict_result.T[0,:100], lw = 1.0, color ="red", label ="x of free runnning prediction")
    plt.plot(one_step_predict_result.T[0,:100],lw = 1.0,  color = "blue", label ="x of one step prediction")
    plt.title('x of free running prediction and one step prediction')
    plt.legend()
    
    ###################    Teacher Data     #############################################
    
    plt.figure()
    plt.plot(teacher_input_init.T[0, 50101:5201],lw = 1.0,  color = "blue")
    plt.title('real time series')
    # ax.set_xlim([-1.0, 1.0])
    # ax.set_ylim([-1.0, 1.0])
    # ax.set_zlim([-1.0, 1.0])
    
    # ax.plot(states.T[0,:],states.T[1,:],states.T[2,:], color = "red", linewidth = 1)
    
    #######################################################################################
    
    plt.show()
    #******************************************************************************************************************************#
    
    print("OK!")
if __name__=="__main__":
    main()