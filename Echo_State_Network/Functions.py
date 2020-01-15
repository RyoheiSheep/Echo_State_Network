import numpy as np

    ###################################################################
    ##############  Get Reservoir State   #############################
    ###################################################################
# @jit
def get_reservoir_states(inputs, states_init, weight_input, weight_reservoir, leak_rate):
    states = states_init
    for i in range(1, len(inputs),1):
        states[i] = (1-leak_rate)* np.array([states[i-1]]) + leak_rate *np.tanh(np.dot(np.array([inputs[i-1]]), weight_input) +np.dot( np.array([states[i]]), weight_reservoir))
    # print(states)
    # print("shape of states")
    # print(states.shape)
    return states
    ######################################################################
    #####################   Train  #######################################
    ######################################################################
# @jit
def train(num_reservoir_node, num_output_node, train_data_output, states, weight_output_initial,wash_time, LAMBDA = 0.00221):
    
    ################# Set Length of Training #########
    
    Length_Training = len(train_data_output)
    #############Initialize Variables#################
    states = states[wash_time:]
    
    re = 0
    kp = np.zeros((num_reservoir_node,1))
    e = np.zeros((1,num_output_node))
    P = 1/LAMBDA * np.identity(num_reservoir_node)
    weight_output = weight_output_initial
    #*******************************************************************#
    ###########  Training Using Recursive Least Square ##################
    for j in range(2):
        for i in range(len(train_data_output)):
            
            re = 1 + np.dot(np.dot(np.array([states[i]]), P), np.array([states[i]]).T)
            kp = np.dot(P,np.array([states[i]]).T )/ re
            e = np.array([train_data_output[i]]) -  np.dot(np.array([states[i]]),weight_output)
            weight_output = weight_output + np.dot(kp, e)
            P = P  -( np.dot(P,np.dot (np.array([states[i]]).T,np.dot(np.array([states[i]]), P))) / re)
            
    return weight_output

######################################################################
##############   prediction against the teacher output  ##############
######################################################################
# @jit
def one_step_predict(num_output_node, states, weight_output, length_prediction):
    one_step_predict_result = np.zeros((length_prediction,num_output_node))
    # print("length of states")
    # print(len(states))
    # print("length of teacher data")
    # print(len(teacher_input))
    for i in range(length_prediction):
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