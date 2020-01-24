import numpy as np

    ###################################################################
    ##############  Get Reservoir State   #############################
    ###################################################################
# @jit
def get_reservoir_states(inputs, states_init, weight_input, weight_reservoir, leak_rate, weight_return, output_return = None, input_bias = None):
    states = states_init
    # if output_return is None:
    #     for i in range(1, len(inputs),1):
    #         states[i] = (1-leak_rate)* np.array([states[i-1]]) + leak_rate *np.tanh(np.dot(np.array([inputs[i-1]]), weight_input) +np.dot( np.array([states[i-1]]), weight_reservoir))
    #     # print(states)
    #     # print("shape of states")
    #     # print(states.shape)
    #     return states
    # else:
    #     for i in range(1, len(inputs),1):
    #         states[i] = (1-leak_rate)* np.array([states[i-1]]) + leak_rate *np.tanh(np.dot(np.array([inputs[i-1]]), weight_input) +np.dot( np.array([states[i-1]]), weight_reservoir) )
    #     return states
    if input_bias is None:
        for i in range(1, len(inputs),1):
            states[i] = np.tanh((1-leak_rate)* np.array([states[i-1]]) + leak_rate * np.dot(np.array([inputs[i-1]]), weight_input) +np.dot( np.array([states[i-1]]), weight_reservoir))
    else: 
        inputs =  np.column_stack((np.ones((inputs.shape[0],1)),inputs))
        for i in range(1, len(inputs),1):
            states[i] =  np.tanh((1-leak_rate)* np.array([states[i-1]])+  leak_rate * np.dot(np.array([inputs[i-1]]), weight_input) +np.dot( np.array([states[i-1]]), weight_reservoir))
    return states
    ######################################################################
    #####################   Train  #######################################
    ######################################################################
# @jit
def train(num_reservoir_node, num_output_node, train_data_output, states, weight_output_initial, LAMBDA = 0.00221):

    ################# Set Length of Training #########
    
    Length_Training = len(train_data_output)
    #############Initialize Variables#################
    re = 0
    kp = np.zeros((num_reservoir_node,1))
    e = np.zeros((1,num_output_node))
    P = 1/LAMBDA * np.identity(num_reservoir_node)
    weight_output = weight_output_initial
    #*******************************************************************#
    ###########  Training Using Recursive Least Square ##################
    for j in range(1):
        for i in range(Length_Training):
            
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
def free_run_predict(states, num_output_node, weight_output, weight_input, weight_reservoir, length_freerun, leak_rate, input_bias = None):
    state_free_run = states[0]
    free_run_predict_result = np.zeros((length_freerun, num_output_node))
    if input_bias is None:
      for i in range(length_freerun):
            free_run_predict_result[i] = np.array([state_free_run]) @ weight_output
            state_free_run = (1-leak_rate)* np.array([state_free_run]) + leak_rate *np.tanh(np.dot(np.array([free_run_predict_result[i]]), weight_input) +np.dot( np.array([state_free_run]), weight_reservoir))
    else: 
        for i in range(length_freerun):
            print("shape of state free run")
            print(state_free_run.shape)
            print("state free run")
            print(state_free_run)
            free_run_predict_result[i] =state_free_run @ weight_output
            inputs =  np.concatenate([np.ones((1,1)),np.array([free_run_predict_result[i]])], axis =1)
            state_free_run = (1-leak_rate)* np.array([state_free_run]) + leak_rate *np.tanh(np.dot(inputs, weight_input) +np.dot( np.array([state_free_run]), weight_reservoir))
      
    return free_run_predict_result
    

##########################################################################
###############  Calculation of Eroor ####################################
##########################################################################

def RMSE(outputs, teacher):
    error_vector = outputs - teacher
    summation = np.sum(np.square(error_vector))/len(error_vector)
    RMSE_ERROR = np.sqrt(summation)
    return RMSE_ERROR