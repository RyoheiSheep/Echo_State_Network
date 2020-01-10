import autograd.numpy as np
from autograd import grad
from input_generator import InputGenerator
from reservoir_computing import ReservoirNetWork
import matplotlib.pyplot as plt

T = 20
RATIO_TRAIN = 0.6
dt = T*0.001
AMPLITUDE = 0.9
LEAK_RATE=0.02
NUM_INPUT_NODES = 1
NUM_RESERVOIR_NODES = 10
NUM_OUTPUT_NODES = 1
NUM_TIME_STEPS = int(T/dt)

# example of activator
# def ReLU(x):
#     return np.maximum(0, x)

def main():
    i_gen = InputGenerator(0, T, NUM_TIME_STEPS)
    data = i_gen.generate_logistic()
    num_train = int(len(data) * RATIO_TRAIN)
    train_data = data[:num_train]
    print("number of training is"+ str(num_train))
    print(dt)
    model = ReservoirNetWork(inputs=train_data,
        teacher = train_data,
        num_input_nodes=NUM_INPUT_NODES, 
        num_reservoir_nodes=NUM_RESERVOIR_NODES, 
        num_output_nodes=NUM_OUTPUT_NODES, 
        leak_rate=LEAK_RATE)

    model.offline_training() #オフライン学習で出力重みを初期化
    model.online_training()#(オンライン学習で)重み更新
    
    train_result = model.get_train_result() # 訓練の結果を取得
    num_predict = int(len(data[num_train:]))
    predict_result = model.predict(length_of_sequence =num_predict)
    

    t = np.linspace(0, T, NUM_TIME_STEPS)
    ## plot
    plt.plot(t, data, label="inputs")
    plt.plot(t[:num_train], train_result, label="trained")
    plt.plot(t[num_train:], predict_result, label="predicted")
    plt.axvline(x=int(T * RATIO_TRAIN), label="end of train", color="red") # border of train and prediction
    plt.legend()
    plt.title("Echo State Network Logistic Prediction" + "//Number of Reservoir Nodes " + str(NUM_RESERVOIR_NODES))
    plt.xlabel("time")
    plt.ylabel("y(t)")
    plt.show()

if __name__=="__main__":
    main()