# import numpy as np
from scipy import linalg
import autograd.numpy as np # Numpy用の薄いラッパ
from autograd import grad

class ReservoirNetWork:

    def __init__(self, inputs, teacher, num_input_nodes, num_reservoir_nodes, num_output_nodes, leak_rate=0.1, activator=np.tanh):
        self.inputs = inputs # 学習で使う入力
        self.teacher = teacher
        self.log_reservoir_nodes = np.array([np.zeros(num_reservoir_nodes)]) # reservoir層のノードの状態を記録

        # init weights
        self.weights_input = self._generate_variational_weights(num_input_nodes, num_reservoir_nodes)
        self.weights_reservoir = self._generate_reservoir_weights(num_reservoir_nodes)
        self.weights_output = np.zeros([num_reservoir_nodes, num_output_nodes])

        # それぞれの層のノードの数
        self.num_input_nodes = num_input_nodes
        self.num_reservoir_nodes = num_reservoir_nodes
        self.num_output_nodes = num_output_nodes

        self.leak_rate = leak_rate # 漏れ率
        self.activator = activator # 活性化関数

    # reservoir層のノードの次の状態を取得
    def _get_next_reservoir_nodes(self, input, current_state):
        next_state = (1 - self.leak_rate) * current_state
        next_state += self.leak_rate * ((input) @ self.weights_input
            + current_state @ self.weights_reservoir)
        return self.activator(next_state)

    # 出力層の重みを更新
    def _update_weights_output(self, lambda0):
        # Ridge Regression
        E_lambda0 = np.identity(self.num_reservoir_nodes) * lambda0 # lambda0
        inv_x = np.linalg.inv(self.log_reservoir_nodes.T @ self.log_reservoir_nodes + E_lambda0)
        # update weights of output layer
        self.weights_output = (inv_x @ self.log_reservoir_nodes.T) @ self.inputs

    # 学習する(offline_training)
    def offline_training(self, lambda0=0.1):
        for input in self.inputs:
            current_state = np.array(self.log_reservoir_nodes[-1])
            self.log_reservoir_nodes = np.append(self.log_reservoir_nodes,
                [self._get_next_reservoir_nodes(input, current_state)], axis=0)
        self.log_reservoir_nodes = self.log_reservoir_nodes[1:]
        self._update_weights_output(lambda0)
        print("result of offline training")
        print(self.weights_output)
    
    def online_training(self, eta = 0.001,lambda0 = 0.1 ):
        output_weight = np.zeros(self.num_reservoir_nodes*self.num_output_nodes).reshape(self.num_reservoir_nodes,self.num_output_nodes)
        states = self.log_reservoir_nodes
        teacher = self.teacher
    
        def loss_function(output_weight):
            s = np.zeros(3)            
            for i in range(len(self.inputs)-100):    
                s += np.square(teacher[i+100] - states[i+100] @ output_weight)           ##初期の状態は偏移中なのでカットすべき
            loss = 1/2*(np.sum(s)/(len(states)-100)+np.sum(np.square(output_weight)))    
            return loss
    
        loss_grad = grad(loss_function)
        
        def error(output_weight):
            s = np.zeros(3)            
            for i in range(len(self.inputs)-100):    
                s += np.square(teacher[i+100] - states[i+100] @ output_weight)           ##初期の状態は偏移中なのでカットすべき
            error = 1/2*np.sum(s)/(len(states)-100)
            return error
        
        for i in range(1000):    
            output_weight = output_weight - eta* loss_grad(output_weight)
            print(error(output_weight))        
        self.weights_output = output_weight

    # 学習で得られた重みを基に訓練データを学習できているかを出力
    def get_train_result(self):
        outputs = []
        reservoir_nodes = np.zeros(self.num_reservoir_nodes)
        for input in self.inputs:
            reservoir_nodes = self._get_next_reservoir_nodes(input, reservoir_nodes)
            outputs.append(self.get_output(reservoir_nodes))
        return outputs

    # 予測する
    def predict(self, length_of_sequence, lambda0=0.01):
        predicted_outputs = [self.inputs[-1]] # 最初にひとつ目だけ学習データの最後のデータを使う
        reservoir_nodes = self.log_reservoir_nodes[-1] # 訓練の結果得た最後の内部状態を取得
        for _ in range(length_of_sequence):
            reservoir_nodes = self._get_next_reservoir_nodes(predicted_outputs[-1], reservoir_nodes)
            predicted_outputs.append(self.get_output(reservoir_nodes))
        return predicted_outputs[1:] # 最初に使用した学習データの最後のデータを外して返す

    # get output of current state
    def get_output(self, reservoir_nodes):
        # return self.activator(reservoir_nodes @ self.weights_output) 修正前
        return reservoir_nodes @ self.weights_output # 修正後

    #############################
    ##### private method ########
    #############################

    # 重みを0.1か-0.1で初期化したものを返す
    def _generate_variational_weights(self, num_pre_nodes, num_post_nodes):
        return (np.random.normal(0, 1, num_pre_nodes * num_post_nodes).reshape([num_pre_nodes, num_post_nodes]) * 2 - 1) * 0.1

    # Reservoir層の重みを初期化
    def _generate_reservoir_weights(self, num_nodes):
        matrix_nonzero1 = np.random.randint(0,2,(num_nodes,num_nodes))
        matrix_nonzero2 = np.random.randint(0,2,(num_nodes,num_nodes))
        weight_normal = np.random.normal(0, 1, num_nodes * num_nodes).reshape([num_nodes, num_nodes])
        weights = matrix_nonzero1 * matrix_nonzero2 * weight_normal
        spectral_radius = max(abs(linalg.eigvals(weights)))
        return weights / spectral_radius