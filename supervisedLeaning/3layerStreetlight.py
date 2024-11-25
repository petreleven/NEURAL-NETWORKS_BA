import numpy as np

streetlights = np.array(
    [
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
)
outputs = np.array(
    [
        [1],
        [1],
        [0],
        [0]
    ]
)
number_of_outputs = 1
number_of_hidden_layer_nodes = 4
number_of_inputs = 3

weights_0_1 = 2 * np.random.random((3, 4)) - 1
weights_0_2 = 2 * np.random.random((4, 1)) - 1
lr = 0.01

def relu(x):
    #x = -1
    return (x > 0) * x
    # 1
    # 0

def relu_derivative(x):
    return x > 0


#Teach our network 100 times
for i in range(4000):
    total_error = 0
    for j in range(len(streetlights)):
        layer_1 = relu(np.dot(
                        streetlights[j : j+1], weights_0_1
                        ))

        layer_2 = np.dot(layer_1, weights_0_2)#[0.4]
        error =   np.sum((layer_2 - outputs[j : j+1]) ** 2)# msq
        total_error += error
        #get the deltas so that we update weights
        delta_layer_2 = layer_2 - outputs[j : j+1]
        delta_layer_1 = np.dot(delta_layer_2, weights_0_2.T) * relu_derivative(layer_1)
        weight_delta_2 = np.dot(layer_1.T , delta_layer_2)
        weight_delta_1 = np.dot(streetlights[j : j+1].T , delta_layer_1)
        #update weights
        weights_0_1 -= weight_delta_1 * lr
        weights_0_2 -= weight_delta_2 * lr

    print(f"Error is {total_error}")
