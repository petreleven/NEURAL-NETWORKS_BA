import numpy as np
from keras.datasets import mnist
#import matplotlib.pyplot as plt
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print(train_data.shape)
print(train_labels.shape)

#reduce our training to 1000images
train_data = (train_data[0:1000]) / 255
train_data = train_data.reshape(1000, 784)


#print(train_data.shape)
#print(train_data[0])
train_labels = train_labels[0:1000]
print(train_labels[0])
target_labels = np.zeros((1000, 10))
for index,value in enumerate(train_labels):
    #print(index , " ", value)
    target_labels[index][value] = 1
    #target_labels[0][6] = 1

print( train_labels[6])
print( target_labels[6])

hidden_layer_size = 10
output_layer_size = 10
weights_1 = 2 * np.random.random((784, hidden_layer_size)) - 1
weights_2 = 2 * np.random.random((hidden_layer_size,  output_layer_size)) - 1
lr = 0.1

def relu(x):
    return (x > 0) * x
def relu2deriv(output):
    return output > 0

for i in range(100):
    error = 0
    for j in range(len(train_data)):
        #input layer
        layer_1 = train_data[j : j+1]
        #layer_2 is hidden layer
        layer_2 = relu(np.dot(layer_1, weights_1))  #*****
        #output layer (0...9)
        layer_3 = np.dot(layer_2, weights_2)
        #WE NEED TO GET THE ERROR
        #error = (pred - target) ** 2
        error += np.sum(layer_3 - target_labels[j:j+1]) ** 2
        #WE NEED DELTAS SO WE UPDATE OUR WEIGHTS
        #delta = pred - target
        delta_l3 = layer_3 - target_labels[j:j+1]
        delta_l2 = np.dot(delta_l3, weights_2.T) * relu2deriv(layer_2)
        #WE NEED WEIGHTS DELTAS
        #weight_delta = input * delta
        weight_deltas_l3 = np.dot(layer_2.T, delta_l3 )
        weights_deltas_l2 = np.dot(layer_1.T, delta_l2)
