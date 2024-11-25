import numpy as np
from keras.datasets import mnist
from supervisedLeaning.streetlights2 import hidden_size
#import matplotlib.pyplot as plt
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
print(train_data.shape)
print(train_labels.shape)

#reduce our training to 1000images
train_data = (train_data[0:1000]) / 255
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
