import numpy as np
import matplotlib.pyplot as plot
import sys
from neuralNetworks import neuralNetwork

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
print(n.query([1.0, 0.5, -1.5]))

data_file = open("./mnist_dataset/mnist_train_100.csv", 'r') 
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])
all_values = data_list[1].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plot.imshow(image_array,cmap='Greys', interpolation='none')
plot.show()