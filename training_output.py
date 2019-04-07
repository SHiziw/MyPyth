import sys
from neuralNetworks import neuralNetwork
import numpy
import csv
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 5

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
#python2可以用file替代open

with open("mnist_dataset/weight_matrix.csv","w", newline='') as csvfile: 
    writer = csv.writer(csvfile)

    #先写入columns_name
    writer.writerow([input_nodes, hidden_nodes, output_nodes, learning_rate, '0'])
    #写入多行用writerows
    for e in range(int(hidden_nodes)):
        writer.writerow(n.wih[e])
        pass
    for c in range(int(output_nodes)):
        writer.writerow(n.who[c])
        pass