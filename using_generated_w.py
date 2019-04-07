import numpy as np
import scipy.special
# load the generated weight matrix CSV file into a list

weight_file = open("mnist_dataset/weight_matrix.csv", 'r')
weight_list = weight_file.readlines()
weight_file.close()

# reading the number of input, hidden and output nodes
first_line = weight_list[0].split(',')
#print(first_line)
input_nodes = int(first_line[0])
hidden_nodes = int(first_line[1])
output_nodes = int(first_line[2])
# learning rate
learning_rate = float(first_line[3])

wih = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
who = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))
# 生成容器装已经训练好的矩阵
for e in range(1, hidden_nodes):
    #避开第一行，从之后的 hidden_nodes 行开始
    all_values = weight_list[e].split(',')
    wih[e - 1] = np.asfarray(all_values[0:])
    pass
for e in range(1+hidden_nodes, 1 + hidden_nodes + output_nodes):
    # 再之后的 output_nodes 行
    all_values = weight_list[e].split(',')
    who[e - 1 - hidden_nodes] = np.asfarray(all_values[0:])
    pass

# print('in:', input_nodes, 'hid:', hidden_nodes, 'out:', output_nodes, 'rate:', learning_rate,'\n')
# print(w_ho)

class after_trained :
    # initialise the neural network
	def __init__(self): 
		# set number of nodes in each input, hidden, output layer
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes
		
		# learning rate
		self.lr = learning_rate
		
		# link weight matrices, wih and who
		# weights inside the arrays are w_i_j, where link is from the next layer
		# w11 w22
		# w12 w22 etc
		self.wih = wih
		self.who = who

		# activation function is the sigmoid function 
		self.activation_function = lambda x: scipy.special.expit(x)

		pass
		

		
	# query the neural network
	def query(self, inputs_list):
		# convert inputs list to 2d array 
		inputs = np.array(inputs_list, ndmin= 2).T
		# calculate signals into hidden layer 
		hidden_inputs = np.dot(self.wih, inputs) 
		# calculate the signals emerging from hidden layer 
		hidden_outputs = self.activation_function( hidden_inputs) 
		# calculate signals into final output layer 
		final_inputs = np.dot( self.who, hidden_outputs) 
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function( final_inputs)
		return final_outputs

