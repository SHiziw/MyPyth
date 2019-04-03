import numpy as np
# scipy. special for the sigmoid function expit() 
import scipy.special
# neural network class definition
class neuralNetwork :
	
	# initialise the neural network
	def _init_(self, inputnodes, hiddennodes, outputnodes, learningrate): 
		# set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		
		# learning rate
		self.lr = learningrate
		
		# link weight matrices, wih and who
		# weights inside the arrays are w_i_j, where link is from the next layer
		# w11 w22
		# w12 w22 etc
		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		# learning rate 
		self. lr = learningrate
		# activation function is the sigmoid function 
		self.activation_function = lambda x: scipy.special.expit(x)

		pass
		
	# train the neural network
	def train():
		pass
		
	# query the neural network
	def query(self, inputs_list):
		# convert inputs list to 2d array 
		inputs = np.array(inputs_list, ndmin= 2).T
		# calculate signals into hidden layer 
		hidden_inputs = np.dot( self.wih, inputs) 
		# calculate the signals emerging from hidden layer 
		hidden_outputs = self.activation_function( hidden_inputs) 
		# calculate signals into final output layer 
		final_inputs = np. dot( self.who, hidden_outputs) 
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function( final_inputs)
		return final_outputs


	