import numpy as np
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

		pass
		
	# train the neural network
	def train():
		pass
		
	# query the neural network
	def query():
		pass