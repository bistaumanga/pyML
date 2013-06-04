import numpy as np
import math
from Activation import sigmoid


##########################################################################
################### ANN Class ############################################

class ANN:

	def __init__(self, arch, activation = sigmoid()):

		# define activation and derivative of activation functions
		self.h, self.dh = activation.h, activation.dh

		# Architecture, i.e. number of units/neurons in each layer
		self.arch = arch   
		# Number of Layers in the Network
		self.L = len(arch)
		# List of Weight Matrices, initialize each with small random numbers
		self.W = []
		for l in range(1, self.L):
			#Wl = np.matrix([[gauss(0, math.sqrt(6.0 / (arch[l] * arch[l - 1]))) \
			#	for i in xrange(arch[l])] for j in range(arch[l - 1] + 1)])
			epsilon = math.sqrt(6.0 / (arch[l] * arch[l - 1]))
			Wl = np.random.random((arch[l - 1] + 1, arch[l])) - 0.5
			self.W.append(epsilon * Wl)

	def displaySynpWt(self):
		''' This displays the matrix of synaptic Weights iniyialized'''
		for l in range(self.L - 1):
			w = self.W[l]
			print ('\nWeights between layer %d and %d :') % (l, l+1)
			print w

	def predict(self, x):
		''' Predicts the output for given pattern x'''
		a = [np.matrix(np.zeros(shape = (self.arch[l], 1))) \
				for l in range(self.L)]
		a[0] = np.vstack(([1], np.transpose(np.matrix(x)))) # adding bias

		for l in range(1, self.L):
			z = self.h(np.transpose(self.W[l - 1]) * a[l - 1])
			a[l] = np.vstack(([1], z))

		if self.arch[-1] > 2:
			h = np.argmax(a[self.L - 1][1:])
		else:
			h = (a[self.L - 1][1] >= 0.5)		
		return h


############################################################################
################### Training Portion #######################################
############################################################################
############################################################################
################## BackProp Cost Function ##################################

def bpCost(data, net, regLambda):
	# initialization activation units, delta terms and induced vector field
	a = [0] * (net.L)
	delta = [0] * (net.L)
	Delta = []
	for l in range(net.L - 1):
		Delta.append(np.zeros_like(net.W[l]))
	z = [0] * (net.L)
	D = [0] * (net.L - 1)

	J_nonReg = 0.0 # cost without regularization
	J_reg = 0.0
	for l in range(net.L - 1):
		J_reg += np.sum(np.multiply(net.W[l][1:], net.W[l][1:]))

	for i in range(data.m):

		# feed-Forward
		xi = data.X[:, i]
		yi = np.transpose(np.matrix(data.y[i]))
		a[0] = np.vstack(([1], xi)) # adding bias
		for l in range(1, net.L):
			z[l] = np.transpose(net.W[l - 1]) * a[l - 1]
			#adding bias
			a[l] = np.vstack(([1], net.h(z[l])))
	
		# removing added bias in activation units of output layer
		a[net.L - 1] = a[net.L - 1][1:] 

		# Backpropagation
		# Delta terms for output layer 
		h = a[net.L - 1]
		delta[net.L - 1] = yi - h

		# cost function  updates
		J_nonReg += (np.sum(np.multiply(yi , np.log(h)) + \
						np.multiply(1 - yi, np.log(1 - h))))

		# delta terms for hidden layers
		for l in range(net.L - 2 , 0, -1):
			delta[l] = np.multiply(net.dh(z[l]), net.W[l][1:] * delta[l + 1])

		# Delta terms for each layers
		for l in range(net.L - 1): 
			Delta[l] = Delta[l] +  a[l] * np.transpose(delta[l + 1])

	for l in range(net.L - 1):
		D[l] = 1.0 / data.m * Delta[l]
		D[l][1:] += (regLambda / data.m * net.W[l][1:]) 

	#sum up regularized and non rgularized cost func terms
	J = - 1.0 / data.m * J_nonReg + regLambda / (2.0 * data.m) * J_reg
	return J, D


############################################################################
############## function for training by optimization #######################

def train(data, net, epochs = 25000, regLambda = 0.0):

	J = []
	for k in range(epochs):
		# minimizaing cost function using gradient descent
		j, D = bpCost(data, net, regLambda)
		J.append(j)
		temp = net.W
		for l in range(net.L - 1):
			net.W[l] += (0.8* D[l])
	return J

	########################################################################
	## class for training., lr, momentum and adv optimization


############################################################################
################## Function to test the trainned network ###################

def test(data, net):
	sum = 0.0
	for i in range(data.m):
		xi = np.transpose(data.X[:, i])
		p = net.predict(xi)
		yi = data.y[i]

		print yi, p
		if data.K == 2:
			sum += (p == yi )
		else:
			sum += (p == np.argmax(yi))

	return float(sum / data.m * 100.0)