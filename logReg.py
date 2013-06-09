import numpy as np
from Activation import sigmoid
from optim import gradDesc
from sys import *

def logRegCost(data, theta, regLambda, activation = sigmoid().h):
	''' returns the cost and gradient terms of Linear Regression'''

	theta = np.matrix(theta)
	h = activation(np.transpose(data.X) * theta)
	theta2 = np.vstack(([[0]],theta[1:]))
	J = -1.0 / data.m * (np.transpose(data.y)* np.log(h)  + \
			(1 - np.transpose(data.y)) * np.log(1 - h)) + \
			0.5 * regLambda / data.m * np.transpose(theta2) * theta2
	grad = 1.0 / data.m * (data.X * (h - data.y)) + \
			(regLambda / data.m * theta2)
	return float(J[0][0]), grad

def testLogReg(model, data, activation = sigmoid().h):
	''' testing Linear regression'''

	h = predictLogReg(model, data.X, activation)
	print 'Testing the hypothesis', list(np.transpose(model).flat)
	print 'y\t->\tp'
	for i in range(data.m):
		print('%d -> %d')%(int(data.y[i]), int(h[i]))
	cost, _ = logRegCost(data, model, 0.0, activation)
	print 'total cost is :', cost

def predictLogReg(model, X, activation = sigmoid().h):
	return activation(np.transpose(X) * model) > 0.5

def trainOneVsAllGD(data, act = sigmoid().h, epochs = 10000, lr = 0.5):
	data.addBiasRow()
	theta_init = np.matrix(np.zeros((data.n, 1)))
	from functools import partial
	from copy import deepcopy
	model= np.matrix(np.zeros((data.n, data.K)))
	J = np.matrix(np.zeros((data.K, epochs)))
	
	for k in range(data.K):
		d2 = deepcopy(data)
		d2.y = (data.y == k)
		cost = partial(logRegCost,data = d2,theta = theta_init, \
				regLambda = 0.001, activation = act)
		J[k, :], model[:, k] = gradDesc(cost, theta_init, epochs, lr)
	return model, J

def predictMultiple(model, X, act = sigmoid().h):
	return np.argmax(act(np.transpose(X) * model), axis = 1)