import numpy as np
from Activation import sigmoid

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