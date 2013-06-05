import numpy as np

def linRegCost(data, theta, regLambda):
	''' returns the cost and gradient terms of Linear Regression'''

	theta = np.matrix(theta)
	h = np.transpose(data.X) * theta
	theta2 = np.vstack(([[0]],theta[1:]))
	J = 0.5 / data.m * ((np.transpose(h - data.y) * (h - data.y)) + \
			regLambda * np.transpose(theta2) * theta2)
	grad = 1.0 / data.m * (data.X * (h - data.y)) + \
			(regLambda / data.m * theta2)
	return float(J[0][0]), grad

def testLinReg(model, data):
	''' testing Linear regression'''

	h = np.transpose(data.X) * model
	print 'Testing the hypothesis', list(np.transpose(model).flat)
	print 'y\t->\tp'
	for i in range(data.m):
		print('%.2f -> %.2f')%(float(data.y[i]), float(h[i]))
	print 'total cost is :', 0.5 / data.m * float(np.transpose(h - data.y) * (h - data.y))

def predictLinReg(model, X):
	return np.transpose(X) * model