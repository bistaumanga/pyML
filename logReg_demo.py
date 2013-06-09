from optim import *
import matplotlib.pyplot as plt
from Activation import sigmoid
import numpy as np
from loadData import Data
from functools import partial
from logReg import *

# and function of 3 variables
pat1 = [[[0, 0, 0], [0]],
	[[0, 0, 1], [1]],
	[[0, 1, 0], [1]],
	[[0, 1, 1], [1]],
	[[1, 0, 0], [2]],
	[[1, 0, 1], [3]],
	[[1, 1, 0], [3]],
	[[1, 1, 1], [3]],
	]

d1 = Data()
d1.loadList(pat1, numClasses = 4)
#print d1.y
act = sigmoid().h # our activation function is simgmoid
model, J = trainOneVsAllGD(d1, act,epochs = 5000, lr = 0.25)
#print d1.y
plt.plot(np.transpose(J))
plt.show()

print predictMultiple(model, d1.X, act)
	


# d1.addBiasRow()
# theta_init = np.matrix(np.zeros((d1.n, 1)))

# 
# cost = partial(logRegCost,data = d1,theta = theta_init, \
# 			regLambda = 0.001, activation = act)

# J, theta = gradDesc(cost, theta_init, 1000, 2.5)
# print 'model is :', list(np.transpose(theta).flat)

# plt.plot(J)
# plt.show()

# testLogReg(theta, d1, act)