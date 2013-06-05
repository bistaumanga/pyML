from optim import *
import matplotlib.pyplot as plt
from Activation import sigmoid
import numpy as np
from loadData import Data
from functools import partial
from linReg import *

# data closer to y = x0 + x1 + 2* x2
data1 = [[[1, 1], [4.2]], [[1, 2], [6.0]], [[2, 1], [5.8]],
[[2, 2], [7.2]], [[2, 3], [8.5]], [[3, 3], [9.2]],
[[1, 3], [8.1]], [[3, 1], [6.1]], [[4, 1], [7.1]],
[[1, 4], [10.3]], [[4, 2], [9.5]], [[2, 4], [10.8]]]

d1 = Data()
d1.loadList(data1)

d1.addBiasRow()
print d1.X
theta_init = np.matrix(np.zeros((d1.n, 1)))

cost = partial(linRegCost,data = d1,theta = theta_init, regLambda = 0.001)

J, theta = gradDesc(cost, theta_init, 500, 0.1)
print 'model is :', list(np.transpose(theta).flat)

# test the trained model with data
testLinReg(theta, d1)

# test the model (1, 1, 2)
testLinReg(np.matrix([[1], [1], [2]]), d1)
predictLinReg(theta, d1.X)

# plot the cost vs epochs 
fig = plt.figure(figsize = (8, 5), facecolor =  'w')
fig.add_subplot(111)
plt.plot(J)
plt.title('Plot of Cost function')
plt.xlabel('epochs')
plt.show()