from pyBPN import *
from optim import *
import matplotlib.pyplot as plt
from Activation import sigmoid
import numpy as np
from loadData import Data
from functools import partial
from math import sin, cos


##############################################################
##### demonstration of GradDesc for general optimization #####

# x = np.matrix([0.5, 0.7])
# def f(x):
# 	return np.sin(x) + 2 * x, np.cos(x) + 2
# f = partial(f, x)
# J, x_opt = gradDesc(f , init_x = x, maxEpochs = 100, lr = 0.1)
# print x_opt
#plt.plot(J)
#plt.show()

#############################################################

pat2 = [[[0, 0, 0], [0]],
	[[0, 0, 1], [1]],
	[[0, 1, 0], [1]],
	[[0, 1, 1], [0]],
	[[1, 0, 0], [2]],
	[[1, 0, 1], [3]],
	[[1, 1, 0], [3]],
	[[1, 1, 1], [2]],
	]

pat1 = [[[0, 0, 0], [0]],
	[[0, 0, 1], [1]],
	[[0, 1, 0], [1]],
	[[0, 1, 1], [0]],
	[[1, 0, 0], [1]],
	[[1, 0, 1], [0]],
	[[1, 1, 0], [0]],
	[[1, 1, 1], [1]],
	]

ActSig = sigmoid(4)
# ActSig.view()

d1 = Data()
d1.loadList(pat1)
d2 = Data()
d2.loadList(pat2, 4)

n2 = ANN([3, 3, 4], ActSig)
# n2.displaySynpWt()

arch1 = [3, 4, 1]
n1 = ANN(arch1, ActSig)
# n1.displaySynpWt()

##########################################################
########## Training With Gradient DescentPortion #########

cost = partial(n1.bpCost, data = d1, regLambda = 0.003)
J, Wt = gradDesc(cost, init_x = n1.W, maxEpochs = 500, lr = 0.8)
n1.W = Wt

##########################################################

plt.plot(J)
plt.show()
print 'Accuracy is :' , n1.test(d1), '%'