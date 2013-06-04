from pyBPN import *
import matplotlib.pyplot as plt
from Activation import sigmoid
import numpy as np
from loadData import Data
	

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
#ActSig.view()

d1 = Data()
d1.loadList(pat1)
d2 = Data()
d2.loadList(pat2, 4)

n2 = ANN([3, 5, 4], ActSig)
#n2.displaySynpWt()

arch1 = [3, 4, 1]
n1 = ANN(arch1, ActSig)
#n1.displaySynpWt()

J1 = train(d2, n2, epochs = 500, regLambda = 0.003)
plt.plot(J1)
plt.show()
print 'Accuracy is :' , test(d2, n2), '%'