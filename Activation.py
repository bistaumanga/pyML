import numpy as np
import matplotlib.pyplot as plt

#############################################################################
######### Sigmoid Activation Class #######################################
class sigmoid:
	def __init__(self, k = 1):
		if k <= 0:
			k = 1
		self.k = k

	def h(self, a):
		a = np.matrix(a)
		return 1.0 / (1 + np.exp(-self.k * a))

	def dh(self, a):
		a = np.matrix(a)
		return self.k * np.multiply(self.h(a), 1 - self.h(a))

	def view(self, x = np.arange(-10, 10, 0.1)):
		x = np.array(x)
		hx = np.array(self.h(x))[0]
		dhx = np.array(self.dh(x))[0]
		plt.plot(x, hx, 'r', x, dhx, 'b')
		plt.show()

##############################################################################
############### hyperbolic tan activation Class ###########################

class hypTan:
	def __init__(self, alpha = 1.0, beta = 1.0):
		self.alpha = float(alpha)
		self.beta = float(beta)

	def h(self, a):
		a = np.matrix(a)
		return self.alpha * np.tanh(self.beta * a)

	def dh(self, a):
		a = np.matrix(a)
		return (float(self.beta)/self.alpha) * \
				np.multiply(self.alpha - self.h(a), self.alpha + self.h(a))

	def view(self,  x = np.arange(-10, 10, 0.1)):
		x = np.array(x)
		hx = np.array(self.h(x))[0]
		dhx = np.array(self.dh(x))[0]
		plt.plot(x, hx, 'r', x, dhx, 'b')
		plt.show()