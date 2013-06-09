import numpy as np

##############################################################################
#################### Class for Data that goes for training ###################

class Data:
	'''
	X, y : input Data, desired response
	m : size of Data
	K : # of classes
	n : # of features 
	'''

	def __init__(self):
		self.m , self.n , self.K = 0, 0, 0
		self.X = []
		self.y = []

	def loadList(self, data, numClasses = 2):
		''' loads pattern / data of structure:
		[[[X1, X2... Xn], [y]],
		...
		[[X1, X2... Xn], [y]]]
		cumClasses not significant for regression'''
		
		self.m = len(data)
		self.X = np.transpose(np.matrix([data[i][0] \
								for i in range(self.m)]))
		self.n = self.X.shape[0]
		self.y = np.matrix([data[i][1] for i in range(self.m)])
		self.K = numClasses

	def addBiasRow(self):

		self.X = np.vstack((np.ones((1, self.m)), self.X))
		self.n += 1

	def normalize(self):
		mu = np.mean(self.X, axis = 1)
		sigma = np.std(self.X, axis = 1)
		self.X = (self.X - mu)/ sigma

	def getData(self):
		return X, y
