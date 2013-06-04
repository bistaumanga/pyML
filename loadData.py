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

	def loadList(self, pattern, numClasses = 2):
		''' loads pattern of structure:
		[[[X1, X2... Xn], [y]],
		...
		[[X1, X2... Xn], [y]]]'''
		
		self.m = len(pattern)
		self.X = np.transpose(np.matrix([pattern[i][0] \
								for i in range(self.m)]))
		self.n = self.X.shape[0]
		self.y = np.matrix([pattern[i][1] for i in range(self.m)])
		self.K = numClasses

	def loadXy(self, X, y, numClasses = 2):
		''' loads list or matrixof X and y of the form:
		X = [[X1, X2, ...... Xn]
		[X1, X2, ....... Xn]
		.
		.
		[X1, X2, ....... Xn]], with each row with each training input

		y = [[y], [y]..... [y]], with each row has one element of target output'''

		self.m = len(X)
		self.X = mp.transpose(np.matrix(X))
		self.y = np.matrix(y)
		self.K = numClasses
		self.n = self.X.shape[0]

	def getData(self):
		return X, y