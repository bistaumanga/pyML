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

	def __init__(self, pattern, numClasses = 2):

		self.m = len(pattern)
		self.X = np.transpose(np.matrix([pattern[i][0] \
								for i in range(self.m)]))
		self.n = self.X.shape[0]
		self.y = np.matrix([pattern[i][1] for i in range(self.m)])
		self.K = numClasses

		# convert y to matrix if number of classes is grater than 2
		if self.K > 2:
			temp = self.y
			self.y = np.zeros(shape = (self.m, self.K))
			for i in range(self.K):
				self.y[:, i] = (np.transpose(temp) == i)

	def getData(self):
		return X, y