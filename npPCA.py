import numpy as np
import matplotlib.pyplot as plt
'''
Performs the Principal Coponent analysis of the Matrix X
Matrix must be n * m dimensions
where n is # features
m is # examples
'''
def PCA(X, varRetained = 0.95, show = False):

	# Compute Covariance Matrix Sigma
	(n, m) = X.shape
	Sigma = 1.0 / m * X * np.transpose(X)
	# Compute eigenvectors and eigenvalues of Sigma
	U, s, V = np.linalg.svd(Sigma, full_matrices = True)

	# compute the value k: number of minumum features that 
	# retains the given variance
	sTot = np.sum(s)
	var_i = np.array([np.sum(s[: i + 1]) / \
						sTot * 100.0 for i in range(n)])
	k = len(var_i[var_i < (varRetained * 100)])
	print '%.2f %% variance retained in %d dimensions' \
						% (var_i[k], k)
	
	# plot the variance plot
	if show:
		plt.plot(var_i)
		plt.xlabel('Number of Features')
		plt.ylabel(' Percentage Variance retained')
		plt.show()

	# compute the reduced dimensional features by projction
	U_reduced = U[:, : k]
	Z = np.transpose(U_reduced) * X

	return Z, U_reduced
