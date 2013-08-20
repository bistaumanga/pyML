''' Gaussian Mixture Model
A probabilistic clustring method
Expectation Maximization Algorithm'''
import numpy as np

def p(x, mu, sigma):
	n = len(x)
	import math
	return np.linalg.det(sigma) ** - 0.5 * (2 * math.pi) ** (-n/2) * \
					np.exp( -0.5 * np.dot(x - mu , \
						np.dot(np.linalg.inv(sigma) , x - mu)))

def EM(X, K, max_iters = 100, show_progress = None):

	m, n = X.shape
	mu = X[np.random.choice(np.arange(m), K), :]
	Sigma, w = [np.eye(n)] * K, [1./K] * K #initialize SIgma and w
	
	R = np.zeros((m, K)) # responsibility Matrix
	curr_iter, oldLog, in_optima_for = 0, 1000, 0
	costs = []
	while curr_iter < max_iters:
		# E - Step
		for i in range(m):
			for k in range(K):
				R[i, k] = w[k] * p(X[i], mu[k], Sigma[k])	
		R = (R.T / np.sum(R, axis = 1)).T
		div = np.sum(R, axis = 0)

		# M Step
		w = 1. / m * div
		for k in range(K):
			mu[k] = 1. / div[k] * np.sum(R[:, k] * X.T, axis = 1).T
			#print mu[k]
			x_mu = np.matrix(X - mu[k])
			Sigma[k] = np.zeros((n, n))
			for i in range(m):
				Sigma[k] += R[i, k] * x_mu[i].T * x_mu[i]
			Sigma[k] *= 1./ div[k]
		

		# calculate log Likelihood
		L = 0.0
		for i in range(m):
			temp = 0.0
			for k in range(K):
				temp += w[k] * p(X[i], mu[k], Sigma[k])
			L += np.log(temp)
		costs.append(L)
		if abs(L - oldLog) < 0.001 : break
		oldLog = L
		curr_iter += 1
		if show_progress != None : show_progress(X, mu, Sigma)
	print 'Converged in ', curr_iter, ' Iterations !!'
	return mu, Sigma, w, costs

def test(x, mu, sigma, w):
	probs = np.array([w[k] * p(np.array(x), mu[k], sigma[k]) for k in range(3)])
	# result is prob. of a point lying in each of K clusters
	print 'point ', x, 'prob : ',['%.3f' % i for i in probs/sum(probs)]