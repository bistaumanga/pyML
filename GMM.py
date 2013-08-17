''' Gaussian Mixture Model
A probabilistic clustring method
Expectation Maximization Algorithm'''
import sys, time
import pylab as plt
import numpy as np

from matplotlib.patches import Ellipse

def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(abs(vals))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

plt.ion()

def show(X, mu, cov, keep = False):

    time.sleep(0.1)
    plt.cla()
    K = len(mu) # number of clusters
    colors = ['b', 'k', 'g', 'c', 'm', 'y', 'r']
    plt.plot(X.T[0], X.T[1], 'm*')
    for k in range(K):
    	plot_ellipse(mu[k], cov[k],  alpha=0.6, color = colors[k % len(colors)])  
    if keep :  	plt.ioff()
    else:	plt.draw()

def p(x, mu, sigma):
	n = len(x)
	import math
	return np.linalg.det(sigma) ** - 0.5 * (2 * math.pi) ** (-n/2) * \
					np.exp( -0.5 * np.dot(x - mu , \
						np.dot(np.linalg.inv(sigma) , x - mu)))

def EM(X, K, max_iters = 100, show_progress = False):

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
		if show_progress : show(X, mu, Sigma)
	print 'Converged in ', curr_iter, ' Iterations !!'
	return mu, Sigma, w, costs

def test(x, mu, sigma, w):
	probs = np.array([w[k] * p(np.array(x), mu[k], sigma[k]) for k in range(3)])
	# result is prob. of a point lying in each of K clusters
	print 'point ', x, 'prob : ',['%.3f' % i for i in probs/sum(probs)]

m1, cov1 = [9, 8], [[1.5, 2], [2, 4]]
m2, cov2 = [6, 13], [[2.5, -1.5], [-1.5, 1]]
m3, cov3 = [4, 7], [[0.25, 0.5], [-0.1, 0.5]]
data1 = np.random.multivariate_normal(m1, cov1, 150)
data2 = np.random.multivariate_normal(m2, cov2, 90)
data3 = np.random.multivariate_normal(m3, cov3, 60)
X = np.vstack((data1,np.vstack((data2,data3))))
np.random.shuffle(X)
mu = [m1, m2, m3]
sigma = [cov1, cov2, cov3]
mu, sigma, w, cost = EM(X, K = 3, show_progress = True)
np.set_printoptions(precision = 2)
for k in range(len(w)):
	print 'cluster ->', k+1, 'mean : -> ', mu[k],'weight : ->',  w[k]
#testing a point (6, 8) for membershiop
test([6, 8], mu, sigma, w)
test([11, 12], mu, sigma, w) 

plt.close()
fig = plt.figure(figsize = (13, 6))
fig.add_subplot(121)
show(X, mu, sigma, True)
fig.add_subplot(122)
plt.plot(np.array(cost))
plt.title('Log Likelihood vs iteration plot')
plt.xlabel('Iterations')
plt.ylabel('log likelihood')
plt.show()