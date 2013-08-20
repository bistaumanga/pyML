''' Demonstration of use of GMM module'''
import sys, time
import pylab as plt
import numpy as np
import gmm

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

# Load data
X = np.genfromtxt('data1.csv', delimiter=',')

mu, sigma, w, cost = gmm.EM(X, K = 3, show_progress = show)
np.set_printoptions(precision = 2)
for k in range(len(w)):
	print 'cluster ->', k+1, 'mean : -> ', mu[k],'weight : ->',  w[k]
#testing a point (6, 8) for membershiop
gmm.test([6, 8], mu, sigma, w)
gmm.test([11, 12], mu, sigma, w) 

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