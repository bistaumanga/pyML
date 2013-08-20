'''Implementation and of K Means Clustering
Requires : python 2.7.x, Numpy 1.7.1+, Matplotlib, 1.2.1+'''
import sys
import pylab as plt
import numpy as np
plt.ion()

def show(X, C, centroids, keep = False):
    import time
    time.sleep(0.5)
    plt.cla()
    plt.plot(X[C == 0, 0], X[C == 0, 1], 'ob',
         X[C == 1, 0], X[C == 1, 1], 'or',
         X[C == 2, 0], X[C == 2, 1], 'og')
    plt.plot(centroids[:,0],centroids[:,1],'*m',markersize=20)
    plt.draw()
    if keep :
        plt.ioff()
        plt.show()

# load data
data = np.genfromtxt('data1.csv', delimiter=',')
from kMeans import kMeans
centroids, C = kMeans(data, K = 3, plot_progress = show)
show(data, C, centroids, True)