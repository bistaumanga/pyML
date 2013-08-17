'''Implementation and  Demostration Demostration of K Means Clustering
Requires : python 2.7.x, Numpy 1.7.1, Matplotlib, 1.2.1'''

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

def kMeans(X, K, maxIters = 10, plot_progress = False):

    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        if plot_progress: show(X, C, np.array(centroids))
    return np.array(centroids) , C

# data generation
data1 = np.random.multivariate_normal([9, 8], [[1.5, 2], [2, 4]], 150)
data2 = np.random.multivariate_normal([6, 13], [[2.5, -1.5], [-1.5, 1]], 100)
data3 = np.random.multivariate_normal([4, 7], [[0.25, 0.5], [-1, 0.5]], 50)
data = np.vstack((data1,np.vstack((data2,data3))))

np.random.shuffle(data)
centroids, C = kMeans(data, K = 3, plot_progress = True)
show(data, C, centroids, True)