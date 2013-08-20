import numpy as np

m1, cov1 = [9, 8], [[1.5, 2], [2, 4]]
m2, cov2 = [6, 13], [[2.5, -1.5], [-1.5, 1]]
m3, cov3 = [4, 7], [[0.25, 0.5], [-0.1, 0.5]]
data1 = np.random.multivariate_normal(m1, cov1, 250)
data2 = np.random.multivariate_normal(m2, cov2, 180)
data3 = np.random.multivariate_normal(m3, cov3, 100)
X = np.vstack((data1,np.vstack((data2,data3))))
np.random.shuffle(X)

np.savetxt("data1.csv", X, delimiter=",", fmt = '%.2f')
y = np.genfromtxt('data1.csv', delimiter=',')
print y.shape
import pylab as plt
plt.plot(y.T[0], y.T[1], 'm*')
plt.show()