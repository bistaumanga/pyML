def gradDesc(J, init_x, maxEpochs = 10000, lr = 0.8):
	''' Optimization by Gradient Descent 
	> init_x must be a numpy matrix: n dim vector, for reference updates of x
	> J must be a partil function that takes necessary arguments along with x
		and returns cost and its partial derivatives wrt to all elements'''


	J_val = []
	x = init_x
	for i in range(maxEpochs):
		j, pd_X = J()
		x -= lr * pd_X
		J_val.append(j)
	return J_val, x