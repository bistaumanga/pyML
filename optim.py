def gradDesc(J, init_x, maxEpochs = 10000, lr = 0.8):
	''' Optimization by Gradient Descent '''
	
	J_val = []
	Wt = init_x
	for i in range(maxEpochs):
		j, D = J()
		Wt -= lr * D
		J_val.append(j)
	return J_val, Wt