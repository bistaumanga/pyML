
def gradDesc(cost, init_Wt, maxEpochs = 10000, lr = 0.8, mom = 0.5):
	J = []
	Wt = init_Wt
	for i in range(maxEpochs):
		j, D = cost(Wt)
		Wt -= lr * D
		J.append(j)
	return J, Wt