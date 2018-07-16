import numpy as np

def calc_hammingDist(B1, B2):
	q = B2.shape[1]
	distH = 0.5 * (q - np.dot(B1, B2.transpose()))
	return distH

def calc_map(qB, rB, query_L, retrieval_L):
	# qB: {-1,+1}^{mxq}
	# rB: {-1,+1}^{nxq}
	# query_L: {0,1}^{mxl}
	# retrieval_L: {0,1}^{nxl}
	num_query = query_L.shape[0]
	map = 0
	for iter in xrange(num_query):
		gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
		tsum = np.sum(gnd)
		if tsum == 0:
			continue
		hamm = calc_hammingDist(qB[iter, :], rB)
		ind = np.argsort(hamm)
		gnd = gnd[ind]
		count = np.linspace(1, tsum, tsum)

		tindex = np.asarray(np.where(gnd == 1)) + 1.0
		map = map + np.mean(count / (tindex))
	map = map / num_query
	return map


if __name__=='__main__':
	qB = np.array([[ 1,-1, 1, 1],
								 [-1,-1,-1, 1],
								 [ 1, 1,-1, 1],
								 [ 1, 1, 1,-1]])
	rB = np.array([[ 1,-1, 1,-1],
								 [-1,-1, 1,-1],
								 [-1,-1, 1,-1],
								 [ 1, 1,-1,-1],
								 [-1, 1,-1,-1],
								 [ 1, 1,-1, 1]])
	query_L = np.array([[0, 1, 0, 0],
											[1, 1, 0, 0],
											[1, 0, 0, 1],
											[0, 1, 0, 1]])
	retrieval_L = np.array([[1, 0, 0, 1],
													[1, 1, 0, 0],
													[0, 1, 1, 0],
													[0, 0, 1, 0],
													[1, 0, 0, 0],
													[0, 0, 1, 0]])

	map = calc_map(qB, rB, query_L, retrieval_L)
	print map
