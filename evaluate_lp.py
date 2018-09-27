import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances

def poincare_distance(X):
	norm_X = np.linalg.norm(X, keepdims=True, axis=-1)
	norm_X = np.minimum(norm_X, np.nextafter(1,0, ))
	uu = euclidean_distances(X) ** 2
	dd = (1 - norm_X**2) * (1 - norm_X**2).T
#     print 1 + 2 * uu / dd
	return np.arccosh(1 + 2 * uu / dd)


def evaluate_rank_and_MAP(dists, edgelist, non_edgelist):
	assert not isinstance(edgelist, dict)

	if not isinstance(edgelist, np.ndarray):
		edgelist = np.array(edgelist)

	if not isinstance(non_edgelist, np.ndarray):
		non_edgelist = np.array(non_edgelist)

	edge_dists = dists[edgelist[:,0], edgelist[:,1]]
	non_edge_dists = dists[non_edgelist[:,0], non_edgelist[:,1]]


	labels = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	scores = -np.append(edge_dists, non_edge_dists)
	ap_score = average_precision_score(labels, scores) # macro by default
	auc_score = roc_auc_score(labels, scores)


	idx = non_edge_dists.argsort()
	ranks = np.searchsorted(non_edge_dists, edge_dists, sorter=idx) + 1
	ranks = ranks.mean()

	print ("MEAN RANK =", ranks, "AP =", ap_score, 
		"ROC AUC =", auc_score)

	return ranks, ap_score, auc_score


def main():

	X = np.genfromtxt("cora_ml.embedding", delimiter=",")
	dists = poincare_distance(X)

	print (dists)

	raise NotImplementedError

	pass

if __name__ == "__main__":
	main()