import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"


import numpy as np

import argparse

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances


def convert_edgelist_to_dict(edgelist, undirected=True, self_edges=False):
	if edgelist is None:
		return None
	# sorts = [lambda x: sorted(x)]
	# if undirected:
	# 	sorts.append(lambda x: sorted(x, reverse=True))
	# edges = (sort(edge) for edge in edgelist for sort in sorts)
	edges = edgelist
	edge_dict = {}
	for u, v in edges:
		if self_edges:
			default = set(u)
		else:
			default = set()
		edge_dict.setdefault(u, default).add(v)
		if undirected:
			edge_dict.setdefault(v, default).add(u)

	# for u, v in edgelist:
	# 	assert v in edge_dict[u]
	# 	if undirected:
	# 		assert u in edge_dict[v]
	# raise SystemExit
	edge_dict = {k: list(v) for k, v in edge_dict.items()}

	return edge_dict


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

def evaluate_rank_and_MAP_fb(dists, edge_dict, non_edge_dict):

	assert isinstance(edge_dict, dict)

	ranks = []
	ap_scores = []
	roc_auc_scores = []
	
	for u, neighbours in edge_dict.items():
		# print (neighbours)
		# print (non_edge_dict[u])
		# raise SystemExit
		_dists = dists[u, neighbours + non_edge_dict[u]]
		_labels = np.append(np.ones(len(neighbours)), np.zeros(len(non_edge_dict[u])))
		# _dists = dists[u]
		# _dists[u] = 1e+12
		# _labels = np.zeros(dists.shape[0])
		# _dists_masked = _dists.copy()
		# _ranks = []
		# for v in neighbours:
		# 	_labels[v] = 1
		# 	_dists_masked[v] = np.Inf
		ap_scores.append(average_precision_score(_labels, -_dists))
		roc_auc_scores.append(roc_auc_score(_labels, -_dists))

		neighbour_dists = dists[u, neighbours]
		non_neighbour_dists = dists[u, non_edge_dict[u]]
		idx = non_neighbour_dists.argsort()
		_ranks = np.searchsorted(non_neighbour_dists, neighbour_dists, sorter=idx) + 1

		# _ranks = []
		# _dists_masked = _dists.copy()
		# _dists_masked[:len(neighbours)] = np.inf

		# for v in neighbours:
		# 	d = _dists_masked.copy()
		# 	d[v] = _dists[v]
		# 	r = np.argsort(d)
		# 	_ranks.append(np.where(r==v)[0][0] + 1)

		ranks.append(np.mean(_ranks))
	print ("MEAN RANK =", np.mean(ranks), "MEAN AP =", np.mean(ap_scores), 
		"MEAN ROC AUC =", np.mean(roc_auc_scores))
	return np.mean(ranks), np.mean(ap_scores), np.mean(roc_auc_scores)


def evaluate_direction(embedding, directed_edges, non_edges):

	if not isinstance(directed_edges, np.ndarray):
		directed_edges = np.array(directed_edges)

	if not isinstance(non_edges, np.ndarray):
		non_edges = np.array(non_edges)

	labels = np.append(np.ones(len(directed_edges)), np.zeros(len(non_edges)))
	ranks = embedding[:,-1]

	direction_predictions = ranks[directed_edges[:,0]] > ranks[directed_edges[:,1]]
	non_edge_predictions = ranks[non_edges[:,0]] > ranks[non_edges[:,1]]

	scores = np.append(direction_predictions, non_edge_predictions)

	ap_score = average_precision_score(labels, scores) # macro by default
	auc_score = roc_auc_score(labels, scores)

	print ("AP =", ap_score, 
		"ROC AUC =", auc_score)
	
	return ap_score, auc_score

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def parse_filenames(opts):
	dataset = opts.dset
	seed = opts.seed
	embedding_filename = os.path.join("embeddings", dataset, "seed={}".format(seed),  "embedding.csv")
	removed_edge_dir = os.path.join("removed_edges", dataset, "seed={}".format(seed) )
	val_edges_filename = os.path.join(removed_edge_dir, "val_edges.edgelist")
	val_non_edges_filename = os.path.join(removed_edge_dir, "val_non_edges.edgelist")
	test_edges_filename = os.path.join(removed_edge_dir, "test_edges.edgelist")
	test_non_edges_filename = os.path.join(removed_edge_dir, "test_non_edges.edgelist")
	test_results_filename = os.path.join("test_results", dataset, "test_results.csv")
	test_results_lock_filename = os.path.join("test_results", dataset, "test_results.lock")

	touch(test_results_lock_filename)
	
	return (embedding_filename, val_edges_filename, val_non_edges_filename, 
		test_edges_filename, test_non_edges_filename, test_results_filename. test_results_lock_filename)

def read_edgelist(fn):
	edges = []
	with open(fn, "r") as f:
	    for line in (l.rstrip() for l in f.readlines()):
	        edge = tuple(int(i) for i in line.split(" "))
			edges.append(edge)
	return edges


def lock_method(lock_filename):
	''' Use an OS lock such that a method can only be called once at a time. '''

	def decorator(func):

		@functools.wraps(func)
		def lock_and_run_method(*args, **kwargs):

			# Hold program if it is already running 
			# Snippet based on
			# http://linux.byexamples.com/archives/494/how-can-i-avoid-running-a-python-script-multiple-times-implement-file-locking/
			fp = open(lock_filename, 'r+')
			done = False
			while not done:
				try:
					fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
					done = True
				except IOError:
					pass
			return func(*args, **kwargs)

		return lock_and_run_method

	return decorator 

def threadsafe_fn(lock_filename, fn, *args, **kwargs ):
	lock_method(lock_filename)(fn)(*args, **kwargs)

def save_test_results(filename, seed, data, ):
	d = pd.DataFrame(index=[seed], data=data)
	# try:
	if os.path.exists(filename):
		test_df = pd.read_csv(filename, sep=",", index_col=0)
		test_df = d.combine_first(test_df)
	# except EmptyDataError:
	else:
		test_df = d
	test_df.to_csv(filename, sep=",")

def threadsafe_save_test_results(lock_filename, filename, seed, data):
	threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)
	

def main():
	parser = argparse.ArgumentParser(description='Load Poincare Embeddings and evaluate')
	parser.add_argument('-dset', help='Dataset to embed', type=str, default="cora_ml")
	parser.add_argument('-seed', help='Random seed.', type=int, default=0)
	opt = parser.parse_args()

	(embedding_filename, val_edges_filename, val_non_edges_filename, 
		test_edges_filename, test_non_edges_filename,
		test_results_filename, test_results_lock_filename) = parse_filenames(opt)

	poincare_embedding = np.genfromtxt(embedding_filename, delimiter=",")
	dists = poincare_distance(poincare_embedding)

	val_edges = read_edgelist(val_edges_filename)
	val_non_edges = read_edgelist(val_non_edges_filename)
	test_edges = read_edgelist(test_edges_filename)
	test_non_edges = read_edgelist(test_non_edges_filename)

	test_edge_dict = convert_edgelist_to_dict(test_edges)
	non_edge_dict = convert_edgelist_to_dict(non_edges)

	test_results = dict()

	(mean_rank_lp, map_lp, 
	mean_roc_lp) = evaluate_rank_and_MAP(dists, 
	test_edges, test_non_edges)

	test_results.update({"mean_rank_lp": mean_rank_lp, 
		"map_lp": map_lp,
		"mean_roc_lp": mean_roc_lp})

	(mean_rank_lp_fb, map_lp_fb, 
	mean_roc_lp_fb) = evaluate_rank_and_MAP_fb(dists, 
	test_edge_dict, non_edge_dict)

	test_results.update({"mean_rank_lp_fb": mean_rank_lp_fb, 
		"map_lp_fb": map_lp_fb,
		"mean_roc_lp_fb": mean_roc_lp_fb})

	threadsafe_save_test_results(test_results_lock_filename, test_results_filename, opt.seed, data=test_results )


if __name__ == "__main__":
	main()