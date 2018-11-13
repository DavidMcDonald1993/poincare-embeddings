import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"


import numpy as np
import networkx as nx
import pandas as pd

import argparse

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit

from data_utils import load_g2g_datasets, load_collaboration_network
import functools
import fcntl

def convert_edgelist_to_dict(edgelist, undirected=True, self_edges=False):
	if edgelist is None:
		return None
	if undirected:
		edgelist += [(v, u) for u, v in edgelist]
	edge_dict = {}
	for u, v in edgelist:
		if self_edges:
			default = set(u)
		else:
			default = set()
		edge_dict.setdefault(u, default).add(v)

	edge_dict = {k: list(v) for k, v in edge_dict.items()}

	return edge_dict


def poincare_distance(X):
	norm_X_sq = 1 - np.linalg.norm(X, keepdims=True, axis=-1) ** 2
	norm_X_sq = np.minimum(norm_X_sq, np.nextafter(1,0, ))
	uu = euclidean_distances(X) ** 2
	dd = norm_X_sq * norm_X_sq.T
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

# def evaluate_rank_and_MAP_fb(dists, edge_dict, non_edge_dict):

# 	assert isinstance(edge_dict, dict)

# 	ranks = []
# 	ap_scores = []
# 	roc_auc_scores = []
	
# 	for u, neighbours in edge_dict.items():
# 		_dists = dists[u, neighbours + non_edge_dict[u]]
# 		_labels = np.append(np.ones(len(neighbours)), np.zeros(len(non_edge_dict[u])))
# 		ap_scores.append(average_precision_score(_labels, -_dists))
# 		roc_auc_scores.append(roc_auc_score(_labels, -_dists))

# 		neighbour_dists = dists[u, neighbours]
# 		non_neighbour_dists = dists[u, non_edge_dict[u]]
# 		idx = non_neighbour_dists.argsort()
# 		_ranks = np.searchsorted(non_neighbour_dists, neighbour_dists, sorter=idx) + 1

# 		ranks.append(np.mean(_ranks))
# 	print ("MEAN RANK =", np.mean(ranks), "MEAN AP =", np.mean(ap_scores), 
# 		"MEAN ROC AUC =", np.mean(roc_auc_scores))
# 	return np.mean(ranks), np.mean(ap_scores), np.mean(roc_auc_scores)

def poincare_to_klein(poincare_embedding):
	return 2 * poincare_embedding / (1 + np.sum(np.square(poincare_embedding), axis=-1, keepdims=True))

def evaluate_classification(klein_embedding, labels, 
	label_percentages=np.arange(0.02, 0.11, 0.01), n_repeats=10):

	assert len(labels.shape) == 1

	num_nodes, dim = klein_embedding.shape

	f1_micros = np.zeros((n_repeats, len(label_percentages)))
	f1_macros = np.zeros((n_repeats, len(label_percentages)))

	
	model = LogisticRegressionCV()

	for seed in range(n_repeats):
	
		for i, label_percentage in enumerate(label_percentages):

			sss = StratifiedShuffleSplit(n_splits=1, test_size=1-label_percentage, random_state=seed)
			split_train, split_test = next(sss.split(klein_embedding, labels))
			# num_labels = int(max(num_nodes * label_percentage, len(classes)))
			# idx = np.random.permutation(num_nodes)
			# if len(labels.shape) > 1:
			# 	model =  OneVsRestClassifier(LogisticRegression(random_state=0))
			model.fit(klein_embedding[split_train], labels[split_train])
			predictions = model.predict(klein_embedding[split_test])
			f1_micro = f1_score(labels[split_test], predictions, average="micro")
			f1_macro = f1_score(labels[split_test], predictions, average="macro")
			f1_micros[seed,i] = f1_micro
			f1_macros[seed,i] = f1_macro

	return label_percentages, f1_micros.mean(axis=0), f1_macros.mean(axis=0)


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
	dim = opts.dim
	seed = opts.seed
	experiment = opts.exp
	embedding_filename = os.path.join("embeddings", dataset, "dim={}".format(dim), "seed={}".format(seed),
		experiment, "embedding.csv")

	test_results_dir = os.path.join("test_results", dataset, "dim={}".format(dim), experiment)
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir)
	test_results_filename = os.path.join(test_results_dir, "test_results.csv")
	test_results_lock_filename = os.path.join(test_results_dir, "test_results.lock")

	touch(test_results_lock_filename)


	if experiment == "eval_lp":
		removed_edge_dir = os.path.join("removed_edges", dataset, "seed={}".format(seed), "eval_lp")
		val_edges_filename = os.path.join(removed_edge_dir, "val_edges.edgelist")
		val_non_edges_filename = os.path.join(removed_edge_dir, "val_non_edges.edgelist")
		test_edges_filename = os.path.join(removed_edge_dir, "test_edges.edgelist")
		test_non_edges_filename = os.path.join(removed_edge_dir, "test_non_edges.edgelist")
		
		return (embedding_filename, (val_edges_filename, val_non_edges_filename, 
			test_edges_filename, test_non_edges_filename), test_results_filename, test_results_lock_filename)
	else:

		complete_edgelist_dir = os.path.join("training_edgelists", dataset, "seed={}".format(seed), "eval_class_pred")
		complete_edgelist_fn = os.path.join(complete_edgelist_dir, "training_edges.edgelist")
		complete_non_edgelist_fn = os.path.join(complete_edgelist_dir, "complete_non_edges.edgelist")

		return (embedding_filename, (complete_edgelist_fn, complete_non_edgelist_fn), 
			test_results_filename, test_results_lock_filename)


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
	if os.path.exists(filename):
		test_df = pd.read_csv(filename, sep=",", index_col=0)
		test_df = d.combine_first(test_df)
	else:
		test_df = d
	test_df.to_csv(filename, sep=",")

def threadsafe_save_test_results(lock_filename, filename, seed, data):
	threadsafe_fn(lock_filename, save_test_results, filename=filename, seed=seed, data=data)
	

def main():
	parser = argparse.ArgumentParser(description='Load Poincare Embeddings and evaluate')
	parser.add_argument('-dset', help='Dataset to embed', type=str, default="cora_ml")
	parser.add_argument('-seed', help='Random seed.', type=int, default=0)
	parser.add_argument('-dim', help='Dimension of embedding.', type=int, default=5)
	parser.add_argument('-exp', help='Experiment to perform', type=str, )

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	parser.add_argument("--data-directory", dest="data_directory", type=str, default="/data/",
		help="The directory containing data files (default is '/data/').")

	opt = parser.parse_args()

	assert opt.exp in ["eval_lp", "eval_class_pred"]

	dataset = opt.dset
	if dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
		topology_graph, features, labels, label_info = load_g2g_datasets(dataset, opt)
	elif dataset in ["AstroPh", "CondMat", "GrQc", "HepPh"]:
		topology_graph, features, labels, label_info = load_collaboration_network(opt)

	non_edges = list(nx.non_edges(topology_graph))


	embedding_filename, filenames, test_results_filename, test_results_lock_filename = parse_filenames(opt)

	poincare_embedding = np.genfromtxt(embedding_filename, delimiter=",")
	dists = poincare_distance(poincare_embedding)

	print (dists)

	test_results = dict()

	if opt.exp == "eval_lp":

		(val_edges_filename, val_non_edges_filename, 
			test_edges_filename, test_non_edges_filename) = filenames

		val_edges = read_edgelist(val_edges_filename)
		val_non_edges = read_edgelist(val_non_edges_filename)
		test_edges = read_edgelist(test_edges_filename)
		test_non_edges = read_edgelist(test_non_edges_filename)

		test_edge_dict = convert_edgelist_to_dict(test_edges)
		non_edge_dict = convert_edgelist_to_dict(non_edges)

		(mean_rank_lp, map_lp, 
		mean_roc_lp) = evaluate_rank_and_MAP(dists, 
		test_edges, test_non_edges)

		test_results.update({"mean_rank_lp": mean_rank_lp, 
			"map_lp": map_lp,
			"mean_roc_lp": mean_roc_lp})

		# (mean_rank_lp_fb, map_lp_fb, 
		# mean_roc_lp_fb) = evaluate_rank_and_MAP_fb(dists, 
		# test_edge_dict, non_edge_dict)

		# test_results.update({"mean_rank_lp_fb": mean_rank_lp_fb, 
		# 	"map_lp_fb": map_lp_fb,
		# 	"mean_roc_lp_fb": mean_roc_lp_fb})

	else:
		complete_edgelist_fn, complete_non_edgelist_fn = filenames
		training_edges = read_edgelist(complete_edgelist_fn)
		training_non_edges = read_edgelist(complete_non_edgelist_fn)

		(mean_rank_reconstruction, map_reconstruction, 
		mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
		training_edges, training_non_edges)

		test_results.update({"mean_rank_reconstruction": mean_rank_reconstruction, 
			"map_reconstruction": map_reconstruction,
			"mean_roc_reconstruction": mean_roc_reconstruction})

		klein_embedding = poincare_to_klein(poincare_embedding)
		
		label_percentages, f1_micros, f1_macros = evaluate_classification(klein_embedding, labels, )

		for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
			test_results.update({"{:.2f}_micro".format(label_percentage): f1_micro})
			test_results.update({"{:.2f}_macro".format(label_percentage): f1_macro})
		test_results.update({"micro_sum" : np.sum(f1_micros)})


	print ("saving test results to {}".format(test_results_filename))
	threadsafe_save_test_results(test_results_lock_filename, test_results_filename, opt.seed, data=test_results )


if __name__ == "__main__":
	main()