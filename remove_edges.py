import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import random
import numpy as np
import networkx as nx

import argparse

from data_utils import load_g2g_datasets, load_collaboration_network, load_ppi

def write_edgelist_to_file(edgelist, file):
	with open(file, "w+") as f:
		for u, v in edgelist:
			f.write("{} {}\n".format(u, v))

def split_edges(edges, non_edges, seed, val_split=0.05, test_split=0.1, neg_mul=1):
	
	num_val_edges = int(np.ceil(len(edges) * val_split))
	num_test_edges = int(np.ceil(len(edges) * test_split))

	random.seed(seed)
	random.shuffle(edges)
	random.shuffle(non_edges)

	val_edges = edges[:num_val_edges]
	test_edges = edges[num_val_edges:num_val_edges+num_test_edges]
	train_edges = edges[num_val_edges+num_test_edges:]

	val_non_edges = non_edges[:num_val_edges*neg_mul]
	test_non_edges = non_edges[num_val_edges*neg_mul:num_val_edges*neg_mul+num_test_edges*neg_mul]

	return train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges)

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Hyperbolic Skipgram for feature learning on complex networks")

	parser.add_argument("--data-directory", dest="data_directory", type=str, default="/data/",
		help="The directory containing data files (default is '/data/').")

	parser.add_argument("--dataset", dest="dataset", type=str, default="cora_ml",
		help="Dataset to process.")
	parser.add_argument("--seed", type=int, default=0)


	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	dataset = args.dataset
	seed = args.seed

	complete_edgelist_dir = os.path.join("training_edgelists", dataset, "seed={}".format(seed), "eval_class_pred")
	training_edgelist_dir = os.path.join("training_edgelists", dataset, "seed={}".format(seed), "eval_lp")
	removed_edges_dir = os.path.join("removed_edges", dataset, "seed={}".format(seed), "eval_lp")

	if not os.path.exists(complete_edgelist_dir):
		os.makedirs(complete_edgelist_dir)
	if not os.path.exists(training_edgelist_dir):
		os.makedirs(training_edgelist_dir)
	if not os.path.exists(removed_edges_dir):
		os.makedirs(removed_edges_dir)

	complete_edgelist_fn = os.path.join(complete_edgelist_dir, "training_edges.edgelist")
	complete_non_edgelist_fn = os.path.join(complete_edgelist_dir, "complete_non_edges.edgelist")

	training_edgelist_fn = os.path.join(training_edgelist_dir, "training_edges.edgelist")
	val_edgelist_fn = os.path.join(removed_edges_dir, "val_edges.edgelist")
	val_non_edgelist_fn = os.path.join(removed_edges_dir, "val_non_edges.edgelist")
	test_edgelist_fn = os.path.join(removed_edges_dir, "test_edges.edgelist")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, "test_non_edges.edgelist")


	if dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
		load = load_g2g_datasets
	elif dataset in ["AstroPh", "CondMat", "GrQc", "HepPh"]:
		load = load_collaboration_network
	elif dataset == "ppi":
		load = load_ppi
	else:
		raise Exception

	graph, features, labels = load(dataset, args)

	print("loaded dataset")

	edges = list(graph.edges()) + [(u, u) for u in graph.nodes()]
	non_edges = list(nx.non_edges(graph))

	write_edgelist_to_file(edges, complete_edgelist_fn)
	write_edgelist_to_file(non_edges, complete_non_edgelist_fn)

	train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges) = split_edges(edges, non_edges, seed)
	train_edges += [(u, u) for u in graph.nodes()] # ensure that every node appears at least once

	print ("removed edges")

	graph.remove_edges_from(val_edges + test_edges)

	write_edgelist_to_file(train_edges, training_edgelist_fn)
	write_edgelist_to_file(val_edges, val_edgelist_fn)
	write_edgelist_to_file(val_non_edges, val_non_edgelist_fn)
	write_edgelist_to_file(test_edges, test_edgelist_fn)
	write_edgelist_to_file(test_non_edges, test_non_edgelist_fn)

	print ("done")


if __name__ == "__main__":
	main()
