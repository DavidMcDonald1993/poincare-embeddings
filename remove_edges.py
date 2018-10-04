import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import random
import numpy as np
import networkx as nx

import argparse

from data_utils import load_g2g_datasets, load_collaboration_network

def write_edgelist_to_file(edgelist, file):
	with open(file, "w+") as f:
		for u, v in edgelist:
			f.write("{} {}\n".format(u,v))

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

	edgelist_dir = os.path.join("training_edgelists", dataset, "seed={}".format(seed), )
	removed_edges_dir = os.path.join("removed_edges", dataset, "seed={}".format(seed))

	if not os.path.exists(edgelist_dir):
		os.makedirs(edgelist_dir)
	if not os.path.exists(removed_edges_dir):
		os.makedirs(removed_edges_dir)

	training_edgelist_fn = os.path.join(edgelist_dir, "training_edges.edgelist")
	val_edgelist_fn = os.path.join(removed_edges_dir, "val_edges.edgelist")
	val_non_edgelist_fn = os.path.join(removed_edges_dir, "val_non_edges.edgelist")
	test_edgelist_fn = os.path.join(removed_edges_dir, "test_edges.edgelist")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, "test_non_edges.edgelist")

	if os.path.exists(training_edgelist_fn):
		print ("{} already exists -- terminating".format(training_edgelist_fn))
		return


	if dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
		topology_graph, features, labels, label_info = load_g2g_datasets(dataset, args)
	elif dataset in ["AstroPh", "CondMat", "GrQc", "HepPh"]:
		topology_graph, features, labels, label_info = load_collaboration_network(args)

	print("loaded dataset")


	edges = topology_graph.edges()
	non_edges = list(nx.non_edges(topology_graph))


	train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges) = split_edges(edges, non_edges, seed)

	print ("removed edges")

	topology_graph.remove_edges_from(val_edges + test_edges)

	write_edgelist_to_file(train_edges, training_edgelist_fn)
	write_edgelist_to_file(val_edges, val_edgelist_fn)
	write_edgelist_to_file(val_non_edges, val_non_edgelist_fn)
	write_edgelist_to_file(test_edges, test_edgelist_fn)
	write_edgelist_to_file(test_non_edges, test_edgelist_fn)

	print ("done")


if __name__ == "__main__":
	main()