import os
import random
import numpy as np
import networkx as nx

import argparse

from data_utils import load_g2g_datasets

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

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	for dataset in ["cora_ml", "cora", "pubmed", "citeseer"]:

		for seed in range(100):

			topology_graph, features, labels, label_info = load_g2g_datasets(dataset, args)
			edges = topology_graph.edges()
			non_edges = list(nx.non_edges(topology_graph))

			edgelist_dir = os.path.join("training_edgelists", dataset, "seed={}".format(seed), )
			removed_edges_dir = os.path.join("removed_edges", "seed={}".format(seed))

			if not os.path.exists(edgelist_dir):
				os.makedirs(edgelist_dir)
			if not os.path.exists(removed_edges_dir):
				os.makedirs(removed_edges_dir)

			train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges) = split_edges(edges, non_edges, seed)

			topology_graph.remove_edges_from(val_edges + test_edges)

			write_edgelist_to_file(topology_graph.edges(), os.path.join(edgelist_dir, "training_edges.edgelist"))
			write_edgelist_to_file(val_edges, os.path.join(removed_edges_dir, "val_edges.edgelist"))
			write_edgelist_to_file(val_non_edges, os.path.join(removed_edges_dir, "val_non_edges.edgelist"))
			write_edgelist_to_file(test_edges, os.path.join(removed_edges_dir, "test_edges.edgelist"))
			write_edgelist_to_file(test_non_edges, os.path.join(removed_edges_dir, "test_non_edges.edgelist"))




if __name__ == "__main__":
	main()
