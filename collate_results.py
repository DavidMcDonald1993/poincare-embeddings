import numpy as np
import pandas as pd
import argparse
import os
import itertools

from matplotlib import pyplot as plt

# import glob

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Collate results script")

	parser.add_argument("--test-results", dest="test_results_path", default="test_results/", 
		help="path to save test results (default is 'test_results/)'.")

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	num_seeds = 30
	columns = range(num_seeds)

	datasets = ["cora_ml", "citeseer"]

	exps = ["eval_lp", "eval_class_pred"]
	dims = ["dim={}".format(dim) for dim in (5, 10, 25, 50)]
	# jps = ["no_attributes"] + ["jump_prob={}".format(i) for i in (0.05, 0.1, 0.2, 0.5, 0.8, 1.0)]

	f, a = plt.subplots(nrows=len(datasets), ncols=len(dims), figsize=(5*len(datasets), 5*len(dims)), dpi= 80)

	for exp in exps:

		print (exp)
		print ()

		for i, dataset in enumerate(datasets):
			index = [dims]

			if exp == "eval_lp":
				index.append( ["map_lp", "mean_roc_lp"] )
			else:
				index.append(["map_reconstruction", "mean_roc_reconstruction"] +\
							["{:.2f}_{}".format(perc, avg) for avg, perc in itertools.product(["micro", "macro"], np.arange(0.02, 0.11, 0.01))])

			collated_df = pd.DataFrame(0, index=pd.MultiIndex.from_product(index), columns=columns)
			print collated_df.shape
			for dim in dims:
				# for jp in jps:
				test_results_filename = os.path.join(args.test_results_path, dataset, dim, exp, "test_results.csv")
				if os.path.exists(test_results_filename):
					print ("loading from {}".format(test_results_filename))
					df = pd.read_csv(test_results_filename, sep=",", index_col=0)#.iloc[:num_seeds]
					# assert not df.isna().any().any(), df.to_string()
					cols = ["map_lp", "mean_roc_lp"] if exp == "eval_lp" else ["map_reconstruction", "mean_roc_reconstruction"] +\
						["{:.2f}_{}".format(perc, avg) for avg, perc in itertools.product(["micro", "macro"], np.arange(0.02, 0.11, 0.01))]
					print (dim, cols)
					print (collated_df.loc[dim].shape)
					print (df[cols].values.T.shape)

					collated_df.loc[dim] = df[cols].values.T

			collated_df_mean = collated_df.mean(1)
			collated_df_stderr = collated_df.sem(1)

			if exp == "eval_lp":
				
				desired_columns = ["map_lp", "mean_roc_lp"]

				for dim in dims:
					print (collated_df_mean.loc[dim][desired_columns].shape)
					dim_mean_df = collated_df_mean.loc[dim][desired_columns]
					dim_stderr_df = collated_df_stderr.loc[dim][desired_columns]

					print (dim)
					print (dim_mean_df.to_string())
					print ()
			
			else:

				desired_columns = ["map_reconstruction", "mean_roc_reconstruction"]

				for dim in dims:

					dim_mean_df = collated_df_mean.loc[dim][desired_columns]
					dim_stderr_df = collated_df_stderr.loc[dim][desired_columns]

					print (dim)
					print (dim_mean_df.to_string())
					print ()

				desired_columns = ["{:.2f}_micro".format(perc) for perc in np.arange(0.02, 0.11, 0.01)]

				plt.suptitle(dataset)
				for j, dim in enumerate(dims):

					dim_mean_df = collated_df_mean.loc[dim][desired_columns]
					dim_stderr_df = collated_df_stderr.loc[dim][desired_columns]

					dim_mean_df.plot(yerr=dim_stderr_df, title=dim, ax=a[i, j], xticks=range(len(dim_mean_df.index)))
					a[i, j].set(xlabel="labelled_percentage", ylabel="f1_micro_scores", 
						xticklabels=["{:.2f}".format(x) for x in np.arange(0.02, 0.11, 0.01)])

	plt.show()

if __name__ == "__main__":
	main()