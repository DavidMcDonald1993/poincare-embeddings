import pandas as pd 
import os
import itertools

from pandas.errors import EmptyDataError

def main():

    datasets = ["cora_ml", "citeseer", "ppi", "pubmed", "mit"]
    dims = (5, 10, 25, 50)
    seeds = range(30)
    exps = ["nc_experiment", "lp_experiment"]

    for dataset, dim, seed, exp in itertools.product(
        datasets, dims, seeds, exps
    ):
        embedding_directory = os.path.join(
            "embeddings", dataset, "dim={:02d}".format(dim), 
            "seed={:03d}".format(seed), exp
        )

        filename = os.path.join(embedding_directory, "embedding.csv.gz")

        try:
            pd.read_csv(filename)
        except EmptyDataError:
            print (filename, "is empty removing it")
            os.remove(filename)
        except IOError:
            print (filename, "does not exist")




if __name__ == "__main__":
    main()