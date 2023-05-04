import scanpy as sc
import numpy as np
import pandas as pd
from numpy.random import seed
from scipy.stats import rv_discrete
from scipy.stats import nbinom
from scipy.stats import bernoulli


def artificially_sample_cells(merfish, capture_rate):
    '''
    Simulating Bernoulli sampling with synthetic dataset
    where p = capture_rate (p := probability RNA is included in set)
    '''

    # Give MERFISH dataframe gene names for col names
    merfish_df = merfish.to_df().round(0).astype('Int64')
    ensmug_to_name = dict(zip(merfish.var.index, merfish.var["gene_name"]))
    merfish_df.rename(columns=ensmug_to_name, inplace=True)

    # Apply dropout method
    dropout_df = merfish_df.copy()
    dropout_df = dropout_df.applymap(
        lambda n: np.sum(bernoulli.rvs(capture_rate, size=n)))  # binomial

    dropout_adata = sc.AnnData(dropout_df)
    return dropout_adata


if __name__ == '__main__':
    seed(1738)
    path_to_dir = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/"
    merfish = sc.read_h5ad(
        "/Users/saulvegasauceda/Desktop/Kellis_UROP/BICCN/data/merfish.namespaced.h5ad")

    TARGET_SUM = 500
    NUMBER_OF_CELLS = 10_000
    CAPTURE_RATE = 0.30

    sc.pp.filter_cells(merfish, min_counts=1, inplace=True)
    sc.pp.normalize_total(merfish, target_sum=TARGET_SUM,
                          exclude_highly_expressed=False)

    # sample out 10_000 so I can run it fast
    merfish = merfish[:NUMBER_OF_CELLS]

    print("Corrupting matrix")
    dropout_adata = artificially_sample_cells(merfish, CAPTURE_RATE)
    sc.pp.normalize_total(dropout_adata, target_sum=TARGET_SUM,
                          exclude_highly_expressed=False)

    print("Saving adata files")
    dropout_adata.write_h5ad(
        path_to_dir + f"dropout_capture_rate={CAPTURE_RATE}.h5ad", compression='gzip')
    merfish.write_h5ad(path_to_dir + "merfish_norm.h5ad", compression='gzip')
