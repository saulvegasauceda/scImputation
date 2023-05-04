import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import binom

from sklearn.decomposition import NMF
merfish = sc.read_h5ad("./data/merfish.namespaced.h5ad")

TARGET_SUM = 500
NUMBER_OF_CELLS = 10_000

sc.pp.filter_cells(merfish, min_counts=1, inplace=True)
sc.pp.normalize_total(merfish, target_sum=TARGET_SUM,
                      exclude_highly_expressed=False)

# sample out 10_000 so I can run it fast
merfish = merfish[:NUMBER_OF_CELLS]


def artificially_sample_cells(merfish, capture_rate):
    '''
    Simulating Bernoulli sampling with synthetic dataset 
    where p = capture_rate (p := probability RNA is included in set)
    '''

    # Give MERFISH dataframe gene names for col names
    merfish_df = merfish.to_df()
    ensmug_to_name = dict(zip(merfish.var.index, merfish.var["gene_name"]))
    merfish_df.rename(columns=ensmug_to_name, inplace=True)

    # Apply dropout method
    dropout_df = merfish_df.copy()
    return dropout_df.applymap(lambda n: binom(n, p=capture_rate))  # binomial
