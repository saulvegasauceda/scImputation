from numpy.random import seed
from multiprocessing import Pool
from functools import partial
import scanpy as sc
from nmf import nmf_imputation
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed(1738)
    TARGET_SUM = 500
    NUMBER_OF_CELLS = 10_000
    CAPTURE_RATE = 0.30

    # getting files
    path_to_dir = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/"
    output_path = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/nmf/"
    dropout_file = path_to_dir + "dropout_capture_rate=0.3.h5ad"
    merfish_file = path_to_dir + "merfish_norm.h5ad"

    merfish = sc.read_h5ad(merfish_file)
    dropout_adata = sc.read_h5ad(dropout_file)

    # Setup for grid search for nmf
    n_components_params = [1, 5, 10, 20, 40, 50, 100]
    alpha_W_params = [0, 0.2, 0.5, 0.7, 1]
    param_grid = []
    for dims in n_components_params:
        for alpha_W in alpha_W_params:
            param_grid.append((dims, alpha_W))

    # using partial function to pass in default params
    run_imputation_on_dropout = partial(
        nmf_imputation,
        counts_adata=dropout_adata,
        target_sum=TARGET_SUM,
        output_path=output_path
    )

    print("Processing running NMF imputation...")
    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        p.map(run_imputation_on_dropout, param_grid)

    print("Done!")
