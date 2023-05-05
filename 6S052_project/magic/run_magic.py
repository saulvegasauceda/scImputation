from numpy.random import seed
from multiprocessing import Pool
from functools import partial
import scanpy as sc
from magic_func import magic_imputation
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed(1738)
    TARGET_SUM = 300
    NUMBER_OF_CELLS = 10_000
    CAPTURE_RATE = 0.15

    # getting files
    path_to_dir = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/"
    output_path = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/magic/"
    dropout_file = path_to_dir + f"dropout_capture_rate={CAPTURE_RATE}.h5ad"
    merfish_file = path_to_dir + "merfish_norm.h5ad"

    merfish = sc.read_h5ad(merfish_file)
    dropout_adata = sc.read_h5ad(dropout_file)

    # Setup for grid search for MAGIC
    param_grid = [i for i in range(1, 11)]

    # using partial function to pass in default params
    run_imputation_on_dropout = partial(
        magic_imputation,
        counts_adata=dropout_adata,
        target_sum=TARGET_SUM,
        output_path=output_path
    )

    print("Processing running MAGIC imputation...")
    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        p.map(run_imputation_on_dropout, param_grid)

    print("Done!")
