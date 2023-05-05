from numpy.random import seed
from multiprocessing import Pool
from functools import partial
import scanpy as sc
from nmf import nmf_imputation
import numpy as np
import os
import warnings

def rmse(dropout_file, true_counts_adata):
    imputed_counts = sc.read_h5ad(dropout_file) 
    imputed, merfish = imputed.X, true_counts_adata.X
    return np.sqrt(np.square(np.subtract(imputed,merfish)).mean())

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed(1738)
    TARGET_SUM = 500
    NUMBER_OF_CELLS = 10_000
    CAPTURE_RATE = 0.30

    # getting files
    path_to_dir = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/"
    input_path = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/nmf/"
    dropout_file = path_to_dir + "dropout_capture_rate=0.3.h5ad"
    merfish_file = path_to_dir + "merfish_norm.h5ad"
    output_file = path_to_dir + "nmf_evaluation.csv"

    merfish = sc.read_h5ad(merfish_file)

    # Setup to retrieve grid search files
    n_components_params = [1, 5, 10, 20, 40, 50, 100]
    alpha_W_params = [0, 0.2, 0.5, 0.7, 1]
    dropout_files = []
    params = []
    for dims in n_components_params:
        for alpha_W in alpha_W_params:
            file = input_path + f"nmf_dims={dims}_alpha={alpha_W}_imputed.h5ad"
            params.append((dims, alpha_W))
            dropout_files.append(file)

    # using partial function to pass in default params
    run_evaluation_pipeline = partial(
        rmse,
        counts_adata=merfish,
    )

    print("Processing running NMF imputation...")
    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        rmse_column = p.map(run_evaluation_pipeline, dropout_files)

    n_components_column, alpha_W_column = zip(*parameters)
    # Adding non-imputed dropout counts
    n_components_column = (np.NAN,) + n_components_column
    alpha_W_column = (np.NAN,) + alpha_W_column
    rmse_column = (rmse(dropout_file, merfish), ) + rmse_column

    results = pd.DataFrame(
        {
            "n_components": n_components_column,
            "alpha_w": alpha_W_column,
            "RMSE": rmse_columns,
        }
    )

    results.to_csv(output_file)

    print("Done!")
