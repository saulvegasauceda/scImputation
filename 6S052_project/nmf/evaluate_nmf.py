from numpy.random import seed
from multiprocessing import Pool
from functools import partial
import scanpy as sc
import numpy as np
import pandas as pd
import os
import warnings


def calculate_rmse(dropout_file, true_counts_adata):
    imputed_counts = sc.read_h5ad(dropout_file)
    imputed, merfish = imputed_counts.X, true_counts_adata.X
    return np.sqrt(np.square(np.subtract(imputed, merfish)).mean())

def calculate_rmsre(actual_file, desired_adata):
    """
    Calculating RMSRE according to: https://stats.stackexchange.com/q/413217

    Inputs: 
        dropout_file: path to adata of gene counts
        true_counts_adata: adata of true counts
    Returns:
        rmsre: scalar value of root mean square relative error
    """
    actual_adata = sc.read_h5ad(actual_file)
    actual, desired = actual_adata.X, desired_adata.X
    residual = (desired - actual)

    # if desired entries is 0: set to 1 so calculation is analogous to MSE
    augmented_desired = desired.copy()
    augmented_desired[np.where(augmented_desired == 0)] = 1

    relative_residual = np.divide(residual, augmented_desired)

    squared_relative_residual = np.square(relative_residual)
    rmsre = np.sqrt(squared_relative_residual.mean(axis=None))
    return rmsre


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed(1738)
    TARGET_SUM = 500
    NUMBER_OF_CELLS = 10_000
    CAPTURE_RATE = 0.30

    # getting files
    path_to_dir = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/"
    input_path = "/Users/saulvegasauceda/Documents/Spring_23/6.S052/data/nmf/"
    dropout_file = path_to_dir + f"dropout_capture_rate={CAPTURE_RATE}.h5ad"
    merfish_file = path_to_dir + "merfish_norm.h5ad"
    output_file = path_to_dir + "nmf_evaluation.csv"

    merfish = sc.read_h5ad(merfish_file)

    # Setup to retrieve grid search files
    n_components_params = [1, 5, 10, 20, 40, 50, 100]
    alpha_W_params = [0, 0.2, 0.5, 0.7, 1]
    imputed_files = []
    parameters = []
    for dims in n_components_params:
        for alpha_W in alpha_W_params:
            file = input_path + f"nmf_dims={dims}_alpha={alpha_W}_imputed.h5ad"
            parameters.append((dims, alpha_W))
            imputed_files.append(file)

    # using partial function to pass in default params
    run_evaluation_pipeline = partial(
        calculate_rmsre,
        desired_adata=merfish,
    )

    print("Evaluating NMF imputation...")
    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        rmsre_column = p.map(run_evaluation_pipeline, imputed_files)

    n_components_column, alpha_W_column = zip(*parameters)
    # Adding non-imputed dropout counts
    n_components_column = (np.NAN,) + n_components_column
    alpha_W_column = (np.NAN,) + alpha_W_column
    rmsre_column = [calculate_rmsre(dropout_file, merfish)] + rmsre_column

    results = pd.DataFrame(
        {
            "n_components": n_components_column,
            "alpha_w": alpha_W_column,
            "RMSRE": rmsre_column,
        }
    )

    results.to_csv(output_file)

    print("Done!")
