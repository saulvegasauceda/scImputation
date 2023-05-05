from numpy.random import seed
from multiprocessing import Pool
from functools import partial
import scanpy as sc
import numpy as np
import pandas as pd
import os
import warnings


def rmse(dropout_file, true_counts_adata):
    imputed_counts = sc.read_h5ad(dropout_file)
    imputed, merfish = imputed_counts.X, true_counts_adata.X
    return np.sqrt(np.square(np.subtract(imputed, merfish)).mean())


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
    output_file = path_to_dir + "magic_evaluation.csv"

    merfish = sc.read_h5ad(merfish_file)

    # Setup to retrieve grid search files
    t_column = [i for i in range(1, 11)]
    dropout_file = [output_path + f"magic_t={t}_imputed.h5ad" for t in range(1, 11)]

    # using partial function to pass in default params
    run_evaluation_pipeline = partial(
        rmse,
        true_counts_adata=merfish,
    )

    print("Evaluating NMF imputation...")
    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        rmse_column = p.map(run_evaluation_pipeline, dropout_files)

    # Adding non-imputed dropout counts
    t_column = (np.NAN,) + t_column
    rmse_column = [rmse(dropout_file, merfish)] + rmse_column

    results = pd.DataFrame(
        {
            "t": t_column,
            "RMSE": rmse_column,
        }
    )

    results.to_csv(output_file)

    print("Done!")
