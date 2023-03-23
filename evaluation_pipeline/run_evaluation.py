import pandas as pd
import numpy as np
import scanpy as sc
import evaluation_metrics
import load_synthetic_data
from multiprocessing import Pool
from functools import partial
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    output_path = "/Users/saulvegasauceda/Desktop/Kellis_UROP/synth_runs/"

    parameters = load_synthetic_data.generate_params()
    all_files = load_synthetic_data.get_synthetic_data_paths(output_path = output_path, parameters = parameters)
    ground_truth_file, dropout_files = all_files[0], all_files[1:]

    ground_truth_adata = sc.read_h5ad(ground_truth_file)
    
    metrics = (
        evaluation_metrics.calculate_mse,
        evaluation_metrics.calculate_rmsre,
        evaluation_metrics.calculate_rrmse
    )

    # using partial function to pass in default params
    run_evaluation_on_ground_truth = partial(
        load_synthetic_data.run_evaluation_pipeline, 
        ground_truth_matrix = ground_truth_adata.X, 
        metrics = metrics
        )

    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        error_per_param = p.map(run_evaluation_on_ground_truth, dropout_files)
    
    t_columns, knn_dist_columns, rescaled_columns = zip(*parameters)
    # Adding non-imputed dropout counts
    t_columns = (np.NAN,) + t_columns
    knn_dist_columns = (np.NAN,) + knn_dist_columns
    rescaled_columns = (np.NAN,) + rescaled_columns

    mse_columns, rmsre_columns, rrmse_columns = zip(*error_per_param)

    results = pd.DataFrame(
        {
            "t": t_columns,
            "knn": knn_dist_columns,
            "rescaled": rescaled_columns,
            "MSE": mse_columns,
            "RMSRE": rmsre_columns,
            "RRMSE": rrmse_columns
        }
    )

    results.to_csv("10k_cells_imputation_evaluation.csv")