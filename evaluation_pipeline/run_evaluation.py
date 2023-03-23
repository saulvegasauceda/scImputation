from numpy.random import seed
import evaluate
from multiprocessing import Pool
from functools import partial
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    output_path = "/Users/saulvegasauceda/Desktop/Kellis_UROP/synth_runs/"

    parameters = evaluate.generate_params()
    all_files = evaluate.get_synthetic_data_paths(output_path = output_path, parameters = parameters)
    ground_truth_file, dropout_file, all_synthetic_files = all_files[0], all_files[1], all_files[2:]

    ground_truth_adata = sc.read_h5ad(ground_truth_file)
    
    metrics = [
        evaluate.calculate_mse,
        evaluate.calculate_rmsre,
        evaluate.calculate_rrmse
        ]

    # using partial function to pass in default params
    run_evaluation_on_ground_truth = partial(evaluate.run_evaluation_pipeline, 
        ground_truth_matrix = ground_truth_adata.X, 
        metrics = metrics
        )

    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        p.map(evaluate.run_evaluation_pipeline(ground_truth_matrix, data_to_compare_path, metrics), all_files)