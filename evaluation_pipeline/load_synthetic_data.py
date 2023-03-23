import scanpy as sc

# Functions for getting files
def get_synthetic_data_paths(
        output_path="./", 
        ground_truth_file="ground_truth_synth.h5ad", 
        dropout_file="dropout_capture_rate=0.6_synth.h5ad", 
        parameters=[]):
    """
    Inputs:
        output_path: path to directory containing the synthetic data (must end w/ "/")
        ground_truth_file: file name of ground truth dataset
        dropout_file: file name of original dropout data from ground truth
        parameters: list of tuples containing parameters of grid search 
                    (params should be in the same order as passed to imputaiton pipeline)
    Returns:
       synthetic_data_paths: list of paths of all synthetic data files, where synthetic_data_paths[0] 
                             contains path of groud truth data, synthetic_data_paths[1] is path to original 
                             dropout, and synthetic_data_paths[2:] contains paths of imputed counts
    """
    assert output_path[-1] == "/", "output_path must end with forwardslash '/'"

    synthetic_data_paths = [
        output_path + ground_truth_file,
        output_path + dropout_file,
    ]

    for (t, knn_dist, rescaled) in parameters:
        file_name = f"t={t}_knn_dist={knn_dist}_rescale={rescaled}_imputed_synth.h5ad"
        file_path = output_path + file_name
        synthetic_data_paths.append(file_path)

    return synthetic_data_paths

def generate_params():
    t_selections = [i for i in range(1, 7)]
    knn_dist_selection = ['euclidean', 'cosine']
    rescaled_values = [True, False]

    parameters = [(t, knn_dist, rescaled) for knn_dist in knn_dist_selection
                                              for t in t_selections
                                                  for rescaled in rescaled_values]
    return parameters

# Pipeline
def run_evaluation_pipeline(ground_truth_matrix, data_to_compare_path, metrics):
    """
    Calculating error for every metric specified in metrics

    Inputs: 
        ground_truth_matrix: 2D numpy array corresponding to desire values
        metrics: tuple of functinos that calculate evaluation metrics
        data_to_compare_path: file path to counts that will be compared to ground_truth_matrix
    Returns:
        error_values: tuple of scalar values for each metric in metrics
    """
    other_adata = sc.read_h5ad(data_to_compare_path)
    other_data_matrix = other_adata.X

    error_values = (calculate_metric(ground_truth_matrix, other_data_matrix) 
                        for calculate_metric in metrics)

    return error_values
