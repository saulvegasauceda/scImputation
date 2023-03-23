import scanpy as sc
import numpu as np

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

# Evaluation metrics
def calculate_mse(desired, actual):
    """
    Calculating MSE

    Inputs: 
        desired: 2D numpy array corresponding to desire values
        actual: 2D numpy array corresponding to actual values
        (order does not matter)
    Returns:
        rrmse: scalar value of mean square error
    """
    residual = desired - actual
    squared_residual = np.square(residual)
    mse = squared_residual.mean(axis=None)
    return mse

def calculate_rmsre(desired, actual):
    """
    Calculating RMSRE according to: https://stats.stackexchange.com/q/413217

    Inputs: 
        desired: 2D numpy array corresponding to desire values
        actual: 2D numpy array corresponding to actual values
    Returns:
        rmsre: scalar value of root mean square relative error
    """
    residual = desired - actual
    relative_residual = np.divide(residual, desired)
    squared_relative_residual = np.square(relative_residual)
    rmsre = np.sqrt(squared_relative_residual.mean(axis=None))
    return rmsre

def calculate_rrmse(desired, actual):
    """
    Calculating RRMSE according to: https://stats.stackexchange.com/q/413217

    Inputs: 
        desired: 2D numpy array corresponding to desire values
        actual: 2D numpy array corresponding to actual values
    Returns:
        rrmse: scalar value of relative root mean square error
    """
    mse = calculate_mse(desired, actual)
    sqrt_mse = np.sqrt(mse)

    squared_desire = np.squared(desired)
    sum_squared_desire = np.sum(squared_desire)
    sqrt_sum_squared_desire = np.sqrt(sum_squared_desire)

    rrmse = sqrt_mse / sqrt_sum_squared_desire
    return rrmse
