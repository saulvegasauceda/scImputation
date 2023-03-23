import numpy as np

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

    squared_desire = np.square(desired)
    sum_squared_desire = np.sum(squared_desire)
    sqrt_sum_squared_desire = np.sqrt(sum_squared_desire)

    rrmse = sqrt_mse / sqrt_sum_squared_desire
    return rrmse