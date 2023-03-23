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
    residual = (desired - actual)
    
    # if desired entries is 0: set to 1 so calculation is analogous to MSE
    augmented_desired = desired.copy()
    augmented_desired[np.where(augmented_desired==0)] = 1

    relative_residual = np.divide(residual, augmented_desired)

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

    #print(sqrt_mse)

    sum_desire = np.sum(desired)


    rrmse = sqrt_mse / sum_desire * 100
    return rrmse