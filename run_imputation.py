import impute
import numpy as np
import pandas as pd

np.random.seed(1738)

# Defining hyperparams
t_selections = [i for i in range(1,7)]
knn_dist_selection = ['euclidean', 'cosine']

# get data
def run_parameter_sweep(true_adata, dropout_adata, t_selections, knn_dist_selection):
    # entries for final dataframe
    t_columns, knn_dist_columns, error_per_pair  = [], [], []

    for knn_dist in nn_dist_selection:
        for t in t_selections:
            imputed_counts = impute.run_magic(dropout_adata, t, knn_dist)
            mse_for_params = impute.calculate_error(true_counts, imputed_counts)

            t_columns.append(t)
            knn_dist_columns.append(knn_dist)
            error_per_pair.append(mse_for_params)

    results = pd.DataFrame({
                            "t": t_columns,
                            "knn": knn_dist_columns,
                            "MSE": error_per_pair
                        })
    return results

    # multiprocessing package


    from multiprocessing import Pool


    pool = Pool(os.cpu_count() // 3)
    pool.map()
    
