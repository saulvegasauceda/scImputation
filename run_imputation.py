import magic
import pandas as pd
import scanpy as sc
import numpy as np
import scanpy.external as sce
from numpy.random import seed
from random import choices
import scipy
from scipy.stats import rv_discrete
from scipy.stats import nbinom
from multiprocessing import Pool
from impute import *
import os


if __name__ == '__main__':
      
    np.random.seed(1738)
    t_selections = [i for i in range(1, 6)]
    knn_dist_selection = ['euclidean', 'cosine', 'correlation']

    t_knn_product = []

    for knn_dist in knn_dist_selection:
        for t in t_selections:
            t_knn_product.append((t, knn_dist))

    print("Processing running MAGIC...")
    with Pool(os.cpu_count() // 3) as p:
        error_per_pair = p.map(run_magic_evaluation_pipeline_synth_data, t_knn_product)

    t_columns, knn_dist_columns = zip(*t_knn_product)
    results = pd.DataFrame({
                            "t": t_columns,
                            "distance_metric": knn_dist_columns,
                            "MSE": error_per_pair
                        })

    results.to_csv("small_magic_grid_search.csv")

