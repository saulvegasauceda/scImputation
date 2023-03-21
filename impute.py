import magic
import pandas as pd
import scanpy as sc
import numpy as np
import scanpy.external as sce

def rescale_after_magic(magic_matrix, original_matrix):
    '''
    Function from older version of MAGIC to rescale after imputation
    Arguments:
        data: dense array of imputed counts (adata.X after magic)
        original_matrix:  sparse array, original normalized counts (adata.X before magic)
    '''
    magic_matrix[magic_matrix < 0] = 0                                                          # flatten the imputed data to zero
    original_matrix = np.squeeze(np.asarray(original_matrix.todense()))                         # get the sparse matrix as dense

    M100 = original_matrix.max(axis=0)                                                          # get the max value for each gene
    M99 = np.percentile(original_matrix, 99, axis=0)                                            # find the 99the percentile value for each gene
    indices = np.where(M99 == 0)[0]                                                             # all the places the 99th percentile is zero
    M99[indices] = M100[indices]                                                                # replace those with the max value

    M100_new = magic_matrix.max(axis=0)                                                         # now where the 100th percentile is
    M99_new = np.percentile(magic_matrix, 99, axis=0)                                           # now where the 99th percentile is
    indices = np.where(M99_new == 0)[0]                                                         # again, find where the new 99th percentile is zero
    M99_new[indices] = M100_new[indices]                                                        # and replace it with the 100th percentile
    max_ratio = np.divide(M99, M99_new)                                                         # the ratio of the old 99th percentile to the new 99th percentile
    max_ratio[~np.isfinite(max_ratio)] = 1                                                      # if you had any all-zero columns, you would have gotten an error
    magic_matrix = np.multiply(magic_matrix, np.tile(max_ratio, (original_matrix.shape[0], 1))) # do the rescaling

    return magic_matrix

def run_magic(counts_adata, t, knn_dist, output_path, n_jobs=-1, verbose=True):
    """
    Run MAGIC modifying only t and knn_dist
    saves imputed and imputed-rescaled counts as h5ad

    output_path: should end with "/"
    """
    magic_file = output_path + f"{t}_{knn_dist}_imputed_synth.h5ad"
    rescaled_file = output_path + f"{t}_{knn_dist}_imputed_rescaled_synth.h5ad"

    magic_adata = sce.pp.magic(
                                adata = counts_adata, 
                                name_list='all_genes',
                                solver='exact',
                                t = t, 
                                knn_dist = knn_dist, 
                                n_jobs = n_jobs, 
                                verbose = verbose
                                )

    print('Rescaling matrix...')
    rescaled_matrix = rescale_after_magic(magic_adata.X.copy(), counts_adata.X)
    #np.square(rescaled_matrix)
    rescaled_adata = anndata.AnnData(X=rescaled_matrix, obs=counts_adata.obs, var=counts_adata.var)

    print("Saving h5ad of imputed counts...")
    magic_adata.write_h5ad(magic_file, compression='gzip')
    rescaled_adata.write_h5ad(rescaled_file, compression='gzip')

def run_magic_pipeline(t_knn_dist_prod, n_jobs=-1):
    t, knn_dist = t_knn_dist_prod
    output_path = "/Users/saulvegasauceda/Desktop/Kellis_UROP/synth_runs/"
    print("t:", t)
    print("knn_dist:", knn_dist)
    run_magic(processed_tenx_adata, t, knn_dist, output_path, n_jobs)
