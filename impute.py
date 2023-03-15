import magic
import pandas as pd
import scanpy as sc
import numpy as np
import scanpy.external as sce
from numpy.random import seed
from random import choices
from scipy.stats import rv_discrete
from scipy.stats import nbinom

def create_synthetic_cells(rna_species_char, p, number_cells=100):
    '''
    Inputs
    --------
    p: the probability of a single success (parameter to scipy.stats.nbinom)
    rna_species_char: List of size N representing the profile of N RNA species
    number_cells: number of cells (rows) in synthetic dataset
    
    Sampling using a negative binomial distribution where n = # of median RNA
    
    Outputs
    --------
    Pandas dataframe of size (number_cells x N) where each row represents a cell
    sampled from rna_species_char.
    '''
    cell_generator = [nbinom(n, p) for n in rna_species_char]
    ground_truth_df = pd.DataFrame([dist.rvs(number_cells) for dist in cell_generator]).T
    ground_truth_df = ground_truth_df.set_axis([f"Gene {g + 1}" for g in range(len(ground_truth_df.columns))], axis=1, inplace=False)
    return ground_truth_df



def artificially_sample_cells(true_cells_df, capture_rate):
    '''
    Simulating Bernoulli sampling with synthetic dataset 
    where p = capture_rate (p := probability RNA is included in set)
    '''
    sim_capture = lambda x, p: sum(choices([0, 1], weights=[1-p, p])[0] for _ in range(x))
    return true_cells_df.applymap(lambda x: sim_capture(x, p=capture_rate))

def processing(counts_adata, target_sum=None):
    '''
    Run processing steps before MAGIC
    '''

    if ((scipy.sparse.issparse(counts_adata.X) and not np.all(np.mod(counts_adata.X.data, 1) == 0)) or
        (not scipy.sparse.issparse(counts_adata.X) and not np.all(np.mod(counts_adata.X, 1) == 0))):
        print('Warning: non-integer entries in adata.X. Likely not counts matrix.', flush=True)

    sc.pp.normalize_total(counts_adata,  target_sum=target_sum, exclude_highly_expressed=True)
    sc.pp.sqrt(counts_adata)
    return counts_adata

def run_magic(counts_adata, t, knn_dist, n_jobs=-1, verbose=True):
    """
    Run MAGIC modifying only t and knn_dist
    returns imputed counts data
    """
    sce.pp.magic(adata = counts_adata, 
                name_list='all_genes',
                solver='exact',
                t = t, 
                knn_dist = knn_dist, 
                n_jobs = n_jobs, 
                verbose = verbose)
    counts_adata.X = counts_adata.X**2
    return counts_adata

def calculate_error(true_adata, imputed_adata):
    mse = (np.square(true_adata.X - imputed_adata.X)).mean()
    return mse

def run_create_synthetic_dataset_pipeline(rna_species_char, number_cells=100, capture_rate=0.6, p=0.5):
    ground_truth_df = create_synthetic_cells(rna_species_char, p, number_cells)
    dropout_df = artificially_sample_cells(ground_truth_df, capture_rate)

    ground_truth_adata = sc.AnnData(ground_truth_df)
    dropout_adata = sc.AnnData(dropout_df)
    return ground_truth_adata, dropout_adata

def run_magic_evaluation_pipeline(true_adata, procesed_adata, t, knn_dist, n_jobs=-1):
    imputed_counts = run_magic(procesed_adata, t, knn_dist, n_jobs)
    error = calculate_error(true_adata, imputed_adata)
    return error

def run_magic_evaluation_pipeline_synth_data(t, knn_dist, n_jobs=-1):
    '''
    Partial function on the same ground truth dataset
    '''
    return run_magic_evaluation_pipeline(true_adata=synth_data, procesed_adata=procesed_adata, t=t, knn_dist=knn_dist, n_jobs=n_jobs)