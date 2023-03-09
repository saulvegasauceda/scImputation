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
    df = pd.DataFrame([dist.rvs(number_cells) for dist in cell_generator]).T
    df = df.set_axis([f"Gene {g + 1}" for g in range(len(df.columns))], axis=1, inplace=False)
    return df

sim_capture = lambda x, p: sum(choices([0, 1], weights=[1-p, p])[0] for _ in range(x))

def artificially_sample_cells(true_cells_df, capture_rate):
    '''
    Simulating Bernoulli sampling with synthetic dataset 
    where p = capture_rate (p := probability RNA is included in set)
    '''
    return true_cells_df.applymap(lambda x: sim_capture(x, p=capture_rate))

def processing(counts_df, target_sum=None):
    '''
    Run processing steps before MAGIC
    '''
    counts_adata = sc.AnnData(counts_adata)
    sc.pp.normalize_total(counts_adata, target_sum=target_sum)
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
