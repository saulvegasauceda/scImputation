import numpy as np
import scanpy as sc
from evaluation_metrics import calculate_rmsre
from sklearn.decomposition import NMF

path_to_dir = "/Users/saulvegasauceda/Desktop/Kellis_UROP/synth_runs/"
dropout_file = path_to_dir + "dropout_capture_rate=0.15_synth.h5ad"
ground_truth_file = path_to_dir + "ground_truth_synth.h5ad"

dropout_adata = sc.read_h5ad(dropout_file)
ground_truth_adata = sc.read_h5ad(ground_truth_file)
dropout_matrix = dropout_adata.X
ground_truth_matrix = ground_truth_adata.X

components = [
    1, 2, 4, 6, 8, 10
]

calculated_loss = [calculate_rmsre(dropout_matrix, ground_truth_matrix)]
for dim in components:
    imputed_adata = dropout_adata.copy()

    # Run NMF
    model = NMF(n_components=dim, init='nndsvda', random_state=0, solver='mu')
    W = model.fit_transform(dropout_matrix)
    H = model.components_
    imputed_matrix = W @ H

    imputed_adata.X = imputed_matrix
    sc.pp.normalize_total(imputed_adata, exclude_highly_expressed=True)

    error = calculate_rmsre(ground_truth_matrix, imputed_adata.X)
    calculated_loss.append(error)

average_loss = sum(calculated_loss) / len(calculated_loss)

# saving results
output_path = "/Users/saulvegasauceda/Desktop/Kellis_UROP/synth_runs/"
output_file = output_path + "nmf_components=10_imputed_synth.h5ad"
imputed_adata.write_h5ad(output_file, compression='gzip')