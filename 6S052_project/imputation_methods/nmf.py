from sklearn.decomposition import NMF

def nmf_imputation(param_pair, counts_adata, target_sum, output_path):
    """
    Run nmf modifying only n_components & alpha_W
    saves imputed as h5ad

    output_path: should end with "/"
    """
    assert output_path[-1] == "/", "output_path must end with forwardslash '/'"

    dims, alpha_W = param_pair
    imputed_file = output_path + f"nmf_dims={dim}_alpha={alpha_W}_imputed.h5ad"
    
    print(30*'-', flush=True)
    print("dims:", dims, flush=True)
    print("alpha:", alpha_W, flush=True)
    print(30*'-', flush=True)

    imputed_adata = counts_adata.copy()
    dropout_matrix = counts_adata.copy().X

    # Run NMF
    model = NMF(n_components=dims, init='nndsvda', random_state=0, alpha_W=alpha_W)
    W = model.fit_transform(dropout_matrix)
    H = model.components_
    # Reconstruct matrix
    imputed_matrix = W @ H

    imputed_adata.X = imputed_matrix

    sc.pp.normalize_total(imputed_adata, target_sum=target_sum, exclude_highly_expressed=False)

    print("Saving h5ad of imputed counts...", flush=True)
    imputed_adata.write_h5ad(imputed_file, compression='gzip')