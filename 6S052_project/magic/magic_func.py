import magic
import scanpy as sc
import numpy as np
import scanpy.external as sce


def magic_imputation(t, counts_adata, target_sum, output_path):
    """
    Run MAGIC modifying only t and knn_dist
    saves imputed and imputed-rescaled counts as h5ad

    output_path: should end with "/"
    """
    assert output_path[-1] == "/", "output_path must end with forwardslash '/'"

    magic_file = output_path + f"magic_t={t}_imputed.h5ad"

    print(30*'-', flush=True)
    print("t:", t, flush=True)

    dropout_adata = counts_adata.copy()
    sc.pp.sqrt(dropout_adata)  # sqrt transformation

    magic_adata = sce.pp.magic(
        adata=dropout_adata,
        name_list='all_genes',
        solver='exact',
        t=t,
        knn_dist='euclidean',
        n_jobs=-1,
        copy=True,
        verbose=True
    )

    magic_adata.X = np.square(magic_adata.X)

    sc.pp.normalize_total(magic_adata, target_sum=target_sum,
                          exclude_highly_expressed=False)

    print("Saving h5ad of imputed counts...", flush=True)
    magic_adata.write_h5ad(magic_file, compression='gzip')
