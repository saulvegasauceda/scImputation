from numpy.random import seed
from multiprocessing import Pool
from generate_synthetic_data import run_create_synthetic_dataset_pipeline
from impute import run_magic
from functools import partial
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    seed(1738)

    # Setup for generating synth cells
    output_path = "/Users/saulvegasauceda/Desktop/Kellis_UROP/synth_runs/"
    TARGET_SUM = None
    CAPTURE_RATE = 0.15
    NUMBER_OF_CELLS = 10_000
    Rna_species_characteristic_numbers = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 20, 40, 60, 300, 40]
    genes_characteristics = []
    for _ in range(10):
        genes_characteristics.extend(Rna_species_characteristic_numbers[:])

    print("Processing generating synthetic data...")
    ground_truth_adata, processed_tenx_adata = run_create_synthetic_dataset_pipeline(
        output_path,
        genes_characteristics, 
        number_cells=NUMBER_OF_CELLS, 
        capture_rate=CAPTURE_RATE,
        target_sum=TARGET_SUM,
        p=0.5
        )
    # Setup for grid search
    t_selections = [i for i in range(1, 7)]
    knn_dist_selection = ['euclidean', 'cosine']
    t_knn_dist_product = []
    for knn_dist in knn_dist_selection:
        for t in t_selections:
            t_knn_dist_product.append((t, knn_dist))

    # using partial function to pass in default params
    run_magic_on_tenx = partial(
        run_magic, 
        counts_adata=processed_tenx_adata, 
        target_sum=TARGET_SUM,
        output_path=output_path
        )

    print("Processing running MAGIC...")
    CPUS_TO_USE = os.cpu_count() // 3
    with Pool(CPUS_TO_USE) as p:
        p.map(run_magic_on_tenx, t_knn_dist_product)

    print("Done!")

