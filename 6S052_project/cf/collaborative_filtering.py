import copy
import numpy as np
from sklearn.linear_model import Ridge
import scanpy as sc


def fit_linear_regression_model(X, y, lmbd):
    '''
    Fits a linear regression model on the data (X, y) with regularization constant lmbd. 
    Returns the learned regression coefficients theta

    Parameters
    ----------
    X: np.array
        2-D feature matrix
    y: np.array
        1-D array of targets
    lmbd: float
        Regularization penalty weight

    Returns
    ----------
    model._coef: np.array
        Vector of trained model coefficents
    '''
    model = Ridge(alpha=lmbd, fit_intercept=False)
    model.fit(X, y)
    return model.coef_

def get_cell_gene_counts(cell_k, Y):
    '''
    Creates vector of the gene counts in Y observed by cell at row cell_k, and 
    returns the column indices of the genes the cell expressed.

    Parameters
    ----------
    cell_k: int in [1,N]
        row index of cell in gene counts matrix
    Y: np.array of size N x M
        matrix of gene counts

    Returns
    ----------
    Y_cell: np.array of floats
        List of observed gene counts for user at row user_index
    has_expression: np.array of ints
        List of column indices of movies that the user has rated
    '''
    Y_cell = Y[cell_k]
    # bool vector indicating which movies the user has rated
    has_expression_bool = Y_cell != 0
    Y_cell = Y_cell[has_expression_bool]
    # get indices where has_expression_bool == true
    has_expression = np.arange(Y.shape[1])[has_expression_bool]
    return Y_cell, has_expression

def compute_mse(X, Y):
    '''
    Computes mean squared error between predicted ratings X and and ground truth ratings Y,
    for observed datapoints (Y != 0).
    
    Parameters
    ----------
    X: np.array
        n x m matrix, representing predicted ratings
    Y: np.array
        n x m matrix of observed movie ratings

    Returns
    ----------
    mse: float
        mean squared error between predicted and observed movie ratings
    '''
    # filter out predictions for zero (unobserved) entries
    X_obs = copy.deepcopy(X)
    X_obs = X_obs[Y != 0].flatten()
    Y_obs = copy.deepcopy(Y)
    Y_obs = Y_obs[Y != 0].flatten()
    mse = np.mean((X_obs - Y_obs) ** 2)
    return mse

def get_gene_counts_for_gene(gene_k, Y):
    '''
    Creates vector of the gene counts in Y expressed for the gene at column gene_k, and 
    returns the row indices of the cells that have expressed the gene.

    Parameters
    ----------
    gene_k: int in [1,N]
        column index of user in ratings matrix
    Y: np.array of size N x M
        matrix of gene counts

    Returns
    ----------
    Y_gene: np.array of floats
        List of observed ratings for movie at column movie_index
    has_expression: np.array of ints
        List of row indices of users that have rated the movie
    '''
    Y_gene = Y[:, gene_k]
    # bool vector indicating which users have rated the movie
    has_expression_bool = Y_gene != 0
    Y_gene = Y_gene[has_expression_bool]
    # get indices where has_expression_bool == true
    has_expression = np.arange(Y.shape[0])[has_expression_bool]
    return Y_gene, has_expression

def run_alternating_least_squares(Y, d=10, lmbd=0.1, n_it=100, verbose=False, print_n=1):
    '''
    Runs the alternating least squares algorithm for matrix factorization.
    Estimates the matrix underlying observations Y as X = UV^T, where
    U and V are N x d and M x d matrices, respectively.

    Parameters
    ----------
    Y: np.array of size N x M
        matrix of movies ratings
    d: int
        Rank of estimated matrix X
    lmbd: float
        Weight of regularization penalty
    n_it: int
        Number of iterations to train for
        # NOTE: you can set this lower to help develop your algorithm quickly,
        but make sure to set back to 20 to compute your final results
    verbose: bool
        If true, will print training progress
    print_n: int
        If verbose is set to true, will print train MSE every {print_n} iterations

    Returns
    ----------
    U: np.array
        matrix of size N x d representing user features

    V: np.array 
        matrix of size of size V x d representing movie features
    '''
    n_cells, n_genes = Y.shape
    # initialize U, V, where predictions X = UV^T
    U = np.random.normal(0, 1, size=(n_cells, d))
    V = np.random.normal(0, 1, size=(n_genes, d))

    # for checking convergence
    n_it_since_improvement = 0
    best_mse = float("inf")

    for it in range(n_it):
        # (1) Fix V, optimize U
        # Hint: make sure to skip over users that have zero observed ratings
        # and to use (some of) the helper functions above!
        for k, u_k in enumerate(U):
            Y_cell, has_expression = get_cell_gene_counts(k, Y)

            # skip over users that have zero observed ratings
            if (has_expression.size == 0):
                continue

            V_n = np.take(V, has_expression, axis=0)
            # transpose y to get n x 1
            new_u_k = fit_linear_regression_model(V_n, Y_cell, lmbd)
            U[k][:] = new_u_k

        # (2) Fix U, optimize V
        # Hint: make sure to skip over movies that have zero observed ratings
        # and to use (some of) the helper functions above!
        for k, v_k in enumerate(V):
            Y_gene, has_expression = get_gene_counts_for_gene(k, Y)

            # skip over movies that have zero observed ratings
            if (has_expression.size == 0):
                continue

            U_n = np.take(U, has_expression, axis=0)

            new_v_k = fit_linear_regression_model(U_n, Y_gene, lmbd)
            V[k][:] = new_v_k
        
        # check convergence
        X = np.matmul(U, V.T)
        train_mse = compute_mse(X, Y)
        if train_mse > best_mse:
            if n_it_since_improvement >= 3:
                break  # converged, stop training
            n_it_since_improvement += 1
        else:
            best_mse = train_mse
            n_it_since_improvement = 0

        if verbose and it % print_n == 0:
            print(f"train MSE at iteration {it}: {train_mse}")
        it += 1
    return U, V

def als_imputation(dims_lambda_pair, counts_adata, target_sum=None, output_path="/", verbose=False, print_n=1):
    """
    Computes the resulting imputed matrix
    """
    assert output_path[-1] == "/", "output_path must end with forwardslash '/'"
    dims, lmd = dims_lambda_pair

    imputed_file = output_path + f"als_dims={dims}_lambda={lmd}_imputed.h5ad"

    print(30*'-', flush=True)
    print("dim:", dims, flush=True)
    print("lambda:", lmd, flush=True)
    print(30*'-', flush=True)
    
    imputed_adata = counts_adata.copy()
    dropout_matrix = counts_adata.X.copy()

    U, V = run_alternating_least_squares(dropout_matrix, dims, lmd)
    imputed_matrix = U @ V.T

    # setting values to be positive
    negatives = np.where(imputed_matrix < 0)
    imputed_matrix[negatives] = 0
    imputed_adata.X = imputed_matrix

    sc.pp.normalize_total(imputed_adata, target_sum=target_sum,
                          exclude_highly_expressed=False)

    print("Saving h5ad of imputed counts...", flush=True)
    imputed_adata.write_h5ad(imputed_file, compression='gzip')
