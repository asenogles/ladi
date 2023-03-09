import numpy as np

from ._ext import *

def v_dot_square(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """perform the dot product between a square DxD matrices and len(D) vector

    Args:
        X (np.ndarray): square matrix DxD matrix
        Y (np.ndarray): len(D) vector

    Returns:
        np.ndarray: result of X dot Y
    """
    assert(X.shape[0] == Y.shape[0])
    assert(X.shape[0] == X.shape[0])
    return cy_v_dot_square(X.astype(np.float32, order='C'), Y.astype(np.float32, order='C'))

def square_matrix_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """perform a matrix product between two square DxD matrices

    Args:
        X (np.ndarray): left square matrix DxD matrix
        Y (np.ndarray): right square matrix DxD matrix

    Returns:
        np.ndarray: matrix product of X & Y
    """
    assert(X.shape == Y.shape)
    return cy_square_matrix_product(X.astype(np.float32, order='C'), Y.astype(np.float32, order='C'))

def square_matrix_product_t(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """perform a matrix product between a square DxD matrix and the transpose of a second square DxD matrix

    Args:
        X (np.ndarray): left square matrix DxD matrix
        Y (np.ndarray): right square matrix DxD matrix (transposed)

    Returns:
        np.ndarray: matrix product of X & Y
    """
    assert(X.shape == Y.shape)
    return cy_square_matrix_product_t(X.astype(np.float32, order='C'), Y.astype(np.float32, order='C'))

def t_square_matrix_product(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """perform a matrix product between the transpose of a square DxD matrix and a second square DxD matrix

    Args:
        X (np.ndarray): left square matrix DxD matrix (transposed)
        Y (np.ndarray): right square matrix DxD matrix

    Returns:
        np.ndarray: matrix product of X & Y
    """
    assert(X.shape == Y.shape)
    return cy_t_square_matrix_product(X.astype(np.float32, order='C'), Y.astype(np.float32, order='C'))

def determinant(mat: np.ndarray) -> float:
    """Compute the determinant of a square matrix

    Args:
        mat (np.ndarray): square DxD matrix to compute determinant of

    Returns:
        float: determinant of mat
    """
    return cy_determinant(mat.astype(np.float32, order='C'))

def inverse(mat: np.ndarray) -> np.ndarray:
    """Compute the inverse of a square matrix

    Args:
        mat (np.ndarray): square DxD matrix to compute inverse of

    Returns:
        np.ndarray: inverse of mat
    """
    return cy_inverse_matrix(mat.astype(np.float32, order='C'))
