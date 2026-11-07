import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    # Calculate the selected norm
    if norm_type == "l1":
        norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)

    elif norm_type == "l2":
        norm = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))

    elif norm_type == "max":
        norm = np.max(np.abs(matrix), axis=axis, keepdims=True)

    else:
        raise ValueError("Invalid norm_type")

    # Replace zero norms with 1 to avoid division by zero
    norm = np.where(norm == 0, 1, norm)

    # Normalize
    return matrix / norm