import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pos = np.arange(seq_length)[:, np.newaxis]

    i = np.arange(d_model)[np.newaxis, :]

    angles = pos / np.power(10000, (2 * (i // 2)) / d_model)

    encoding = np.zeros((seq_length, d_model))

    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding