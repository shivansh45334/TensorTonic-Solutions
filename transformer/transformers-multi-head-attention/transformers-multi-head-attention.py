import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, query_length, d_model = Q.shape
    key_length = K.shape[1]
    d_k = d_model // num_heads

    # 1. Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # 2. Split into multiple heads
    Q_heads = Q_proj.reshape(batch_size, query_length, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, key_length, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, key_length, num_heads, d_k)

    # 3. Move heads dimension forward
    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)

    # 4. Calculate scaled dot-product attention
    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)

    # 5. Numerically stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    # 6. Weighted sum of values
    head_outputs = attention_weights @ V_heads

    # 7. Concatenate heads
    concatenated = head_outputs.transpose(0, 2, 1, 3)
    concatenated = concatenated.reshape(batch_size, query_length, d_model)

    # 8. Final output projection
    output = concatenated @ W_o

    return output.astype(np.float64)