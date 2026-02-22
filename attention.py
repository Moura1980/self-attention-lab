import numpy as np


class ScaledDotProductAttention:
  
    def __init__(self, eps: float = 1e-9):
        self.eps = eps

    def softmax(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + self.eps)

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray):
        
        if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
            raise ValueError("Q, K, V devem ser matrizes 2D (n_tokens, d).")

        n_q, d_k_q = Q.shape
        n_k, d_k_k = K.shape
        n_v, d_v = V.shape

        d_k = d_k_q

        # 1) Scores = Q K^T  -> (n_tokens, n_tokens)
        scores = Q @ K.T

        # 2) Scaling factor
        scores_scaled = scores / np.sqrt(d_k)

        attn_weights = self.softmax(scores_scaled, axis=1)

        output = attn_weights @ V

        return output, attn_weights