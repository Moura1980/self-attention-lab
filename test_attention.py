import numpy as np
from attention import ScaledDotProductAttention


def main():
    Q = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    K = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    V = np.array([
        [10.0, 0.0],
        [0.0, 10.0],
        [5.0, 5.0],
    ])

    attn = ScaledDotProductAttention()
    output, weights = attn.forward(Q, K, V)

    # Checks básicos
    assert weights.shape == (3, 3)
    assert output.shape == (3, 2)

    # Softmax por linha
    row_sums = weights.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    print("Q:\n", Q)
    print("K:\n", K)
    print("V:\n", V)
    print("\nPesos de atenção (softmax por linha):\n", weights)
    print("\nOutput (Attention * V):\n", output)

if __name__ == "__main__":
    main()