# LAB P1-01 — Scaled Dot-Product Attention

Implementação *from scratch* do mecanismo de Self-Attention.

```bash
▶️ Como rodar
pip install numpy
python test_attention.py

🧠 Como a normalização (√dₖ) foi aplicada
Após calcular o produto escalar Q @ K.T, o resultado é dividido por:
scores_scaled = scores / np.sqrt(d_k)

📌 Softmax
O softmax é aplicado em cada linha da matriz de scores:
attn_weights = self.softmax(scores_scaled, axis=1)
Cada linha passa a representar uma distribuição de probabilidade (soma ≈ 1).

📊 Exemplo de entrada
Q = [[1,0], [0,1], [1,1]]
K = [[1,0], [0,1], [1,1]]
V = [[10,0], [0,10], [5,5]]

📊 Saída esperada
Matriz de atenção 3x3 (cada linha soma ~1)
Output final 3x2 após multiplicação pelos valores V
