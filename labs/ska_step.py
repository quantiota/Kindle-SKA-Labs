import math
import torch

LN2 = math.log(2)


def ska_step(X, W, b, D_prev, eta):
    """One SKA step for all layers, as pure tensor operations.

    X: (N, n_0) input batch.  W[l]: (n_l, n_{l-1}) weights.  b[l]: (n_l,) biases.
    D_prev: list of (N, n_l) decisions from the previous step, or None at step 0.
    Returns the new decisions and, from step 1 on, each layer's step entropy and cosine.
    """
    A, Z, D = X, [], []
    for Wl, bl in zip(W, b):                          # forward pass
        Zl = A @ Wl.T + bl                            # knowledge tensor  (N, n_l)
        A = torch.sigmoid(Zl)                         # decision tensor   (N, n_l)
        Z.append(Zl)
        D.append(A)
    if D_prev is None:
        return D, None, None
    dH, cos = [], []
    for l in range(len(W)):                           # independent per layer
        dD = D[l] - D_prev[l]                         # shift tensor
        dot = (Z[l] * dD).sum()                       # Frobenius inner product
        dH.append(-dot / LN2)                         # step entropy (bits)
        cos.append(dot / (Z[l].norm() * dD.norm() + 1e-12))
        G = -(Z[l] * D[l] * (1 - D[l]) + dD) / LN2    # entropy gradient tensor
        A_in = X if l == 0 else D_prev[l - 1]         # layer input, previous step
        W[l] -= eta * G.T @ A_in / X.shape[0]         # outer-product update
        b[l] -= eta * G.mean(0)
    return D, torch.stack(dH), torch.stack(cos)
