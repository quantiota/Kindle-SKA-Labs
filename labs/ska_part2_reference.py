"""Reference runs for Foundations of Structured Knowledge Accumulation, Part II.

Same computation and data as Paper II's script (2504.0321.py):
network 784-256-128-64-10 with sigmoid layers, 100 MNIST training images per digit,
weights 0.01*N(0,1) and zero biases with torch seed 42,
forward-only SKA updates with the discrete entropy gradient.

Step convention (Paper I and Part I): forward passes are numbered 0, 1, ..., K-1; step k = 1, 2, ...
compares forward pass k with forward pass k-1, so a run of K steps has K-1 records, k = 1, ..., K-1.

Data: labs/dataset/mnist_subset_100_per_class.pt of quantiota/Kindle-SKA-Labs, the 1,000
images (100 per digit) that Paper II's script loads (the same file as Part I), read from
dataset/ next to this script.

Writes data/*.csv (in the current folder):
  ref_<metric>.csv        reference run, K = 50, eta = 0.01 (T = 0.5)
  ti_<metric>_K<K>.csv    the six eta*K = 0.5 runs of Paper II (entropy, cosine)
  budget_<metric>_T<T>.csv  K = 50 with T = 0.25 and T = 1.0

Run:  python ska_part2_reference.py        (from the labs/ folder)
"""
import os
import pathlib

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
NAME = "mnist_subset_100_per_class.pt"
OUT = pathlib.Path(os.environ.get("SKA_OUT", pathlib.Path.cwd() / "data"))
LAYERS = [256, 128, 64, 10]
LN2 = np.log(2)


def mnist_subset():
    """The 1,000 images (100 per digit) of Paper II's script, as an (N, 784) tensor."""
    for path in (HERE / "dataset" / NAME, HERE.parents[2] / "Kindle-SKA-Labs/labs/dataset" / NAME):
        if path.exists():
            items = torch.load(path)
            return torch.stack([img for img, _ in items]).view(len(items), -1)
    raise FileNotFoundError(f"{NAME}: get it from github.com/quantiota/Kindle-SKA-Labs (labs/dataset/)")


def run(X, K, eta, layers=LAYERS):
    """One SKA run; returns a dict of per-layer histories (lists of length K-1 for k = 1..K-1, Frobenius K)."""
    torch.manual_seed(42)
    np.random.seed(42)
    W, b, prev = [], [], X.shape[1]
    for n in layers:
        W.append(torch.randn(prev, n) * 0.01)
        b.append(torch.zeros(n))
        prev = n
    L = len(layers)
    h = {m: [[] for _ in range(L)] for m in
         ("entropy", "cosine", "flow", "net", "lagrangian", "net_printed", "weight")}
    h["frobenius"] = [[] for _ in range(L)]
    h["output"] = []
    Z_prev = D_prev = None
    for k in range(K):
        Z, D, x = [], [], X
        for l in range(L):
            z = x @ W[l] + b[l]
            d = torch.sigmoid(z)
            Z.append(z)
            D.append(d)
            h["frobenius"][l].append(torch.norm(z).item())
            x = d
        h["output"].append(D[-1].mean(0).numpy())
        if k > 0:
            for l in range(L):
                dD = D[l] - D_prev[l]
                dZ = Z[l] - Z_prev[l]
                Dp = D[l] * (1 - D[l])
                h["entropy"][l].append((-(1 / LN2) * (Z[l] * dD)).sum().item())
                h["cosine"][l].append(((Z[l] * dD).sum() / (Z[l].norm() * dD.norm())).item())
                h["flow"][l].append((dZ.norm() / eta).item())
                grad_c = -(1 / LN2) * Z[l] * Dp           # continuous gradient, Paper II
                h["net"][l].append((dZ * (D[l] + grad_c)).sum().item())        # Eq. 8, plus sign
                h["net_printed"][l].append((dZ * (D[l] - grad_c)).sum().item())  # printed sign
                h["lagrangian"][l].append((-(Z[l] * Dp * (dZ / eta))).sum().item())
            for l in range(L):                              # SKA update, discrete gradient
                a = X if l == 0 else D_prev[l - 1]
                g = -(1 / LN2) * (Z[l] * D[l] * (1 - D[l]) + (D[l] - D_prev[l]))
                W[l] = W[l] - eta * (a.T @ g) / a.shape[0]
                b[l] = b[l] - eta * g.mean(0)
                h["weight"][l].append(torch.norm(W[l]).item())
        Z_prev = [z.clone() for z in Z]
        D_prev = [d.clone() for d in D]
    return h


def save(h, prefix, metrics):
    OUT.mkdir(exist_ok=True)
    for m in metrics:
        data = np.array(h[m])
        cols = [f"k={j + 1}" for j in range(data.shape[1])] if m != "frobenius" else \
            [f"pass {j}" for j in range(data.shape[1])]
        pd.DataFrame(data, index=[f"Layer {l+1}" for l in range(data.shape[0])],
                     columns=cols).to_csv(OUT / f"{prefix}_{m}.csv")


if __name__ == "__main__":
    X = mnist_subset()
    ref = run(X, 50, 0.01)
    save(ref, "ref", ["entropy", "cosine", "flow", "net", "net_printed", "lagrangian",
                      "weight", "frobenius"])
    pd.DataFrame(np.array(ref["output"]), columns=[f"Neuron {i}" for i in range(10)]
                 ).to_csv(OUT / "ref_output.csv", index_label="forward pass")
    for eta, K in [(0.02, 25), (0.01, 50), (0.005, 100), (0.0033, 150), (0.0025, 200), (0.001, 500)]:
        save(run(X, K, eta), f"ti_K{K}", ["entropy", "cosine", "flow", "net"])
    for T in (0.25, 1.0):
        save(run(X, 50, T / 50), f"budget_T{T}", ["entropy", "flow", "net", "lagrangian"])
    for K in (100, 200):
        save(run(X, K, 0.5 / K), f"path_K{K}", ["entropy", "lagrangian"])
    print("wrote", OUT)
