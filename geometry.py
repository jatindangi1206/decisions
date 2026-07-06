"""Pure-numpy representation metrics on the fixed probe. No torch here.

Headline = mutual-kNN (local neighbor agreement). CKA is logged too but is the secondary,
global metric — it tends to wash out under calibration (the Aristotelian critique), so it is
NOT the number the verdict is built on. Participation ratio tracks effective dimensionality.
"""
import numpy as np


def _sqdist(X):
    g = X @ X.T
    d = np.diag(g)
    return np.maximum(d[:, None] + d[None, :] - 2 * g, 0.0)


def _knn_idx(X, k):
    d = _sqdist(X)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, 1)[:, :k]


def mutual_knn(X, Y, k):
    """Fraction of each point's k nearest neighbours that are shared between spaces X and Y.
    Used both for between-seed agreement (X, Y = two seeds' latents) and for anchor alignment
    (X = latents, Y = Z or V). Rank-based, so relative scale of X vs Y does not matter."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    Y = np.atleast_2d(np.asarray(Y, np.float64))
    if X.shape[1] == 1:  # 1-D anchor (e.g. V): argsort ties are fine
        pass
    a, b = _knn_idx(X, k), _knn_idx(Y, k)
    overlap = [len(set(a[i]) & set(b[i])) for i in range(len(a))]
    return float(np.mean(overlap)) / k


def linear_cka(X, Y):
    X = np.asarray(X, np.float64) - np.asarray(X, np.float64).mean(0)
    Y = np.asarray(Y, np.float64) - np.asarray(Y, np.float64).mean(0)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    n = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (n + 1e-12))


def participation_ratio(X):
    X = np.asarray(X, np.float64) - np.asarray(X, np.float64).mean(0)
    lam = np.linalg.eigvalsh((X.T @ X) / len(X))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (np.square(lam).sum() + 1e-12))


def gaussian_null_knn(n, k, reps=5, seed=0):
    """Chance-level mutual-kNN between two independent Gaussian feature sets — the absolute
    floor. Any real between-seed agreement must clear this to mean anything."""
    rng = np.random.default_rng(seed)
    return float(np.mean([mutual_knn(rng.standard_normal((n, 8)),
                                     rng.standard_normal((n, 8)), k) for _ in range(reps)]))


def collapse_test(E, Z, V, k):
    """The reality-vs-value probe made concrete. Find state pairs FAR apart in reality (Z) but
    CLOSE in value (V) — these exist only because walls separate states that are similarly close
    to the goal. Report the encoder's normalized distance on those "value pairs" vs the mirror
    "reality pairs" (close in Z, far in V). If value-pair distance shrinks as competence rises,
    the encoder is collapsing reality-distinct-but-value-equal states -> drifting toward VALUE.
    """
    E, Z = np.asarray(E, np.float64), np.asarray(Z, np.float64)
    V = np.asarray(V, np.float64).reshape(-1)
    dZ = np.sqrt(_sqdist(Z))
    dV = np.abs(V[:, None] - V[None, :])
    dE = np.sqrt(_sqdist(E))
    iu = np.triu_indices(len(E), 1)
    dZu, dVu, dEu = dZ[iu], dV[iu], dE[iu]
    dEn = dEu / (dEu.max() + 1e-12)
    zhi, zlo = np.percentile(dZu, 80), np.percentile(dZu, 20)
    vhi, vlo = np.percentile(dVu, 80), np.percentile(dVu, 20)
    value_pairs = (dZu >= zhi) & (dVu <= vlo)     # far in reality, near in value
    reality_pairs = (dZu <= zlo) & (dVu >= vhi)   # near in reality, far in value
    return {
        "value_pair_enc_dist": float(dEn[value_pairs].mean()) if value_pairs.any() else None,
        "reality_pair_enc_dist": float(dEn[reality_pairs].mean()) if reality_pairs.any() else None,
        "n_value_pairs": int(value_pairs.sum()),
        "n_reality_pairs": int(reality_pairs.sum()),
    }


def between_seed(latents, k):
    """Average mutual-kNN over all unordered pairs of seed latents at one (objective, step)."""
    L = list(latents)
    scores = [mutual_knn(L[i], L[j], k) for i in range(len(L)) for j in range(i + 1, len(L))]
    return float(np.mean(scores)) if scores else float("nan")


if __name__ == "__main__":
    # self-check: identical spaces -> mutual-kNN 1.0; independent Gaussians -> near the null.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 16))
    assert abs(mutual_knn(A, A.copy(), 10) - 1.0) < 1e-9, "identity must be 1.0"
    assert mutual_knn(A, rng.standard_normal((200, 16)), 10) < 0.15, "independent must be low"
    assert linear_cka(A, A.copy()) > 0.999, "CKA identity"
    # a rotation preserves neighbours -> mutual-kNN stays 1.0
    Q = np.linalg.qr(rng.standard_normal((16, 16)))[0]
    assert abs(mutual_knn(A, A @ Q, 10) - 1.0) < 1e-9, "rotation invariance"
    print("geometry self-check OK")
