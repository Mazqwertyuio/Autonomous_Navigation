import numpy as np


def bimodality(samples_xy, iters=20, seed=0):
    """samples: (S,K,2). k-means (k=2) за кінцевими точками.
    Повертає (separation, sizes): відстань між центрами / середній розкид кластерів."""
    end = np.asarray(samples_xy)[:, -1, :]
    rng = np.random.default_rng(seed)
    c = end[rng.choice(len(end), 2, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((end[:, None] - c[None]) ** 2).sum(-1), axis=1)
        for k in range(2):
            if (lab == k).any():
                c[k] = end[lab == k].mean(0)
    spread = np.mean([np.linalg.norm(end[lab == k] - c[k], axis=1).mean() if (lab == k).any() else 0
                      for k in range(2)])
    sep = float(np.linalg.norm(c[0] - c[1]))
    return sep / (spread + 1e-6), [int((lab == k).sum()) for k in range(2)]
