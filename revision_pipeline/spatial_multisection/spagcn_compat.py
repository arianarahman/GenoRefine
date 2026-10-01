# Purpose: Device-compatible port of SpaGCN 1.2.7 ``simple_GC_DEC.fit``.
# Author: Ariana Rahman (Arizona State University)
# Upstream basis: SpaGCN 1.2.7, Copyright (c) 2020 JianHu, MIT License.
# Required notice: THIRD_PARTY_NOTICES.md

"""Device-compatible port of SpaGCN 1.2.7 ``simple_GC_DEC.fit``.

The update equations, initialization, target refresh interval, stopping rule and
optimizer mirror the vendored official implementation.  The only intended
difference is explicit tensor-device placement so the same API can use CUDA.
The Author line identifies the compatibility-port author; upstream copyright
and license terms are preserved in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from .common import ROOT, specification


def load_official_spagcn():
    spec = specification()
    package = (ROOT / spec["spagcn"]["vendored_path"] / "SpaGCN_package").resolve()
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    import SpaGCN as module
    if (getattr(module, "__version__", None) != spec["spagcn"]["version"]
            or not Path(module.__file__).resolve().is_relative_to(package)):
        raise RuntimeError("Did not import the pinned vendored SpaGCN package")
    return module


def seed_all(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid SpaGCN seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def fit_device_compatible(
    embedding: np.ndarray,
    adjacency_exp: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    learning_rate: float = 0.005,
    max_epochs: int = 2000,
    tolerance: float = 0.001,
    weight_decay: float = 0.0,
    update_interval: int = 3,
    kmeans_n_init: int = 20,
    model_alpha: float = 0.2,
    device: str = "cpu",
    reseed: bool = True,
) -> dict:
    """Fit the official one-layer GC-DEC objective on an explicit device."""
    if reseed:
        seed_all(seed)
    elif type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid SpaGCN seed")
    spg = load_official_spagcn()
    x_np = np.asarray(embedding, dtype=np.float32)
    adj_np = np.asarray(adjacency_exp, dtype=np.float32)
    if (x_np.ndim != 2 or adj_np.shape != (len(x_np), len(x_np))
            or not np.isfinite(x_np).all() or not np.isfinite(adj_np).all()
            or type(n_clusters) is not int or not 2 <= n_clusters < len(x_np)):
        raise ValueError("Invalid SpaGCN port inputs")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA SpaGCN requested but unavailable")
    if update_interval != 3 or kmeans_n_init != 20 or model_alpha != 0.2:
        raise ValueError("SpaGCN compatibility port settings differ from the locked official logic")
    model = spg.models.simple_GC_DEC(x_np.shape[1], x_np.shape[1], alpha=model_alpha).to(selected)
    # SpaGCN 1.2.7 constructs Adam before registering ``mu``; consequently its
    # cluster centers remain fixed.  Preserve that published implementation
    # detail for comparator fidelity rather than silently "fixing" it.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # FloatTensor and ``.data`` below intentionally mirror SpaGCN 1.2.7.
    # Device transfer is the sole computational extension.
    x = torch.FloatTensor(x_np).to(selected)
    adjacency = torch.FloatTensor(adj_np).to(selected)
    features = model.gc(x, adjacency)
    kmeans = KMeans(n_clusters, n_init=kmeans_n_init)
    predicted = kmeans.fit_predict(features.detach().cpu().numpy())
    model.n_clusters = n_clusters
    model.mu = torch.nn.Parameter(torch.empty(n_clusters, model.nhid, device=selected))
    feature_frame = pd.DataFrame(features.detach().cpu().numpy())
    groups = pd.Series(predicted, name="Group")
    centers = np.asarray(pd.concat([feature_frame, groups], axis=1).groupby("Group").mean())
    model.mu.data.copy_(torch.Tensor(centers).to(selected))
    model.train()
    predicted_last = predicted.copy()
    losses, deltas = [], []
    p = None
    epochs_completed = 0
    for epoch in range(max_epochs):
        if epoch % update_interval == 0:
            _, q = model(x, adjacency)
            p = model.target_distribution(q).data
        optimizer.zero_grad()
        latent, q = model(x, adjacency)
        loss = model.loss_function(p, q)
        loss.backward()
        optimizer.step()
        predicted = torch.argmax(q, dim=1).detach().cpu().numpy()
        # Mirror the published implementation's float32 numerator exactly;
        # this can matter when a change fraction is on the tolerance boundary.
        delta = float(np.sum(predicted != predicted_last).astype(np.float32) / x_np.shape[0])
        predicted_last = predicted.copy()
        losses.append(float(loss.detach().cpu()))
        deltas.append(delta)
        epochs_completed = epoch + 1
        if epoch > 0 and (epoch - 1) % update_interval == 0 and delta < tolerance:
            break
    with torch.no_grad():
        latent, probabilities = model(x, adjacency)
    probabilities_np = probabilities.detach().cpu().numpy()
    predicted_np = np.argmax(probabilities_np, axis=1).astype(np.int64)
    return {
        "latent": latent.detach().cpu().numpy(),
        "probabilities": probabilities_np,
        "predicted": predicted_np,
        "epochs_completed": epochs_completed,
        "losses": losses,
        "label_change_fraction": deltas,
        "device": str(selected),
        "model": model,
    }
