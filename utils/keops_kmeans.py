"""
================================
K-means clustering - PyTorch API
================================

The :meth:`pykeops.torch.LazyTensor.argmin` reduction supported by KeOps :class:`pykeops.torch.LazyTensor` allows us
to perform **bruteforce nearest neighbor search** with four lines of code.
It can thus be used to implement a **large-scale**
`K-means clustering <https://en.wikipedia.org/wiki/K-means_clustering>`_,
**without memory overflows**.

.. note::
  For large and high dimensional datasets, this script
  **outperforms its NumPy counterpart**
  as it avoids transfers between CPU (host) and GPU (device) memories.


"""

########################################################################
# Setup
# -----------------
# Standard imports:

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from IPython import embed
import ipdb


def run_kmeans(x, num_cluster_list, Niter, temperature, verbose=True):
    results = {'inst2cluster': [], 'centroids': [], 'Dist': [], 'density': []}

    for K in num_cluster_list:
        cl, centroids, Dist = KMeans(x, K, Niter, verbose)

        # sample-to-centroid distances for each cluster
        Dist = Dist.cpu()
        Dcluster = [[] for c in range(K)]
        for im, i in enumerate(cl):
            Dcluster[i].append(Dist[im][0])

        # print('Dcluster', len(Dcluster), len(Dcluster[0]), Dcluster[0][0])  # k, points_per_cluster, dist

        # concentration estimation (phi)
        density = np.zeros(K)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d
        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(
            np.percentile(density, 10), np.percentile(density, 90)
        )  # clamp extreme values for stability
        density = temperature * density / density.mean()  # scale the mean to temperature

        # centroids = F.normalize(centroids, p=2, dim=1)
        density = torch.Tensor(density).cuda()

        results['inst2cluster'].append(cl)
        results['centroids'].append(centroids)
        results['Dist'].append(Dist)
        results['density'].append(density)

    return results


########################################################################
# Simple implementation of the K-means algorithm:


def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    centroids = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(centroids.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        Dist = D_ij.min(dim=1)

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        centroids.zero_()
        centroids.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(centroids).view(K, 1)
        centroids /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        torch.cuda.synchronize()
        end = time.time()
        print(f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, centroids, Dist


def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    if verbose:  # Fancy display -----------------------------------------------
        torch.cuda.synchronize()
        end = time.time()
        print(f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c
