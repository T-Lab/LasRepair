import os

import numpy as np
import pandas as pd
from openai import OpenAI


def embedding2graph(clean_csv, num_rows=20, dimensions=200):
    """
    read in the clean part of the data, choose at least num_rows(20) and compute the similarity matrix
    dimensions: the dimension of the embedding
    output the similarity matrix
    """
    clean_csv = clean_csv.sample(min(num_rows, len(clean_csv)))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    for col in clean_csv.columns:
        input_list = clean_csv[col].astype(str).tolist()
        input_text = col + ": " + " ".join(input_list)
        response = client.embeddings.create(
            input=input_text,
            model="text-embedding-3-small",
            dimensions=dimensions
        )
        embeddings.append(response.data[0].embedding)

    similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j])

    return similarity_matrix


def prompt_casual(clean_csv, num_rows=20, dimensions=200):
    # claude
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def modularity(A, labels, resolution=1.0):
    """
    计算给定划分 labels 的 modularity（加权图）。
    A: (n,n) adjacency matrix, assumed symmetric
    labels: array-like of size n, community id for each node (ints)
    resolution: gamma parameter
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square")
    # ensure symmetry
    A = (A + A.T) / 2.0
    k = A.sum(axis=1)
    m2 = k.sum()  # equals 2*m
    if m2 == 0:
        return 0.0
    unique = np.unique(labels)
    Q = 0.0
    for c in unique:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        subA = A[np.ix_(idx, idx)]
        sum_in = subA.sum()  # sum of internal weights (each edge counted twice if i!=j)
        # for weighted graphs modularity definition with 1/(2m):
        kc = k[idx].sum()
        Q += (sum_in - resolution * (kc * kc) / m2)
    Q = Q / m2
    return Q


def _leading_eigenvector_partition(A_sub, k_sub, m2, resolution=1.0, eps=1e-10):
    """
    For node-set with adjacency A_sub, try to find a bipartition using leading eigenvector.
    Return s vector of +1/-1 (np.array) if improvement > 0, else None.
    """
    # Build modularity matrix for the subgraph: B = A_sub - gamma * k_sub k_sub^T / (2m)
    outer = np.outer(k_sub, k_sub) / m2  # note m2 = 2*m
    B = A_sub - resolution * outer

    # compute largest eigenvalue and eigenvector (symmetric matrix -> use eigh)
    # eigh returns ascending eigenvalues
    vals, vecs = np.linalg.eigh(B)
    largest = vals[-1]
    v = vecs[:, -1]
    if largest <= eps:
        return None  # no positive eigenvalue -> no split
    # partition by sign
    s = np.ones(len(v), dtype=int)
    s[v < 0] = -1

    # compute modularity gain for this split: deltaQ = (1/(4m)) * s^T B s
    delta = (s @ (B @ s)) / (4.0 * (m2/2.0))  # note: m = m2/2
    if delta <= eps:
        return None
    return s


def _recursive_partition(A, nodes, k, m2, resolution, communities):
    """
    Recursively partition the set `nodes` (list/array of node indices).
    communities: list to append final communities (each as list of node indices)
    """
    if len(nodes) == 0:
        return
    if len(nodes) == 1:
        communities.append([nodes[0]])
        return

    A_sub = A[np.ix_(nodes, nodes)]
    k_sub = k[nodes]
    s = _leading_eigenvector_partition(A_sub, k_sub, m2, resolution=resolution)
    if s is None:
        # cannot split -> this set is a final community
        communities.append(list(nodes))
        return
    # split indices
    pos_idx = [nodes[i] for i in range(len(nodes)) if s[i] == 1]
    neg_idx = [nodes[i] for i in range(len(nodes)) if s[i] == -1]
    # If one side empty (shouldn't happen if s has both signs), treat as unsplittable
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        communities.append(list(nodes))
        return
    # recursively split both sides
    _recursive_partition(A, pos_idx, k, m2, resolution, communities)
    _recursive_partition(A, neg_idx, k, m2, resolution, communities)


def spectral_modularity_maximization(adj_matrix, resolution=1.0):
    """
    主函数：谱方法做 modularity maximization（递归使用 leading eigenvector）。
    输入:
      adj_matrix: (n,n) numpy array or array-like; weighted adjacency (assumed undirected)
      resolution: optional gamma (default 1.0)
      gamma: <1 for bigger communities, 1 for standart, >1 for smaller communities
    返回:
      labels: numpy array length n, community id (0..C-1)
      Q: modularity value
    """
    A = np.asarray(adj_matrix, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    # symmetrize
    A = (A + A.T) / 2.0
    n = A.shape[0]
    k = A.sum(axis=1)
    m2 = k.sum()  # equals 2*m
    if m2 == 0:
        # no edges
        return np.arange(n), 0.0

    communities = []
    all_nodes = list(range(n))
    _recursive_partition(A, all_nodes, k, m2, resolution, communities)

    # build labels
    labels = np.empty(n, dtype=int)
    for cid, comm in enumerate(communities):
        for v in comm:
            labels[v] = cid

    Q = modularity(A, labels, resolution=resolution)

    return labels, Q


# --------------------------
# Example usage / quick test
if __name__ == "__main__":
    # small synthetic test
    np.random.seed(0)
    n = 12
    # create a toy graph: two clusters with stronger intra-cluster weights
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if (i < n//2 and j < n//2) or (i >= n//2 and j >= n//2):
                w = np.random.uniform(0.6, 1.0)  # stronger within
            else:
                w = np.random.uniform(0.0, 0.4)  # weaker between
            A[i, j] = w
            A[j, i] = w
    labels, Q = spectral_modularity_maximization(A)
    print("labels:", labels)
    print("modularity Q:", Q)
