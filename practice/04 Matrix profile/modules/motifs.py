import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    mp = matrix_profile['mp']
    mpi = matrix_profile['mpi']
    m = matrix_profile['m']
    excl_zone = matrix_profile['excl_zone']

    sorted_indices = np.argsort(mp)

    filtered_indices = []
    for idx in sorted_indices:
        if all(abs(idx - other_idx) > excl_zone for other_idx in filtered_indices):
            filtered_indices.append(idx)

    for idx in filtered_indices[:top_k]:
        left_idx = idx
        right_idx = mpi[idx]
        motifs_idx.append((left_idx, right_idx))
        motifs_dist.append(mp[idx])

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }
