import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    # INSERT YOUR CODE

    mp = matrix_profile['mp']
    mpi = matrix_profile['mpi']
    excl_zone = matrix_profile['excl_zone']

    sorted_indices = np.argsort(mp)[::-1]

    for idx in sorted_indices:
        if len(discords_idx) >= top_k:
            break
        if mp[idx] != np.inf:
            
            discords_idx.append(idx)
            discords_dist.append(mp[idx])
            mp = apply_exclusion_zone(mp, idx, excl_zone, np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
